from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PySide6.QtGui import QPixmap


def render_gradcam_overlay(
    *,
    image_path: Path,
    checkpoint_path: Path,
    model_name: str,
    image_size: int,
    device: str,
) -> QPixmap:
    return pixmap_from_png_bytes(
        render_gradcam_overlay_bytes(
            image_path=image_path,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            image_size=image_size,
            device=device,
        )
    )


def render_gradcam_overlay_bytes(
    *,
    image_path: Path,
    checkpoint_path: Path,
    model_name: str,
    image_size: int,
    device: str,
) -> bytes:
    overlay_image = render_gradcam_overlay_image(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        image_size=image_size,
        device=device,
    )
    return png_bytes_from_pil_image(overlay_image)


def render_gradcam_overlay_bytes_with_diagnostics(
    *,
    image_path: Path,
    checkpoint_path: Path,
    model_name: str,
    image_size: int,
    device: str,
) -> tuple[bytes, str | None]:
    overlay_image, diagnostic_reason = render_gradcam_overlay_image_with_diagnostics(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        image_size=image_size,
        device=device,
    )
    return png_bytes_from_pil_image(overlay_image), diagnostic_reason


def render_gradcam_overlay_image(
    *,
    image_path: Path,
    checkpoint_path: Path,
    model_name: str,
    image_size: int,
    device: str,
):
    image, _diagnostic_reason = render_gradcam_overlay_image_with_diagnostics(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        image_size=image_size,
        device=device,
    )
    return image


def render_gradcam_overlay_image_with_diagnostics(
    *,
    image_path: Path,
    checkpoint_path: Path,
    model_name: str,
    image_size: int,
    device: str,
):
    import torch
    from PIL import Image

    from pipeline.predicting import build_transform, load_model

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint_path.expanduser().resolve(), model_name, resolved_device)
    model.eval()
    image = Image.open(image_path).convert("RGB")
    target_candidates = list(iter_gradcam_target_candidates(model, model_name))
    if not target_candidates:
        return image, "No conv layer found"

    transform = build_transform(image_size)
    tensor = transform(image).unsqueeze(0).to(resolved_device)

    last_reason = "No gradients captured"
    for _candidate_name, target_layer in target_candidates:
        overlay_image, diagnostic_reason = try_render_gradcam_for_target_layer(
            model=model,
            tensor=tensor,
            image=image,
            target_layer=target_layer,
        )
        if diagnostic_reason is None:
            return overlay_image, None
        last_reason = diagnostic_reason

    return image, last_reason


def try_render_gradcam_for_target_layer(*, model, tensor, image, target_layer):
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        tensor_value = first_tensor_from_output(output)
        if tensor_value is not None:
            activations["value"] = tensor_value.detach()

    def backward_hook(module, grad_input, grad_output):
        tensor_value = first_tensor_from_output(grad_output)
        if tensor_value is not None:
            gradients["value"] = tensor_value.detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        output = model(tensor)
        pred_index = int(output.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        output[:, pred_index].sum().backward()
        if "value" not in activations or "value" not in gradients:
            return image, "No gradients captured"
        heatmap = build_heatmap(activations["value"], gradients["value"])
        return overlay_heatmap_on_image(image, heatmap), None
    finally:
        forward_handle.remove()
        backward_handle.remove()


def first_tensor_from_output(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor_value = first_tensor_from_output(item)
            if tensor_value is not None:
                return tensor_value
    return None


def iter_gradcam_target_candidates(model, model_name: str | None = None):
    import torch.nn as nn

    seen: set[int] = set()

    def add_candidate(module):
        if not isinstance(module, nn.Module):
            return
        module_id = id(module)
        if module_id in seen:
            return
        seen.add(module_id)
        yield module

    if hasattr(model, "features") and isinstance(getattr(model, "features"), nn.Module):
        features = getattr(model, "features")
        children = list(features.children())
        for child in reversed(children):
            if hasattr(child, "base_module") and isinstance(getattr(child, "base_module"), nn.Module):
                base_module = getattr(child, "base_module")
                for module in add_candidate(base_module):
                    yield ("features.base_module", module)
                conv = find_last_conv_layer(base_module)
                if conv is not None:
                    for module in add_candidate(conv):
                        yield ("features.base_module.conv", module)
            for module in add_candidate(child):
                yield ("features.block", module)
            conv = find_last_conv_layer(child)
            if conv is not None:
                for module in add_candidate(conv):
                    yield ("features.block.conv", module)

    if hasattr(model, "layer4") and isinstance(getattr(model, "layer4"), nn.Module):
        layer4 = getattr(model, "layer4")
        for child in reversed(list(layer4.children())):
            if hasattr(child, "base_module") and isinstance(getattr(child, "base_module"), nn.Module):
                base_module = getattr(child, "base_module")
                for module in add_candidate(base_module):
                    yield ("layer4.base_module", module)
                conv = find_last_conv_layer(base_module)
                if conv is not None:
                    for module in add_candidate(conv):
                        yield ("layer4.base_module.conv", module)
            for module in add_candidate(child):
                yield ("layer4.block", module)
            conv = find_last_conv_layer(child)
            if conv is not None:
                for module in add_candidate(conv):
                    yield ("layer4.block.conv", module)

    named_modules = list(model.named_modules())
    for name, module in reversed(named_modules):
        if not name:
            continue
        lowered = name.lower()
        if any(skip in lowered for skip in ("classifier", ".adapter", "adapter.", "fc", "head")):
            continue
        if isinstance(module, nn.Conv2d):
            for candidate in add_candidate(module):
                yield (name, candidate)


def find_last_conv_layer(model):
    import torch.nn as nn

    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    return None


def build_heatmap(activations, gradients) -> np.ndarray:
    import torch

    pooled_gradients = gradients.mean(dim=(2, 3), keepdim=True)
    weighted = activations * pooled_gradients
    heatmap = weighted.sum(dim=1).squeeze(0)
    heatmap = torch.relu(heatmap)
    if float(heatmap.max().item()) > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap.detach().cpu().numpy()


def overlay_heatmap_on_image(image, heatmap: np.ndarray):
    from PIL import Image

    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0), mode="L").resize(image.size)
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    emphasis = np.clip(heatmap_array ** 0.55, 0.0, 1.0)
    color = np.zeros((*heatmap_array.shape, 3), dtype=np.float32)
    color[..., 0] = np.clip(0.10 + emphasis * 1.10, 0.0, 1.0)
    color[..., 1] = np.clip((emphasis - 0.10) / 0.50, 0.0, 1.0) * 0.90
    color[..., 2] = np.clip((emphasis - 0.70) / 0.25, 0.0, 1.0) * 0.40
    alpha = np.clip(0.12 + emphasis * 0.58, 0.0, 0.70)[..., None]
    boosted_image = np.clip(image_array * 1.08, 0.0, 1.0)
    overlay = np.clip(boosted_image * (1.0 - alpha) + color * alpha, 0.0, 1.0)
    return Image.fromarray(np.uint8(overlay * 255.0))


def pixmap_from_pil_image(image) -> QPixmap:
    return pixmap_from_png_bytes(png_bytes_from_pil_image(image))


def png_bytes_from_pil_image(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def pixmap_from_png_bytes(data: bytes) -> QPixmap:
    pixmap = QPixmap()
    pixmap.loadFromData(data, "PNG")
    return pixmap
