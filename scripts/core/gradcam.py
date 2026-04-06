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


def render_gradcam_overlay_image(
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
    target_layer = find_last_conv_layer(model)
    if target_layer is None:
        return Image.open(image_path).convert("RGB")

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        image = Image.open(image_path).convert("RGB")
        transform = build_transform(image_size)
        tensor = transform(image).unsqueeze(0).to(resolved_device)
        output = model(tensor)
        pred_index = int(output.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        output[:, pred_index].sum().backward()
        if "value" not in activations or "value" not in gradients:
            return image
        heatmap = build_heatmap(activations["value"], gradients["value"])
        return overlay_heatmap_on_image(image, heatmap)
    finally:
        forward_handle.remove()
        backward_handle.remove()


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

    emphasis = np.clip(heatmap_array ** 0.72, 0.0, 1.0)
    color = np.zeros((*heatmap_array.shape, 3), dtype=np.float32)
    color[..., 0] = np.clip(0.32 + emphasis * 0.68, 0.0, 1.0)
    color[..., 1] = np.clip((emphasis - 0.18) / 0.62, 0.0, 1.0) * 0.96
    color[..., 2] = np.clip((emphasis - 0.84) / 0.16, 0.0, 1.0) * 0.25

    alpha = np.clip(emphasis * 0.36, 0.0, 0.36)[..., None]
    boosted_image = np.clip(image_array * 1.04, 0.0, 1.0)
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
