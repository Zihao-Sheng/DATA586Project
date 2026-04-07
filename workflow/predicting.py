from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

from workflow.model_registry import discover_model_names, load_model_module


def normalize_model_name(model_name: str | None) -> str | None:
    if not isinstance(model_name, str):
        return None
    normalized = model_name.strip().lower()
    if not normalized:
        return None
    for available_model in discover_model_names():
        if available_model.lower() == normalized:
            return available_model
    return None


def guess_model_name_from_checkpoint_path(checkpoint_path: Path) -> str | None:
    path_text = " ".join([checkpoint_path.name.lower(), checkpoint_path.stem.lower(), checkpoint_path.parent.name.lower()])
    for candidate in discover_model_names():
        if candidate.lower() in path_text:
            return candidate
    return None


def infer_model_name_from_checkpoint(checkpoint_path: Path) -> str | None:
    import torch

    guessed = guess_model_name_from_checkpoint_path(checkpoint_path)
    if guessed is not None:
        return guessed
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    direct_name = normalize_model_name(checkpoint.get("model_name") if isinstance(checkpoint, dict) else None)
    if direct_name is not None:
        return direct_name
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if isinstance(state_dict, dict):
        keys = [str(key) for key in state_dict.keys()]
        if any(key.startswith("features.") for key in keys):
            for candidate in discover_model_names():
                if "efficientnet" in candidate:
                    return candidate
        if any(key.startswith("layer1.") or key.startswith("conv1.") for key in keys):
            for candidate in discover_model_names():
                if "resnet18" in candidate:
                    return candidate
    return guess_model_name_from_checkpoint_path(checkpoint_path)


def load_model(checkpoint_path: Path, model_name: str, device: str):
    import torch

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    checkpoint_model_name = normalize_model_name(checkpoint.get("model_name") if isinstance(checkpoint, dict) else None)
    requested_model_name = normalize_model_name(model_name)
    if checkpoint_model_name is not None and requested_model_name is not None and checkpoint_model_name != requested_model_name:
        raise ValueError(
            f"Checkpoint model mismatch: checkpoint is '{checkpoint_model_name}', but requested '{requested_model_name}'."
        )
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = checkpoint["num_classes"]
    model_module = load_model_module(model_name)
    model = model_module.build_model(num_classes=num_classes, freeze_backbone=False, device=resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_to_idx


def build_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])


def supported_image_extensions() -> tuple[str, ...]:
    return (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class ImagePathDataset:
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        from PIL import Image
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, str(image_path)


def predict_images_batch(
    model,
    image_paths: list[Path],
    transform,
    idx_to_class: dict[int, str],
    device: str,
    batch_size: int = 16,
    num_workers: int = 0,
    progress_callback=None,
) -> list[dict[str, str | float]]:
    import torch
    from torch.utils.data import DataLoader

    dataset = ImagePathDataset(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    results: list[dict[str, str | float]] = []
    processed = 0
    total = len(dataset)
    with torch.no_grad():
        for tensors, batch_paths in dataloader:
            tensors = tensors.to(device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1)
            pred_indices = torch.argmax(probs, dim=1)
            for batch_index, image_path in enumerate(batch_paths):
                pred_idx = int(pred_indices[batch_index].item())
                results.append(
                    {
                        "image_path": image_path,
                        "predicted_class": idx_to_class[pred_idx],
                        "confidence": float(probs[batch_index, pred_idx].item()),
                    }
                )
            processed += len(batch_paths)
            if progress_callback is not None:
                progress_callback(processed, total)
    return results


def _image_tag_for_pil(image) -> str:
    buffer = io.BytesIO()
    image.thumbnail((320, 320))
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"<img src='data:image/jpeg;base64,{encoded}' style='max-width:100%;max-height:100%;object-fit:contain;'/>"


def display_gradcam_comparison(*, image_path: Path, model_specs: list[tuple[str, Path]], image_size: int, device: str) -> None:
    from IPython.display import HTML, display
    from PIL import Image

    from workflow.gradcam import build_heatmap, find_last_conv_layer, overlay_heatmap_on_image

    resolved_device = device
    if device == "auto":
        import torch
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_image = image_path.expanduser().resolve()
    original = Image.open(resolved_image).convert("RGB")
    cards = [("Original", _image_tag_for_pil(original))]
    for model_name, checkpoint_path in model_specs:
        model, _ = load_model(checkpoint_path.expanduser().resolve(), model_name, resolved_device)
        model.eval()
        target_layer = find_last_conv_layer(model)
        if target_layer is None:
            cards.append((model_name, _image_tag_for_pil(original)))
            continue
        activations = {}
        gradients = {}

        def forward_hook(module, inputs, output):
            activations["value"] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        try:
            import torch

            tensor = build_transform(image_size)(original).unsqueeze(0).to(resolved_device)
            output = model(tensor)
            pred_index = int(output.argmax(dim=1).item())
            model.zero_grad(set_to_none=True)
            output[:, pred_index].sum().backward()
            if "value" in activations and "value" in gradients:
                heatmap = build_heatmap(activations["value"], gradients["value"])
                overlay = overlay_heatmap_on_image(original, heatmap)
                cards.append((model_name, _image_tag_for_pil(overlay)))
            else:
                cards.append((model_name, _image_tag_for_pil(original)))
        finally:
            forward_handle.remove()
            backward_handle.remove()
    html = "<div style=\"display:flex;gap:14px;flex-wrap:wrap;\">" + "".join(
        f"<div style='width:220px;'><div style='font-weight:700;margin-bottom:8px;color:#0f172a;'>{escape(label)}</div>{image_html}</div>"
        for label, image_html in cards
    ) + "</div>"
    display(HTML(html))
