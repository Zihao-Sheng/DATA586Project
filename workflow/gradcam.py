from __future__ import annotations

import numpy as np


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
