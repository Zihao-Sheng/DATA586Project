from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path


def predict_and_display_compact(
    *,
    image_paths: list[Path],
    checkpoint_path: Path,
    model_name: str,
    image_size: int,
    device: str,
):
    import torch
    from IPython.display import HTML, display
    from PIL import Image
    from tqdm.auto import tqdm

    from pipeline.predicting import build_transform, load_model, predict_images_batch

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_checkpoint = checkpoint_path.expanduser().resolve()
    resolved_images = [path.expanduser().resolve() for path in image_paths]

    model, class_to_idx = load_model(resolved_checkpoint, model_name, resolved_device)
    transform = build_transform(image_size)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    progress = tqdm(total=len(resolved_images), desc="Predict", unit="image")
    results = predict_images_batch(
        model,
        resolved_images,
        transform,
        idx_to_class,
        resolved_device,
        batch_size=16,
        progress_callback=lambda processed, total: progress.update(max(processed - progress.n, 0)),
    )
    progress.close()

    normalized_results: list[dict[str, object]] = []
    for result in results:
        resolved_image = Path(str(result["image_path"])).resolve()
        actual_label = resolved_image.parent.name if resolved_image.parent.name in class_to_idx else None
        normalized_results.append(
            {
                **result,
                "image_path": resolved_image,
                "actual_label": actual_label,
                "is_correct": None if actual_label is None else result["predicted_class"] == actual_label,
            }
        )

    display(HTML(_build_compact_html(normalized_results)))
    return normalized_results


def compare_models_and_display_compact(
    *,
    image_paths: list[Path],
    model_specs: list[tuple[str, Path]],
    image_size: int,
    device: str,
):
    import torch
    from IPython.display import HTML, display
    from tqdm.auto import tqdm

    from pipeline.predicting import build_transform, load_model, predict_images_batch

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_images = [path.expanduser().resolve() for path in image_paths]
    transform = build_transform(image_size)
    combined: dict[str, dict[str, object]] = {
        str(path): {"image_path": path, "comparisons": {}, "actual_label": None}
        for path in resolved_images
    }

    progress = tqdm(total=len(resolved_images) * max(len(model_specs), 1), desc="Compare", unit="step")
    for model_name, checkpoint_path in model_specs:
        resolved_checkpoint = checkpoint_path.expanduser().resolve()
        model, class_to_idx = load_model(resolved_checkpoint, model_name, resolved_device)
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}
        batch_results = predict_images_batch(
            model,
            resolved_images,
            transform,
            idx_to_class,
            resolved_device,
            batch_size=16,
        )
        for result in batch_results:
            resolved_image = Path(str(result["image_path"])).resolve()
            actual_label = resolved_image.parent.name if resolved_image.parent.name in class_to_idx else None
            row = combined[str(resolved_image)]
            row["actual_label"] = actual_label
            comparisons = row["comparisons"]
            assert isinstance(comparisons, dict)
            comparisons[model_name] = {
                **result,
                "checkpoint_path": str(resolved_checkpoint),
                "actual_label": actual_label,
                "is_correct": None if actual_label is None else result["predicted_class"] == actual_label,
            }
        progress.update(len(resolved_images))
    progress.close()

    results = [combined[str(path)] for path in resolved_images]
    display(HTML(_build_compare_compact_html(results, [name for name, _ in model_specs])))
    return results


def display_gradcam_comparison(
    *,
    image_path: Path,
    model_specs: list[tuple[str, Path]],
    image_size: int,
    device: str,
):
    from IPython.display import HTML, display

    from core.gradcam import overlay_heatmap_on_image, build_heatmap, find_last_conv_layer
    from pipeline.predicting import build_transform, load_model
    import torch
    from PIL import Image

    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_image = image_path.expanduser().resolve()
    original = Image.open(resolved_image).convert("RGB")
    actual_label = resolved_image.parent.name.replace("_", " ").title() if resolved_image.parent.name else "Unknown"
    cards: list[dict[str, str]] = []
    for model_name, checkpoint_path in model_specs:
        model, class_to_idx = load_model(checkpoint_path.expanduser().resolve(), model_name, resolved_device)
        model.eval()
        target_layer = find_last_conv_layer(model)
        if target_layer is None:
            cards.append(
                {
                    "model_name": model_name,
                    "image_html": _image_tag_for_pil(original),
                    "prediction_label": "Grad-CAM unavailable",
                    "confidence_label": "",
                }
            )
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
            tensor = build_transform(image_size)(original).unsqueeze(0).to(resolved_device)
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_index = int(output.argmax(dim=1).item())
            idx_to_class = {idx: name for name, idx in class_to_idx.items()}
            predicted_class = idx_to_class.get(pred_index, str(pred_index))
            confidence = float(probabilities[0, pred_index].item())
            model.zero_grad(set_to_none=True)
            output[:, pred_index].sum().backward()
            if "value" in activations and "value" in gradients:
                heatmap = build_heatmap(activations["value"], gradients["value"])
                overlay = overlay_heatmap_on_image(original, heatmap)
                cards.append(
                    {
                        "model_name": model_name,
                        "image_html": _image_tag_for_pil(overlay),
                        "prediction_label": predicted_class.replace("_", " ").title(),
                        "confidence_label": f"Confidence {confidence:.1%}",
                    }
                )
            else:
                cards.append(
                    {
                        "model_name": model_name,
                        "image_html": _image_tag_for_pil(original),
                        "prediction_label": predicted_class.replace("_", " ").title(),
                        "confidence_label": f"Confidence {confidence:.1%}",
                    }
                )
        finally:
            forward_handle.remove()
            backward_handle.remove()
    original_card = f"""
    <div style="display:flex;justify-content:center;margin-bottom:24px;">
      <div style="width:min(360px, 100%);">
        <div style="font-weight:800;font-size:18px;line-height:1.3;margin-bottom:10px;color:#f8fafc;background:linear-gradient(135deg,#0f172a,#334155);padding:10px 14px;border-radius:12px;text-align:center;letter-spacing:.02em;">
          Original Image
        </div>
        <div style="border:1px solid #475569;border-radius:16px;padding:14px;background:#111827;box-shadow:0 10px 30px rgba(15,23,42,.28);">
          {_image_tag_for_pil(original)}
          <div style="margin-top:12px;text-align:center;">
            <div style="display:inline-block;padding:6px 12px;border-radius:999px;background:#f8fafc;color:#111827;font-weight:800;font-size:14px;box-shadow:0 2px 8px rgba(15,23,42,.15);">
              Food: {escape(actual_label)}
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    comparison_cards = "".join(
        f"""
        <div style="border:1px solid #334155;border-radius:16px;padding:12px;background:#0f172a;box-shadow:0 10px 26px rgba(15,23,42,.24);">
          <div style="font-weight:800;font-size:16px;line-height:1.3;margin-bottom:10px;color:#ffffff;background:linear-gradient(135deg,#1e293b,#475569);padding:10px 12px;border-radius:12px;text-align:center;text-shadow:0 1px 2px rgba(0,0,0,.55);">
            {escape(card["model_name"])}
          </div>
          {card["image_html"]}
          <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px;align-items:center;">
            <div style="display:inline-block;padding:6px 12px;border-radius:999px;background:#f8fafc;color:#111827;font-weight:800;font-size:14px;box-shadow:0 2px 8px rgba(15,23,42,.18);text-align:center;">
              Pred: {escape(card["prediction_label"])}
            </div>
            {"<div style='display:inline-block;padding:5px 10px;border-radius:999px;background:#dbeafe;color:#0f172a;font-weight:700;font-size:13px;box-shadow:0 2px 8px rgba(15,23,42,.14);text-align:center;'>" + escape(card["confidence_label"]) + "</div>" if card["confidence_label"] else ""}
          </div>
        </div>
        """
        for card in cards
    )
    html = f"""
    <div style="max-width:1400px;">
      {original_card}
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;align-items:start;">
        {comparison_cards}
      </div>
    </div>
    """
    display(HTML(html))


def _build_compact_html(results: list[dict[str, object]]) -> str:
    cards: list[str] = []
    for result in results:
        image_path = Path(str(result["image_path"]))
        predicted_class = escape(str(result["predicted_class"]))
        confidence = float(result["confidence"])
        actual_label = result.get("actual_label")
        is_correct = result.get("is_correct")

        if is_correct is True:
            status_text = "Correct"
            status_color = "#1f7a45"
            border_color = "#86efac"
        elif is_correct is False:
            status_text = "Wrong"
            status_color = "#991b1b"
            border_color = "#fca5a5"
        else:
            status_text = "Unknown"
            status_color = "#4b5563"
            border_color = "#cbd5e1"

        actual_text = escape(str(actual_label)) if actual_label is not None else "Unknown"
        image_html = _image_tag_for_path(image_path)
        cards.append(
            f"""
            <div style="border:1px solid {border_color};border-radius:14px;padding:12px;background:#ffffff;box-shadow:0 4px 14px rgba(15,23,42,.06);">
              <div style="aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#f8fafc;border-radius:10px;margin-bottom:10px;">
                {image_html}
              </div>
              <div style="font-weight:700;color:#0f172a;margin-bottom:4px;">{predicted_class}</div>
              <div style="color:#334155;font-size:13px;margin-bottom:4px;">Confidence: {confidence:.2%}</div>
              <div style="color:#475569;font-size:13px;margin-bottom:6px;">Ground Truth: {actual_text}</div>
              <div style="display:inline-block;padding:4px 8px;border-radius:999px;background:{border_color};color:{status_color};font-size:12px;font-weight:700;">
                {status_text}
              </div>
              <div style="margin-top:8px;color:#64748b;font-size:11px;word-break:break-all;">{escape(str(image_path))}</div>
            </div>
            """
        )

    return (
        "<div style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;\">"
        + "".join(cards)
        + "</div>"
    )


def _image_tag_for_path(image_path: Path) -> str:
    try:
        from PIL import Image
    except Exception:
        return f"<div style='color:#94a3b8;font-size:12px;padding:16px;'>Preview unavailable<br>{escape(str(image_path.name))}</div>"

    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((320, 320))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"<img src='data:image/jpeg;base64,{encoded}' style='max-width:100%;max-height:100%;object-fit:contain;'/>"
    except Exception:
        return f"<div style='color:#94a3b8;font-size:12px;padding:16px;'>Could not load<br>{escape(str(image_path.name))}</div>"


def _image_tag_for_pil(image) -> str:
    buffer = io.BytesIO()
    image.thumbnail((320, 320))
    image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"<img src='data:image/jpeg;base64,{encoded}' style='max-width:100%;max-height:100%;object-fit:contain;'/>"


def _build_compare_compact_html(results: list[dict[str, object]], model_names: list[str]) -> str:
    cards: list[str] = []
    for result in results:
        image_path = Path(str(result["image_path"]))
        actual_label = result.get("actual_label")
        actual_text = escape(str(actual_label).replace("_", " ").title()) if actual_label is not None else "Unknown"
        comparisons = result.get("comparisons") if isinstance(result.get("comparisons"), dict) else {}
        lines = []
        for model_name in model_names:
            entry = comparisons.get(model_name)
            if not isinstance(entry, dict):
                continue
            status = entry.get("is_correct")
            status_text = "Correct" if status is True else ("Wrong" if status is False else "Unknown")
            status_bg = "#dcfce7" if status is True else ("#fee2e2" if status is False else "#e2e8f0")
            status_fg = "#166534" if status is True else ("#991b1b" if status is False else "#334155")
            lines.append(
                f"<div style='display:flex;justify-content:space-between;gap:10px;align-items:center;padding:8px 10px;border-radius:10px;background:#f8fafc;border:1px solid #e2e8f0;margin-top:8px;'>"
                f"<div style='min-width:0;'>"
                f"<div style='font-size:12px;font-weight:800;color:#0f172a;'>{escape(model_name)}</div>"
                f"<div style='font-size:12px;color:#334155;'>{escape(str(entry.get('predicted_class', '-')).replace('_', ' ').title())}</div>"
                f"</div>"
                f"<div style='text-align:right;flex-shrink:0;'>"
                f"<div style='font-size:12px;font-weight:700;color:#0f172a;'>{float(entry.get('confidence', 0.0)):.2%}</div>"
                f"<div style='display:inline-block;margin-top:4px;padding:3px 8px;border-radius:999px;background:{status_bg};color:{status_fg};font-size:11px;font-weight:800;'>{status_text}</div>"
                f"</div>"
                f"</div>"
            )
        cards.append(
            f"""
            <div style="border:1px solid #cbd5e1;border-radius:16px;padding:12px;background:#ffffff;box-shadow:0 8px 22px rgba(15,23,42,.08);">
              <div style="aspect-ratio:1/1;display:flex;align-items:center;justify-content:center;overflow:hidden;background:#f8fafc;border-radius:10px;margin-bottom:10px;">
                {_image_tag_for_path(image_path)}
              </div>
              <div style="font-weight:800;color:#0f172a;margin-bottom:4px;">{escape(image_path.name)}</div>
              <div style="display:inline-block;padding:5px 10px;border-radius:999px;background:#eff6ff;color:#0f172a;font-size:12px;font-weight:800;margin-bottom:8px;">Ground Truth: {actual_text}</div>
              {''.join(lines)}
            </div>
            """
        )
    return "<div style=\"display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;\">" + "".join(cards) + "</div>"
