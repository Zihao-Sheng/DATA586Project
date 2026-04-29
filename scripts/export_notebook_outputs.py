import argparse
import copy
import io
import json
import re
from pathlib import Path


def slugify(text: str) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_.-]", "", text)
    return text[:80] or "item"


def find_bonus_start(cells):
    for i, c in enumerate(cells):
        if c.get("cell_type") != "markdown":
            continue
        src = "".join(c.get("source", [])).lower()
        if "## bonus" in src or "bonus follow-up" in src:
            return i
    return len(cells)


def apply_trace_update(fig, trace_update, trace_indices=None):
    if trace_indices is None:
        trace_indices = list(range(len(fig.data)))
    if isinstance(trace_indices, int):
        trace_indices = [trace_indices]
    if not isinstance(trace_update, dict):
        return

    visible_values = trace_update.get("visible")
    other = {k: v for k, v in trace_update.items() if k != "visible"}

    if isinstance(visible_values, list) and len(visible_values) == len(fig.data):
        for i, vis in enumerate(visible_values):
            fig.data[i].visible = vis
    elif isinstance(visible_values, list) and len(visible_values) == len(trace_indices):
        for pos, idx in enumerate(trace_indices):
            fig.data[idx].visible = visible_values[pos]
    elif visible_values is not None:
        for idx in trace_indices:
            fig.data[idx].visible = visible_values

    if other:
        for pos, idx in enumerate(trace_indices):
            per_trace = {}
            for k, v in other.items():
                if isinstance(v, list):
                    if len(v) == len(fig.data):
                        per_trace[k] = v[idx]
                    elif len(v) == len(trace_indices):
                        per_trace[k] = v[pos]
                    elif len(v) == 1 and len(trace_indices) == 1:
                        # Plotly dropdown "update/restyle" for a single trace commonly wraps
                        # real payload as [payload], e.g. z=[matrix], x=[labels].
                        per_trace[k] = v[0]
                    else:
                        per_trace[k] = v
                else:
                    per_trace[k] = v
            fig.data[idx].update(per_trace)


def apply_button_state(fig, button):
    method = button.get("method", "restyle")
    args = button.get("args", []) or []

    if method == "restyle":
        trace_update = args[0] if len(args) > 0 else {}
        trace_indices = args[1] if len(args) > 1 else None
        apply_trace_update(fig, trace_update, trace_indices)
    elif method == "relayout":
        layout_update = args[0] if len(args) > 0 else {}
        if isinstance(layout_update, dict):
            fig.update_layout(layout_update)
    elif method == "update":
        trace_update = args[0] if len(args) > 0 else {}
        layout_update = args[1] if len(args) > 1 else {}
        apply_trace_update(fig, trace_update, None)
        if isinstance(layout_update, dict):
            fig.update_layout(layout_update)


def export_tables_to_png(html_str: str, png_base: Path, summary_tables: list):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return

    try:
        tables = pd.read_html(io.StringIO(html_str))
    except Exception:
        return

    for ti, df in enumerate(tables):
        df = df.fillna("")
        col_labels = [str(c) for c in df.columns]
        cell_text = df.astype(str).values

        fig_w = max(8, min(24, 1.35 * max(1, len(col_labels))))
        fig_h = max(2.5, min(30, 0.32 * (len(df) + 2)))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)

        out_png = png_base if len(tables) == 1 else png_base.with_name(f"{png_base.stem}_{ti:02d}.png")
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        summary_tables.append(str(out_png))


def export_notebook(nb_path: Path, out_dir: Path):
    import plotly.graph_objects as go

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    bonus_start = find_bonus_start(cells)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "graphs"
    table_dir = out_dir / "tables"
    other_dir = out_dir / "other_html"
    plot_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)
    other_dir.mkdir(exist_ok=True)

    summary = {
        "bonus_start_cell": bonus_start,
        "plots": [],
        "dropdown_pages": [],
        "tables": [],
        "other_html": [],
    }

    for ci, cell in enumerate(cells[:bonus_start]):
        if cell.get("cell_type") != "code":
            continue
        for oi, output in enumerate(cell.get("outputs", [])):
            if output.get("output_type") not in ("display_data", "execute_result"):
                continue
            data = output.get("data", {})

            if "text/html" in data:
                html_val = data["text/html"]
                if isinstance(html_val, list):
                    html_str = "".join(html_val)
                else:
                    html_str = str(html_val)
                if "<table" in html_str.lower():
                    p = table_dir / f"cell{ci:02d}_out{oi:02d}_table.html"
                    p.write_text(html_str, encoding="utf-8")
                    summary["tables"].append(str(p))
                    png_base = table_dir / f"cell{ci:02d}_out{oi:02d}_table.png"
                    export_tables_to_png(html_str, png_base, summary["tables"])
                else:
                    p = other_dir / f"cell{ci:02d}_out{oi:02d}.html"
                    p.write_text(html_str, encoding="utf-8")
                    summary["other_html"].append(str(p))

            if "application/vnd.plotly.v1+json" in data:
                fig_json = data["application/vnd.plotly.v1+json"]
                fig = go.Figure(fig_json, skip_invalid=True)
                base = f"cell{ci:02d}_out{oi:02d}"

                html_path = plot_dir / f"{base}.html"
                fig.write_html(str(html_path), include_plotlyjs="cdn")
                summary["plots"].append(str(html_path))

                try:
                    png_path = plot_dir / f"{base}.png"
                    fig.write_image(str(png_path), format="png", scale=2)
                    summary["plots"].append(str(png_path))
                except Exception:
                    pass

                updatemenus = (fig_json.get("layout") or {}).get("updatemenus", [])
                for mi, menu in enumerate(updatemenus):
                    buttons = menu.get("buttons", [])
                    for bi, button in enumerate(buttons):
                        label = slugify(str(button.get("label", f"btn{bi}")))
                        f2 = copy.deepcopy(fig)
                        apply_button_state(f2, button)
                        page_base = f"{base}_menu{mi:02d}_btn{bi:02d}_{label}"

                        page_html = plot_dir / f"{page_base}.html"
                        f2.write_html(str(page_html), include_plotlyjs="cdn")
                        summary["dropdown_pages"].append(str(page_html))

                        try:
                            page_png = plot_dir / f"{page_base}.png"
                            f2.write_image(str(page_png), format="png", scale=2)
                            summary["dropdown_pages"].append(str(page_png))
                        except Exception:
                            pass

    summary_path = out_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export notebook plots/tables (excluding bonus section).")
    parser.add_argument("--notebook", default="notebooks/experiment_analysis.ipynb")
    parser.add_argument("--out", default="exports/experiment_analysis_exports")
    args = parser.parse_args()

    summary_file = export_notebook(Path(args.notebook), Path(args.out))
    print(f"Export done: {summary_file}")
