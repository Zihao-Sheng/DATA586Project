from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox  # noqa: E402

from app.custom_models_canvas import CustomModelCanvasWidget  # noqa: E402
from app.training_gui import CustomModelsWorkspaceWidget  # noqa: E402
from core import custom_model_generator  # noqa: E402
from core.model_registry import load_model_module, resolve_model_spec_path  # noqa: E402


PREFIX = "zz_relpass"


class Failure(RuntimeError):
    pass


class DialogPatch:
    def __init__(self) -> None:
        self._originals: dict[tuple[object, str], object] = {}

    def set_static(self, owner: object, name: str, func: Callable) -> None:
        key = (owner, name)
        if key not in self._originals:
            self._originals[key] = getattr(owner, name)
        setattr(owner, name, staticmethod(func))

    def restore(self) -> None:
        for (owner, name), value in self._originals.items():
            setattr(owner, name, value)
        self._originals.clear()


def ensure_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def dispose_widget(widget) -> None:
    if widget is None:
        return
    try:
        widget.close()
    except Exception:
        pass
    try:
        widget.deleteLater()
    except Exception:
        pass
    app = QApplication.instance()
    if app is not None:
        try:
            app.processEvents()
        except Exception:
            pass


def safe_temp_paths(model_name: str) -> list[Path]:
    return [
        custom_model_generator.MODEL_DIR / f"{model_name}.py",
        custom_model_generator.SPEC_DIR / f"{model_name}.json",
    ]


def cleanup_temp_artifacts() -> None:
    for directory in (custom_model_generator.MODEL_DIR, custom_model_generator.SPEC_DIR):
        if not directory.is_dir():
            continue
        for path in directory.glob(f"{PREFIX}*"):
            if path.is_file():
                path.unlink()


def spec_dict(spec: custom_model_generator.CustomModelSpec) -> dict:
    return custom_model_generator.spec_to_dict(spec)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def assert_equal(left, right, message: str) -> None:
    if left != right:
        raise Failure(f"{message}\nleft={left!r}\nright={right!r}")


def generated_metadata(model_name: str) -> dict:
    module = load_model_module(model_name)
    metadata = module.get_model_metadata() if hasattr(module, "get_model_metadata") else {}
    if not isinstance(metadata, dict):
        raise Failure(f"Generated model '{model_name}' returned invalid metadata: {type(metadata)!r}")
    return metadata


def verify_generated_outputs(spec: custom_model_generator.CustomModelSpec) -> None:
    model_path = custom_model_generator.MODEL_DIR / f"{spec.model_name}.py"
    spec_path = custom_model_generator.SPEC_DIR / f"{spec.model_name}.json"
    assert_true(model_path.is_file(), f"Expected generated model file missing: {model_path}")
    assert_true(spec_path.is_file(), f"Expected generated spec file missing: {spec_path}")
    assert_equal(read_json(spec_path), spec_dict(spec), f"Generated spec content drifted for '{spec.model_name}'")
    metadata = generated_metadata(spec.model_name)
    assert_equal(metadata.get("model_name"), spec.model_name, f"Generated metadata model_name mismatch for '{spec.model_name}'")
    assert_equal(metadata.get("spec_file"), f"model_specs/{spec.model_name}.json", f"Generated metadata spec_file mismatch for '{spec.model_name}'")
    assert_equal(
        metadata.get("source_spec_file"),
        f"model_specs/{spec.model_name}.json",
        f"Generated metadata source_spec_file mismatch for '{spec.model_name}'",
    )


def prepare_message_box_patches(patch: DialogPatch, *, overwrite_response=QMessageBox.Yes) -> None:
    patch.set_static(QMessageBox, "question", lambda *args, **kwargs: overwrite_response)
    patch.set_static(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)
    patch.set_static(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.Ok)
    patch.set_static(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.Ok)


def build_case_name(group: str, index: int, base_model: str, method_type: str) -> str:
    return f"{PREFIX}_{group}_{index:02d}_{base_model}_{method_type}"


def set_workspace_for_case(
    widget: CustomModelsWorkspaceWidget,
    *,
    model_name: str,
    base_model: str,
    method_type: str,
) -> custom_model_generator.CustomModelSpec:
    widget.new_spec()
    widget.model_name_edit.setText(model_name)
    widget.base_model_combo.setCurrentText(base_model)
    widget.method_combo.setCurrentText(method_type)
    return widget._collect_spec_from_form()


def test_generator_matrix(results: list[str]) -> None:
    index = 0
    for base_model in custom_model_generator.list_supported_base_models():
        supported_methods = custom_model_generator.supported_methods_for_base(base_model)
        for method_type in supported_methods:
            model_name = build_case_name("gen_matrix", index, base_model, method_type)
            pretrained = index % 2 == 0
            spec = custom_model_generator.build_preset_spec(model_name=model_name, base_model=base_model, method_type=method_type)
            payload = spec_dict(spec)
            payload["pretrained"] = pretrained
            spec = custom_model_generator.spec_from_dict(payload)
            custom_model_generator.generate_custom_model(spec, overwrite=True)
            custom_model_generator.save_spec_file(spec)
            verify_generated_outputs(spec)
            results.append(f"generator_matrix:{base_model}:{method_type}:{'pt' if pretrained else 'scratch'}")
            index += 1


def test_workspace_generation_matrix(results: list[str]) -> None:
    patch = DialogPatch()
    prepare_message_box_patches(patch)
    widget = None
    try:
        widget = CustomModelsWorkspaceWidget()
        index = 0
        for base_model in custom_model_generator.list_supported_base_models():
            supported_methods = custom_model_generator.supported_methods_for_base(base_model)
            for method_type in supported_methods:
                model_name = build_case_name("ws_matrix", index, base_model, method_type)
                cleanup_temp_artifacts()
                spec = set_workspace_for_case(
                    widget,
                    model_name=model_name,
                    base_model=base_model,
                    method_type=method_type,
                )
                widget.generate_model()
                verify_generated_outputs(spec)
                results.append(f"workspace_matrix:{base_model}:{method_type}")
                index += 1
    finally:
        patch.restore()


def test_workspace_identity_sequences(results: list[str]) -> None:
    patch = DialogPatch()
    prepare_message_box_patches(patch)
    widget = None
    try:
        widget = CustomModelsWorkspaceWidget()

        old_name = f"{PREFIX}_ws_identity_old"
        new_name = f"{PREFIX}_ws_identity_new"
        save_as_name = f"{PREFIX}_ws_saveas_model"
        for name in (old_name, new_name, save_as_name):
            for path in safe_temp_paths(name):
                if path.exists():
                    path.unlink()

        old_spec = custom_model_generator.build_preset_spec(model_name=old_name, base_model="resnet18", method_type="lora")
        old_path = custom_model_generator.save_spec_file(old_spec)
        old_before = old_path.read_text(encoding="utf-8")

        widget._spec_path = old_path
        widget._apply_spec_to_form(old_spec)
        widget.model_name_edit.setText(new_name)
        renamed_spec = widget._collect_spec_from_form()
        widget.generate_model()
        assert_equal(old_path.read_text(encoding="utf-8"), old_before, "Workspace generate mutated the loaded source spec after renaming.")
        assert_equal(widget._spec_path, custom_model_generator.SPEC_DIR / f"{new_name}.json", "Workspace did not retarget spec path after rename+generate.")
        verify_generated_outputs(renamed_spec)
        results.append("workspace:load_spec_rename_generate_isolated")

        widget.new_spec()
        widget.model_name_edit.setText(save_as_name)
        widget.base_model_combo.setCurrentText("efficientnet_v2_s")
        widget.method_combo.setCurrentText("bn_last2")
        save_as_target = custom_model_generator.SPEC_DIR / f"{PREFIX}_mismatched_filename.json"
        patch.set_static(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(save_as_target), "Spec JSON (*.json)"))
        expected_save_as = custom_model_generator.SPEC_DIR / f"{save_as_name}.json"
        save_as_spec = widget._collect_spec_from_form()
        widget.save_spec_as()
        assert_equal(widget._spec_path, expected_save_as, "Workspace Save As did not canonicalize spec filename to model name.")
        assert_equal(read_json(expected_save_as), spec_dict(save_as_spec), "Workspace Save As wrote incorrect spec content.")
        widget.generate_model()
        verify_generated_outputs(save_as_spec)
        results.append("workspace:new_blank_save_as_generate")

        widget._spec_path = old_path
        widget._apply_spec_to_form(old_spec)
        widget.new_spec()
        assert_true(widget._spec_path is None, "Workspace New Blank did not clear loaded spec identity.")
        fresh_spec = widget._collect_spec_from_form()
        assert_equal(fresh_spec.model_name, "efficientnet_custom_baseline", "Workspace New Blank did not restore default spec state.")
        results.append("workspace:new_blank_clears_loaded_identity")
    finally:
        patch.restore()


def open_canvas_from_spec(widget: CustomModelCanvasWidget, spec_path: Path) -> custom_model_generator.CustomModelSpec:
    spec = custom_model_generator.load_spec_file(spec_path)
    widget._spec_path = spec_path
    widget.spec_path_label.setText(f"Spec File: {spec_path}")
    widget._set_structure_origin("explicit", "high")
    widget._set_editing_enabled(True)
    widget._apply_spec(spec)
    return spec


def test_canvas_identity_and_state(results: list[str]) -> None:
    patch = DialogPatch()
    prepare_message_box_patches(patch)
    widget = None
    try:
        widget = CustomModelCanvasWidget()

        old_name = f"{PREFIX}_canvas_old"
        new_name = f"{PREFIX}_canvas_new"
        for name in (old_name, new_name):
            for path in safe_temp_paths(name):
                if path.exists():
                    path.unlink()

        old_spec = custom_model_generator.build_preset_spec(model_name=old_name, base_model="efficientnet_v2_s", method_type="bn_last2")
        old_path = custom_model_generator.save_spec_file(old_spec)
        old_before = old_path.read_text(encoding="utf-8")
        open_canvas_from_spec(widget, old_path)
        widget.model_name_edit.setText(new_name)
        renamed_spec = widget._derive_spec()
        widget.generate_model()
        assert_equal(old_path.read_text(encoding="utf-8"), old_before, "Canvas generate mutated the loaded source spec after renaming.")
        assert_equal(widget._spec_path, custom_model_generator.SPEC_DIR / f"{new_name}.json", "Canvas did not retarget spec path after rename+generate.")
        verify_generated_outputs(renamed_spec)
        results.append("canvas:load_spec_rename_generate_isolated")

        widget.new_spec()
        widget._apply_strategy("classifier", "DoRA")
        widget.base_model_combo.setCurrentText("resnet18")
        switched_spec = widget._derive_spec()
        assert_equal(switched_spec.base_model, "resnet18", "Canvas base switch failed.")
        assert_equal(switched_spec.method_type, "baseline", "Canvas base switch leaked prior DoRA method state into the new family.")
        assert_equal(switched_spec.peft_targets, {"feature_stages": [], "layer_keys": [], "classifier": False}, "Canvas base switch leaked stale PEFT targets.")
        results.append("canvas:base_switch_clears_stage_state")

        widget.new_spec()
        widget._apply_strategy("features.6", "DoRA")
        widget._apply_strategy("classifier", "DoRA")
        widget.state["features.6"]["lr_override_enabled"] = True
        widget.state["features.6"]["lr_override"] = 5e-4
        stage_spec = widget._derive_spec()
        widget.generate_model()
        verify_generated_outputs(stage_spec)
        reopen_path = custom_model_generator.SPEC_DIR / f"{stage_spec.model_name}.json"
        reopened_spec = open_canvas_from_spec(widget, reopen_path)
        assert_equal(spec_dict(reopened_spec), spec_dict(stage_spec), "Canvas round-trip changed spec structure unexpectedly.")
        results.append("canvas:stage_edit_roundtrip")

        legacy_name = "EfficientNet_Baseline"
        mapped_path = resolve_model_spec_path(legacy_name, allow_legacy_mapping=True)
        assert_true(mapped_path is not None, f"Legacy mapping did not resolve for '{legacy_name}'.")
        legacy_before = mapped_path.read_text(encoding="utf-8")
        open_canvas_from_spec(widget, mapped_path)
        legacy_variant_name = f"{PREFIX}_legacy_variant"
        widget.model_name_edit.setText(legacy_variant_name)
        legacy_variant_spec = widget._derive_spec()
        widget.generate_model()
        assert_equal(mapped_path.read_text(encoding="utf-8"), legacy_before, "Canvas variant generation mutated the mapped legacy/generated source spec.")
        verify_generated_outputs(legacy_variant_spec)
        results.append("canvas:legacy_mapping_variant_isolated")
    finally:
        dispose_widget(widget)
        patch.restore()


def test_canvas_open_existing_roundtrip(results: list[str]) -> None:
    patch = DialogPatch()
    prepare_message_box_patches(patch)
    widget = None
    try:
        model_name = f"{PREFIX}_open_existing_src"
        variant_name = f"{PREFIX}_open_existing_variant"
        cleanup_temp_artifacts()
        source_spec = custom_model_generator.build_preset_spec(model_name=model_name, base_model="resnet50", method_type="tsa")
        custom_model_generator.generate_custom_model(source_spec, overwrite=True)
        custom_model_generator.save_spec_file(source_spec)

        widget = CustomModelCanvasWidget()
        display_items, display_to_model = widget._existing_model_choices()
        selected_display = next((label for label, candidate in display_to_model.items() if candidate == model_name), "")
        assert_true(bool(selected_display), f"Canvas open-existing list did not expose generated model '{model_name}'.")
        patch.set_static(QInputDialog, "getItem", lambda *args, **kwargs: (selected_display, True))
        widget.open_existing_model()
        loaded_spec = widget._derive_spec()
        assert_equal(loaded_spec.model_name, model_name, "Canvas open existing did not load the expected generated model.")
        widget.model_name_edit.setText(variant_name)
        variant_spec = widget._derive_spec()
        widget.generate_model()
        verify_generated_outputs(variant_spec)
        results.append("canvas:open_existing_generated_variant")
    finally:
        dispose_widget(widget)
        patch.restore()


def test_canvas_block_inheritance_compat(results: list[str]) -> None:
    patch = DialogPatch()
    prepare_message_box_patches(patch)
    widget = None
    try:
        widget = CustomModelCanvasWidget()

        # Old stage-level spec compatibility: no hierarchy/node_settings fields.
        legacy_stage_name = f"{PREFIX}_old_stage_resnet18"
        for path in safe_temp_paths(legacy_stage_name):
            if path.exists():
                path.unlink()
        legacy_stage_spec = custom_model_generator.build_preset_spec(
            model_name=legacy_stage_name,
            base_model="resnet18",
            method_type="baseline",
        )
        legacy_payload = spec_dict(legacy_stage_spec)
        legacy_payload.pop("hierarchy_mode", None)
        legacy_payload.pop("node_settings", None)
        legacy_payload.pop("node_lr_overrides", None)
        legacy_stage_spec = custom_model_generator.spec_from_dict(legacy_payload)
        legacy_stage_path = custom_model_generator.save_spec_file(legacy_stage_spec)
        open_canvas_from_spec(widget, legacy_stage_path)
        assert_true("layer1.sub1" in widget.state, "Sub-stage nodes missing when opening old resnet stage-level spec.")
        assert_true(bool(widget.state["layer1.sub1"].get("inherit_from_parent", False)), "Old stage spec did not mark sub-stage as inherited.")
        assert_equal(
            bool(widget.state["layer1.sub1"].get("frozen", True)),
            bool(widget.state["layer1"].get("frozen", True)),
            "Old stage spec did not propagate stage->sub-stage state.",
        )
        results.append("canvas:old_stage_spec_resnet_block_compat")

        # Stage -> block inheritance with override preservation.
        block_model_name = f"{PREFIX}_block_inherit_resnet50"
        for path in safe_temp_paths(block_model_name):
            if path.exists():
                path.unlink()
        widget.new_spec()
        widget.model_name_edit.setText(block_model_name)
        widget.base_model_combo.setCurrentText("resnet50")

        widget._select_stage("layer1")
        widget.stage_frozen.setChecked(False)
        widget.stage_train_bn.setChecked(True)
        widget._on_detail_changed()
        widget._toggle_stage_expand("layer1")
        assert_true("layer1.sub1" in widget.nodes, "Expanded sub-stage node missing from canvas render tree.")
        assert_true(
            "Frozen" in widget.nodes["layer1.sub1"].state_label.text() or "Unfrozen" in widget.nodes["layer1.sub1"].state_label.text(),
            "Expanded sub-stage node lost visible state text.",
        )
        assert_equal(bool(widget.state["layer1.sub1"]["frozen"]), False, "Stage edit did not propagate to inheriting sub-stage1.")
        assert_equal(bool(widget.state["layer1.sub2"]["train_bn"]), True, "Stage edit did not propagate to inheriting sub-stage2.")

        widget._select_stage("layer1.sub1")
        widget.stage_frozen.setChecked(True)
        widget._on_detail_changed()
        assert_equal(bool(widget.state["layer1.sub1"]["inherit_from_parent"]), False, "Sub-stage override did not break inheritance.")

        widget._select_stage("layer1")
        widget.stage_train_bn.setChecked(False)
        widget._on_detail_changed()
        assert_equal(bool(widget.state["layer1.sub1"]["train_bn"]), True, "Parent edit incorrectly overwrote explicit sub-stage override.")
        assert_equal(bool(widget.state["layer1.sub2"]["train_bn"]), False, "Parent edit did not refresh inheriting sub-stage.")

        block_spec = widget._derive_spec()
        block_payload = spec_dict(block_spec)
        block_node_settings = block_payload.get("node_settings", {})
        assert_true("layer1.sub1" in block_node_settings, "Explicit sub-stage override missing from serialized node_settings.")
        assert_true("layer1.sub2" not in block_node_settings, "Inheriting sub-stage should not be serialized as explicit override.")
        widget.generate_model()
        verify_generated_outputs(block_spec)

        reopen_path = custom_model_generator.SPEC_DIR / f"{block_model_name}.json"
        reopened = open_canvas_from_spec(widget, reopen_path)
        reopened_payload = spec_dict(reopened)
        assert_equal(
            reopened_payload.get("node_settings", {}),
            block_node_settings,
            "Block node_settings changed after save/generate/reopen round-trip.",
        )
        results.append("canvas:block_inheritance_override_roundtrip")
    finally:
        dispose_widget(widget)
        patch.restore()


def test_canvas_substage_metadata(results: list[str]) -> None:
    widget = None
    try:
        widget = CustomModelCanvasWidget()
        for base_model, layout in widget.SUBSTAGE_LAYOUTS.items():
            widget.new_spec()
            widget.base_model_combo.setCurrentText(base_model)
            spec = widget._derive_spec()
            payload = spec_dict(spec)
            assert_equal(payload.get("hierarchy_mode"), "substage", f"{base_model} did not emit substage hierarchy_mode.")
            node_hierarchy = payload.get("node_hierarchy")
            assert_true(isinstance(node_hierarchy, dict) and bool(node_hierarchy), f"{base_model} did not emit node_hierarchy metadata.")
            for parent_key, count in layout.items():
                if int(count) <= 0:
                    continue
                child_key = f"{parent_key}.sub1"
                assert_true(child_key in widget.state, f"{base_model} missing expected sub-stage node '{child_key}'.")
                child_meta = node_hierarchy.get(child_key, {})
                assert_true(isinstance(child_meta, dict), f"{base_model}:{child_key} missing node_hierarchy payload.")
                for field in ("node_kind", "parent_key", "hierarchy_depth", "source_module", "static_group_label", "structure_mapping", "structure_source", "safe_operations"):
                    assert_true(field in child_meta, f"{base_model}:{child_key} missing hierarchy field '{field}'.")
                break
            results.append(f"canvas:substage_metadata:{base_model}")
    finally:
        dispose_widget(widget)


def run() -> list[str]:
    ensure_app()
    cleanup_temp_artifacts()
    results: list[str] = []
    test_generator_matrix(results)
    test_workspace_generation_matrix(results)
    test_workspace_identity_sequences(results)
    test_canvas_identity_and_state(results)
    test_canvas_open_existing_roundtrip(results)
    test_canvas_block_inheritance_compat(results)
    test_canvas_substage_metadata(results)
    cleanup_temp_artifacts()
    return results


def main() -> int:
    try:
        results = run()
    except Failure as exc:
        print(f"FAIL: {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps({"status": "ok", "checks": results, "count": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
