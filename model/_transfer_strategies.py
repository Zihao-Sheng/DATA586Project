from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


def _normalize_stage_lr_overrides(raw: dict[str, object] | None) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    overrides: dict[str, float] = {}
    for key, value in raw.items():
        stage = str(key).strip()
        if not stage:
            continue
        try:
            lr_value = float(value)
        except Exception:
            continue
        if lr_value > 0:
            overrides[stage] = lr_value
    return overrides


def _normalize_backbone_name(base_model: str | None) -> str | None:
    normalized = str(base_model or "").strip().lower()
    aliases = {
        "efficientnet": "efficientnet_v2_s",
        "resnet": "resnet18",
    }
    canonical = aliases.get(normalized, normalized)
    if canonical in {"resnet18", "resnet50", "efficientnet_v2_s", "convnext_tiny", "mobilenet_v3_large", "densenet121"}:
        return canonical
    return None


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    *,
    optimizer_name: str = "adam",
    base_model: str | None = None,
    stage_lr_overrides: dict[str, object] | None = None,
) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    param_groups: list[dict[str, object]] = [{"params": trainable_params, "lr": float(lr)}]

    normalized_overrides = _normalize_stage_lr_overrides(stage_lr_overrides)
    normalized_backbone = _normalize_backbone_name(base_model)
    if normalized_overrides and normalized_backbone is not None:
        try:
            stage_map = _stage_map_for_backbone(model, normalized_backbone)
        except Exception:
            stage_map = {}
        if stage_map:
            assigned_ids: set[int] = set()
            grouped: list[dict[str, object]] = []
            for stage_key, stage_lr in normalized_overrides.items():
                module = stage_map.get(stage_key)
                if module is None:
                    continue
                params = [param for param in module.parameters() if param.requires_grad and id(param) not in assigned_ids]
                if not params:
                    continue
                for param in params:
                    assigned_ids.add(id(param))
                grouped.append({"params": params, "lr": float(stage_lr), "stage": stage_key})
            default_params = [param for param in trainable_params if id(param) not in assigned_ids]
            if default_params:
                grouped.append({"params": default_params, "lr": float(lr), "stage": "default"})
            if grouped:
                param_groups = grouped

    normalized_optimizer = str(optimizer_name).strip().lower()
    if normalized_optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9)
    if normalized_optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=1e-2)
    return torch.optim.Adam(param_groups)


def _resolved_device(device: str | torch.device) -> str | torch.device:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_resnet18_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except TypeError:
        model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_resnet50_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    except TypeError:
        model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_efficientnet_v2_s_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
    except TypeError:
        model = models.efficientnet_v2_s(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_convnext_tiny_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
    except TypeError:
        model = models.convnext_tiny(pretrained=pretrained)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def load_mobilenet_v3_large_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    except TypeError:
        model = models.mobilenet_v3_large(pretrained=pretrained)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_densenet121_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
    except TypeError:
        model = models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def enable_bn_tuning(model: nn.Module, classifier_modules: list[nn.Module]) -> None:
    freeze_all(model)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            unfreeze_module(module)
    for module in classifier_modules:
        unfreeze_module(module)


def enable_norm_tuning(model: nn.Module, classifier_modules: list[nn.Module]) -> None:
    freeze_all(model)
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    for module in model.modules():
        if isinstance(module, norm_types):
            for name, param in module.named_parameters(recurse=False):
                if name in {"weight", "bias"}:
                    param.requires_grad = True
    for module in classifier_modules:
        unfreeze_module(module)


def enable_bias_tuning(model: nn.Module, *, scope: str = "all_bias", classifier_modules: list[nn.Module] | None = None) -> None:
    freeze_all(model)
    normalized_scope = str(scope).strip().lower() or "all_bias"
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )
    for module in model.modules():
        if normalized_scope == "norm_and_classifier_bias" and not isinstance(module, norm_types):
            continue
        for name, param in module.named_parameters(recurse=False):
            if name == "bias":
                param.requires_grad = True
    if normalized_scope == "norm_and_classifier_bias" and classifier_modules:
        for module in classifier_modules:
            for name, param in module.named_parameters(recurse=False):
                if name == "bias":
                    param.requires_grad = True


def enable_bn_tuning_with_last_feature_stages(
    model: nn.Module,
    features: nn.Sequential | nn.ModuleList,
    classifier_modules: list[nn.Module],
    num_last_stages: int,
) -> None:
    freeze_all(model)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            unfreeze_module(module)
    if num_last_stages > 0:
        for module in list(features.children())[-num_last_stages:]:
            unfreeze_module(module)
    for module in classifier_modules:
        unfreeze_module(module)


class ConvAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck_ratio: float = 0.25, bottleneck_dim: int | None = None) -> None:
        super().__init__()
        bottleneck_channels = max(1, int(bottleneck_dim)) if bottleneck_dim is not None else max(8, int(channels * bottleneck_ratio))
        self.down = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(inputs)))


class AdapterWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, channels: int, *, bottleneck_dim: int | None = None) -> None:
        super().__init__()
        self.base_module = base_module
        self.adapter = ConvAdapter(channels, bottleneck_dim=bottleneck_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.base_module(inputs)
        return outputs + self.adapter(outputs)


class LinearAdapter(nn.Module):
    def __init__(self, features: int, bottleneck_dim: int | None = None) -> None:
        super().__init__()
        hidden = max(1, int(bottleneck_dim)) if bottleneck_dim is not None else max(8, features // 4)
        self.down = nn.Linear(features, hidden, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden, features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(inputs)))


class LinearAdapterWrapper(nn.Module):
    def __init__(self, base_module: nn.Linear, *, bottleneck_dim: int | None = None) -> None:
        super().__init__()
        self.base_module = base_module
        self.adapter = LinearAdapter(base_module.out_features, bottleneck_dim=bottleneck_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.base_module(inputs)
        return outputs + self.adapter(outputs)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        for param in self.base.parameters():
            param.requires_grad = False
        self.scaling = alpha / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_up(self.lora_down(inputs)) * self.scaling


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        for param in self.base.parameters():
            param.requires_grad = False
        self.scaling = alpha / rank
        self.lora_down = nn.Conv2d(base.in_channels, rank, kernel_size=1, stride=base.stride, padding=0, bias=False)
        self.lora_up = nn.Conv2d(rank, base.out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_up(self.lora_down(inputs)) * self.scaling


class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        for param in self.base.parameters():
            param.requires_grad = False
        self.scaling = alpha / rank
        self.dora_down = nn.Linear(base.in_features, rank, bias=False)
        self.dora_up = nn.Linear(rank, base.out_features, bias=False)
        with torch.no_grad():
            magnitude = self.base.weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.magnitude = nn.Parameter(magnitude)
        nn.init.kaiming_uniform_(self.dora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.dora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        adapted_weight = self.base.weight + self.dora_up.weight @ self.dora_down.weight * self.scaling
        direction = adapted_weight / adapted_weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return F.linear(inputs, direction * self.magnitude, self.base.bias)


class DoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        for param in self.base.parameters():
            param.requires_grad = False
        self.scaling = alpha / rank
        self.dora_down = nn.Parameter(torch.empty(rank, base.weight.shape[1], *base.kernel_size))
        self.dora_up = nn.Parameter(torch.empty(base.out_channels, rank))
        with torch.no_grad():
            magnitude = self.base.weight.flatten(1).norm(dim=1).view(base.out_channels, 1, 1, 1).clamp_min(1e-6)
        self.magnitude = nn.Parameter(magnitude)
        nn.init.kaiming_uniform_(self.dora_down, a=5 ** 0.5)
        nn.init.zeros_(self.dora_up)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        delta_weight = torch.einsum("or,rijk->oijk", self.dora_up, self.dora_down) * self.scaling
        adapted_weight = self.base.weight + delta_weight
        direction = adapted_weight / adapted_weight.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
        return F.conv2d(
            inputs,
            direction * self.magnitude,
            self.base.bias,
            self.base.stride,
            self.base.padding,
            self.base.dilation,
            self.base.groups,
        )


class SSFConv2dWrapper(nn.Module):
    def __init__(self, base: nn.Conv2d, *, init_scale: float = 1.0, init_shift: float = 0.0) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        for param in self.base.parameters():
            param.requires_grad = False
        channels = int(base.out_channels)
        self.scale = nn.Parameter(torch.full((channels, 1, 1), float(init_scale)))
        self.shift = nn.Parameter(torch.full((channels, 1, 1), float(init_shift)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.base(inputs)
        return outputs * self.scale + self.shift


class SSFLinearWrapper(nn.Module):
    def __init__(self, base: nn.Linear, *, init_scale: float = 1.0, init_shift: float = 0.0) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        for param in self.base.parameters():
            param.requires_grad = False
        features = int(base.out_features)
        self.scale = nn.Parameter(torch.full((features,), float(init_scale)))
        self.shift = nn.Parameter(torch.full((features,), float(init_shift)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.base(inputs)
        return outputs * self.scale + self.shift


def _replace_child(parent: nn.Module, child_name: str, new_child: nn.Module) -> None:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = new_child
        return
    setattr(parent, child_name, new_child)


def apply_lora_recursively(module: nn.Module) -> None:
    for child_name, child in list(module.named_children()):
        replacement: nn.Module | None = None
        if isinstance(child, nn.Conv2d):
            replacement = LoRAConv2d(child)
        elif isinstance(child, nn.Linear):
            replacement = LoRALinear(child)
        if replacement is not None:
            _replace_child(module, child_name, replacement)
            continue
        apply_lora_recursively(child)


def apply_dora_recursively(module: nn.Module) -> None:
    for child_name, child in list(module.named_children()):
        replacement: nn.Module | None = None
        if isinstance(child, nn.Conv2d):
            replacement = DoRAConv2d(child)
        elif isinstance(child, nn.Linear):
            replacement = DoRALinear(child)
        if replacement is not None:
            _replace_child(module, child_name, replacement)
            continue
        apply_dora_recursively(child)


def apply_ssf_recursively(module: nn.Module, *, max_modules: int = 4, init_scale: float = 1.0, init_shift: float = 0.0) -> int:
    wrapped = 0
    for child_name, child in list(module.named_children()):
        if wrapped >= max_modules:
            break
        replacement: nn.Module | None = None
        if isinstance(child, nn.Conv2d):
            replacement = SSFConv2dWrapper(child, init_scale=init_scale, init_shift=init_shift)
        elif isinstance(child, nn.Linear):
            replacement = SSFLinearWrapper(child, init_scale=init_scale, init_shift=init_shift)
        if replacement is not None:
            _replace_child(module, child_name, replacement)
            wrapped += 1
            continue
        wrapped += apply_ssf_recursively(
            child,
            max_modules=max(0, max_modules - wrapped),
            init_scale=init_scale,
            init_shift=init_shift,
        )
        if wrapped >= max_modules:
            break
    return wrapped


def apply_adapters_recursively(module: nn.Module, *, max_modules: int = 4, bottleneck_dim: int | None = None) -> int:
    wrapped = 0
    for child_name, child in list(module.named_children()):
        if wrapped >= max_modules:
            break
        replacement: nn.Module | None = None
        if isinstance(child, nn.Conv2d):
            replacement = AdapterWrapper(child, channels=child.out_channels, bottleneck_dim=bottleneck_dim)
        elif isinstance(child, nn.Linear):
            replacement = LinearAdapterWrapper(child, bottleneck_dim=bottleneck_dim)
        if replacement is not None:
            _replace_child(module, child_name, replacement)
            wrapped += 1
            continue
        wrapped += apply_adapters_recursively(child, max_modules=max(0, max_modules - wrapped), bottleneck_dim=bottleneck_dim)
        if wrapped >= max_modules:
            break
    return wrapped


def wrap_module_with_adapter(parent: nn.Module, child_name: str, channels: int, *, bottleneck_dim: int | None = None) -> None:
    child = getattr(parent, child_name) if not isinstance(parent, (nn.Sequential, nn.ModuleList)) else parent[int(child_name)]
    _replace_child(parent, child_name, AdapterWrapper(child, channels, bottleneck_dim=bottleneck_dim))


def _finalize_model(model: nn.Module, device: str | torch.device) -> nn.Module:
    model.to(_resolved_device(device))
    return model


def build_resnet18_linear_probe(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_resnet18_classifier(num_classes, pretrained=pretrained)
    freeze_all(model)
    unfreeze_module(model.fc)
    return _finalize_model(model, device)


def build_resnet18_lora(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_resnet18_classifier(num_classes, pretrained=pretrained)
    freeze_all(model)
    apply_lora_recursively(model.layer4)
    model.fc = LoRALinear(model.fc, rank=8, alpha=16.0)
    return _finalize_model(model, device)


def build_resnet18_adapters(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_resnet18_classifier(num_classes, pretrained=pretrained)
    freeze_all(model)
    wrap_module_with_adapter(model.layer3, "0", channels=256)
    wrap_module_with_adapter(model.layer3, "1", channels=256)
    wrap_module_with_adapter(model.layer4, "0", channels=512)
    wrap_module_with_adapter(model.layer4, "1", channels=512)
    unfreeze_module(model.fc)
    return _finalize_model(model, device)


def build_resnet18_bn_tuning(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_resnet18_classifier(num_classes, pretrained=pretrained)
    enable_bn_tuning(model, [model.fc])
    return _finalize_model(model, device)


def build_resnet18_full_finetune(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_resnet18_classifier(num_classes, pretrained=pretrained)
    unfreeze_all(model)
    return _finalize_model(model, device)


def build_efficientnet_linear_probe(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    freeze_all(model)
    unfreeze_module(model.classifier[1])
    return _finalize_model(model, device)


def build_efficientnet_lora(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    freeze_all(model)
    apply_lora_recursively(model.features[6])
    apply_lora_recursively(model.features[7])
    model.classifier[1] = LoRALinear(model.classifier[1], rank=8, alpha=16.0)
    return _finalize_model(model, device)


def build_efficientnet_dora(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    freeze_all(model)
    apply_dora_recursively(model.features[6])
    apply_dora_recursively(model.features[7])
    model.classifier[1] = DoRALinear(model.classifier[1], rank=8, alpha=16.0)
    return _finalize_model(model, device)


def build_efficientnet_adapters(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    freeze_all(model)
    wrap_module_with_adapter(model.features, "5", channels=160)
    wrap_module_with_adapter(model.features, "6", channels=256)
    wrap_module_with_adapter(model.features, "7", channels=1280)
    unfreeze_module(model.classifier[1])
    return _finalize_model(model, device)


def build_efficientnet_bn_tuning(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    enable_bn_tuning(model, [model.classifier[1]])
    return _finalize_model(model, device)


def build_efficientnet_bn_last1(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    enable_bn_tuning_with_last_feature_stages(
        model,
        model.features,
        [model.classifier[1]],
        num_last_stages=1,
    )
    return _finalize_model(model, device)


def build_efficientnet_bn_last2(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    enable_bn_tuning_with_last_feature_stages(
        model,
        model.features,
        [model.classifier[1]],
        num_last_stages=2,
    )
    return _finalize_model(model, device)


def build_efficientnet_full_finetune(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    unfreeze_all(model)
    return _finalize_model(model, device)


def _classifier_module_for_backbone(model: nn.Module, backbone: str) -> nn.Module:
    if backbone in {"resnet18", "resnet50"}:
        return model.fc
    if backbone == "efficientnet_v2_s":
        return model.classifier[1]
    if backbone == "convnext_tiny":
        return model.classifier[2]
    if backbone == "mobilenet_v3_large":
        return model.classifier[3]
    if backbone == "densenet121":
        return model.classifier
    raise ValueError(f"Unsupported backbone for classifier lookup: {backbone}")


def _replace_classifier_with_lora(model: nn.Module, backbone: str) -> None:
    if backbone in {"resnet18", "resnet50"}:
        model.fc = LoRALinear(model.fc, rank=8, alpha=16.0)
        return
    if backbone == "efficientnet_v2_s":
        model.classifier[1] = LoRALinear(model.classifier[1], rank=8, alpha=16.0)
        return
    if backbone == "convnext_tiny":
        model.classifier[2] = LoRALinear(model.classifier[2], rank=8, alpha=16.0)
        return
    if backbone == "mobilenet_v3_large":
        model.classifier[3] = LoRALinear(model.classifier[3], rank=8, alpha=16.0)
        return
    if backbone == "densenet121":
        model.classifier = LoRALinear(model.classifier, rank=8, alpha=16.0)
        return
    raise ValueError(f"Unsupported backbone for LoRA classifier replacement: {backbone}")


def _load_backbone_classifier(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone == "resnet18":
        return load_resnet18_classifier(num_classes, pretrained=pretrained)
    if backbone == "resnet50":
        return load_resnet50_classifier(num_classes, pretrained=pretrained)
    if backbone == "efficientnet_v2_s":
        return load_efficientnet_v2_s_classifier(num_classes, pretrained=pretrained)
    if backbone == "convnext_tiny":
        return load_convnext_tiny_classifier(num_classes, pretrained=pretrained)
    if backbone == "mobilenet_v3_large":
        return load_mobilenet_v3_large_classifier(num_classes, pretrained=pretrained)
    if backbone == "densenet121":
        return load_densenet121_classifier(num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported backbone: {backbone}")


def _stage_map_for_backbone(model: nn.Module, backbone: str) -> dict[str, nn.Module]:
    if backbone in {"resnet18", "resnet50"}:
        return {
            "stem": model.conv1,
            "layer1": model.layer1,
            "layer2": model.layer2,
            "layer3": model.layer3,
            "layer4": model.layer4,
            "classifier": _classifier_module_for_backbone(model, backbone),
        }
    if backbone == "efficientnet_v2_s":
        return {
            "stem": model.features[0],
            **{f"features.{idx}": model.features[idx] for idx in range(len(model.features))},
            "classifier": _classifier_module_for_backbone(model, backbone),
        }
    if backbone == "convnext_tiny":
        return {
            "stem": model.features[0],
            "stage1": model.features[1],
            "stage2": model.features[3],
            "stage3": model.features[5],
            "stage4": model.features[7],
            "classifier": _classifier_module_for_backbone(model, backbone),
        }
    if backbone == "mobilenet_v3_large":
        return {
            "stem": model.features[0],
            "stage1": model.features[3],
            "stage2": model.features[6],
            "stage3": model.features[12],
            "stage4": model.features[16],
            "classifier": _classifier_module_for_backbone(model, backbone),
        }
    if backbone == "densenet121":
        return {
            "stem": model.features.conv0,
            "denseblock1": model.features.denseblock1,
            "denseblock2": model.features.denseblock2,
            "denseblock3": model.features.denseblock3,
            "denseblock4": model.features.denseblock4,
            "classifier": _classifier_module_for_backbone(model, backbone),
        }
    raise ValueError(f"Unsupported backbone for stage map: {backbone}")


def _build_linear_probe_generic(backbone: str, num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    freeze_all(model)
    unfreeze_module(_classifier_module_for_backbone(model, backbone))
    return _finalize_model(model, device)


def _build_bn_tuning_generic(backbone: str, num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    enable_bn_tuning(model, [_classifier_module_for_backbone(model, backbone)])
    return _finalize_model(model, device)


def _build_norm_tuning_generic(backbone: str, num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    enable_norm_tuning(model, [_classifier_module_for_backbone(model, backbone)])
    return _finalize_model(model, device)


def _build_full_finetune_generic(backbone: str, num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    unfreeze_all(model)
    return _finalize_model(model, device)


def _build_lora_generic(
    backbone: str,
    num_classes: int,
    device: str | torch.device,
    pretrained: bool = True,
    target_keys: tuple[str, ...] = (),
) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    freeze_all(model)
    stage_map = _stage_map_for_backbone(model, backbone)
    for key in target_keys:
        module = stage_map.get(key)
        if module is not None:
            apply_lora_recursively(module)
    _replace_classifier_with_lora(model, backbone)
    return _finalize_model(model, device)


def _build_tsa_generic(
    backbone: str,
    num_classes: int,
    device: str | torch.device,
    pretrained: bool = True,
    target_keys: tuple[str, ...] = (),
) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    freeze_all(model)
    stage_map = _stage_map_for_backbone(model, backbone)
    for key in target_keys:
        module = stage_map.get(key)
        if module is not None:
            apply_adapters_recursively(module, max_modules=4)
    unfreeze_module(_classifier_module_for_backbone(model, backbone))
    return _finalize_model(model, device)


def _build_adapter_generic(
    backbone: str,
    num_classes: int,
    device: str | torch.device,
    pretrained: bool = True,
    target_keys: tuple[str, ...] = (),
    bottleneck_dim: int | None = None,
) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    freeze_all(model)
    stage_map = _stage_map_for_backbone(model, backbone)
    for key in target_keys:
        module = stage_map.get(key)
        if module is not None:
            apply_adapters_recursively(module, max_modules=4, bottleneck_dim=bottleneck_dim)
    unfreeze_module(_classifier_module_for_backbone(model, backbone))
    return _finalize_model(model, device)


def _build_bitfit_generic(backbone: str, num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    enable_bias_tuning(model, scope="all_bias")
    return _finalize_model(model, device)


def _build_ssf_generic(
    backbone: str,
    num_classes: int,
    device: str | torch.device,
    pretrained: bool = True,
    target_keys: tuple[str, ...] = (),
    init_scale: float = 1.0,
    init_shift: float = 0.0,
) -> nn.Module:
    model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
    freeze_all(model)
    stage_map = _stage_map_for_backbone(model, backbone)
    for key in target_keys:
        module = stage_map.get(key)
        if module is not None:
            apply_ssf_recursively(module, max_modules=4, init_scale=init_scale, init_shift=init_shift)
    unfreeze_module(_classifier_module_for_backbone(model, backbone))
    return _finalize_model(model, device)


def build_resnet50_linear_probe(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_linear_probe_generic("resnet50", num_classes, device, pretrained=pretrained)


def build_resnet50_bn_tuning(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_bn_tuning_generic("resnet50", num_classes, device, pretrained=pretrained)


def build_resnet50_full_finetune(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_full_finetune_generic("resnet50", num_classes, device, pretrained=pretrained)


def build_resnet50_lora(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_lora_generic("resnet50", num_classes, device, pretrained=pretrained, target_keys=("layer4",))


def build_resnet50_adapters(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_tsa_generic("resnet50", num_classes, device, pretrained=pretrained, target_keys=("layer3", "layer4"))


def build_convnext_tiny_linear_probe(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_linear_probe_generic("convnext_tiny", num_classes, device, pretrained=pretrained)


def build_convnext_tiny_full_finetune(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_full_finetune_generic("convnext_tiny", num_classes, device, pretrained=pretrained)


def build_convnext_tiny_lora(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_lora_generic("convnext_tiny", num_classes, device, pretrained=pretrained, target_keys=("stage4",))


def build_convnext_tiny_adapters(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_tsa_generic("convnext_tiny", num_classes, device, pretrained=pretrained, target_keys=("stage3", "stage4"))


def build_mobilenet_v3_large_linear_probe(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_linear_probe_generic("mobilenet_v3_large", num_classes, device, pretrained=pretrained)


def build_mobilenet_v3_large_bn_tuning(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_bn_tuning_generic("mobilenet_v3_large", num_classes, device, pretrained=pretrained)


def build_mobilenet_v3_large_full_finetune(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_full_finetune_generic("mobilenet_v3_large", num_classes, device, pretrained=pretrained)


def build_mobilenet_v3_large_lora(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_lora_generic("mobilenet_v3_large", num_classes, device, pretrained=pretrained, target_keys=("stage4",))


def build_mobilenet_v3_large_adapters(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_tsa_generic("mobilenet_v3_large", num_classes, device, pretrained=pretrained, target_keys=("stage3", "stage4"))


def build_densenet121_linear_probe(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_linear_probe_generic("densenet121", num_classes, device, pretrained=pretrained)


def build_densenet121_bn_tuning(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_bn_tuning_generic("densenet121", num_classes, device, pretrained=pretrained)


def build_densenet121_full_finetune(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_full_finetune_generic("densenet121", num_classes, device, pretrained=pretrained)


def build_densenet121_lora(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_lora_generic("densenet121", num_classes, device, pretrained=pretrained, target_keys=("denseblock4",))


def build_densenet121_adapters(num_classes: int, device: str | torch.device, pretrained: bool = True) -> nn.Module:
    return _build_tsa_generic("densenet121", num_classes, device, pretrained=pretrained, target_keys=("denseblock3", "denseblock4"))


def _target_keys_from_spec(spec: dict[str, object], backbone: str) -> tuple[str, ...]:
    raw_targets = spec.get("peft_targets")
    targets = raw_targets if isinstance(raw_targets, dict) else {}
    feature_stages = targets.get("feature_stages", [])
    layer_keys = targets.get("layer_keys", [])
    include_classifier = bool(targets.get("classifier", False))

    keys: list[str] = []
    if backbone == "efficientnet_v2_s":
        for stage_idx in feature_stages if isinstance(feature_stages, list) else []:
            try:
                idx = int(stage_idx)
            except Exception:
                continue
            keys.append(f"features.{idx}")
    if isinstance(layer_keys, list):
        for key in layer_keys:
            text = str(key).strip()
            if text:
                keys.append(text)
    if include_classifier:
        keys.append("classifier")
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return tuple(deduped)


def _default_target_keys(backbone: str, method_type: str) -> tuple[str, ...]:
    defaults: dict[tuple[str, str], tuple[str, ...]] = {
        ("efficientnet_v2_s", "lora"): ("features.6", "features.7", "classifier"),
        ("efficientnet_v2_s", "dora"): ("features.6", "features.7", "classifier"),
        ("efficientnet_v2_s", "tsa"): ("features.5", "features.6", "features.7", "classifier"),
        ("efficientnet_v2_s", "adapter"): ("features.5", "features.6", "features.7", "classifier"),
        ("efficientnet_v2_s", "ssf"): ("features.5", "features.6", "features.7", "classifier"),
        ("resnet18", "lora"): ("layer4", "classifier"),
        ("resnet18", "tsa"): ("layer3", "layer4", "classifier"),
        ("resnet18", "adapter"): ("layer3", "layer4", "classifier"),
        ("resnet18", "ssf"): ("layer3", "layer4", "classifier"),
        ("resnet50", "lora"): ("layer4", "classifier"),
        ("resnet50", "tsa"): ("layer3", "layer4", "classifier"),
        ("resnet50", "adapter"): ("layer3", "layer4", "classifier"),
        ("resnet50", "ssf"): ("layer3", "layer4", "classifier"),
        ("convnext_tiny", "lora"): ("stage4", "classifier"),
        ("convnext_tiny", "tsa"): ("stage3", "stage4", "classifier"),
        ("convnext_tiny", "adapter"): ("stage3", "stage4", "classifier"),
        ("convnext_tiny", "ssf"): ("stage3", "stage4", "classifier"),
        ("mobilenet_v3_large", "lora"): ("stage4", "classifier"),
        ("mobilenet_v3_large", "tsa"): ("stage3", "stage4", "classifier"),
        ("mobilenet_v3_large", "adapter"): ("stage3", "stage4", "classifier"),
        ("mobilenet_v3_large", "ssf"): ("stage3", "stage4", "classifier"),
        ("densenet121", "lora"): ("denseblock4", "classifier"),
        ("densenet121", "tsa"): ("denseblock3", "denseblock4", "classifier"),
        ("densenet121", "adapter"): ("denseblock3", "denseblock4", "classifier"),
        ("densenet121", "ssf"): ("denseblock3", "denseblock4", "classifier"),
    }
    return defaults.get((backbone, method_type), ("classifier",))


def build_model_from_spec(
    spec: dict[str, object],
    *,
    num_classes: int,
    device: str | torch.device,
    pretrained: bool,
) -> nn.Module:
    base_model = str(spec.get("base_model", "efficientnet_v2_s")).strip().lower()
    backbone = "efficientnet_v2_s" if base_model == "efficientnet" else base_model
    method_type = str(spec.get("method_type", "baseline")).strip().lower()

    if method_type == "baseline":
        return strategy_builder(backbone, "linear_probe")(num_classes, device, pretrained)
    if method_type == "bn_tuning":
        return strategy_builder(backbone, "bn_tuning")(num_classes, device, pretrained)
    if method_type == "bn_last1":
        return strategy_builder(backbone, "bn_last1")(num_classes, device, pretrained)
    if method_type == "bn_last2":
        return strategy_builder(backbone, "bn_last2")(num_classes, device, pretrained)
    if method_type == "full_finetune":
        return strategy_builder(backbone, "full_finetune")(num_classes, device, pretrained)
    if method_type == "norm_tuning":
        return strategy_builder(backbone, "norm_tuning")(num_classes, device, pretrained)
    if method_type == "bitfit":
        model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
        scope = "all_bias"
        raw_params = spec.get("peft_params")
        if isinstance(raw_params, dict):
            scope = str(raw_params.get("scope", "all_bias")).strip().lower() or "all_bias"
        enable_bias_tuning(model, scope=scope, classifier_modules=[_classifier_module_for_backbone(model, backbone)])
        return _finalize_model(model, device)

    target_keys = _target_keys_from_spec(spec, backbone)
    if not target_keys:
        target_keys = _default_target_keys(backbone, method_type)
    params = spec.get("peft_params")
    peft_params = params if isinstance(params, dict) else {}

    if method_type == "lora":
        return _build_lora_generic(backbone, num_classes, device, pretrained=pretrained, target_keys=target_keys)
    if method_type == "dora":
        model = _load_backbone_classifier(backbone, num_classes, pretrained=pretrained)
        freeze_all(model)
        stage_map = _stage_map_for_backbone(model, backbone)
        for key in target_keys:
            module = stage_map.get(key)
            if module is not None:
                apply_dora_recursively(module)
        unfreeze_module(_classifier_module_for_backbone(model, backbone))
        return _finalize_model(model, device)
    if method_type == "tsa":
        return _build_tsa_generic(backbone, num_classes, device, pretrained=pretrained, target_keys=target_keys)
    if method_type == "adapter":
        bottleneck_dim = int(peft_params.get("bottleneck_dim", 32))
        return _build_adapter_generic(
            backbone,
            num_classes,
            device,
            pretrained=pretrained,
            target_keys=target_keys,
            bottleneck_dim=bottleneck_dim,
        )
    if method_type == "ssf":
        init_scale = float(peft_params.get("init_scale", 1.0))
        init_shift = float(peft_params.get("init_shift", 0.0))
        return _build_ssf_generic(
            backbone,
            num_classes,
            device,
            pretrained=pretrained,
            target_keys=target_keys,
            init_scale=init_scale,
            init_shift=init_shift,
        )
    raise ValueError(f"Unsupported method_type for generated spec model: {method_type}")


def strategy_builder(backbone: str, strategy: str) -> Callable[[int, str | torch.device, bool], nn.Module]:
    registry: dict[tuple[str, str], Callable[[int, str | torch.device, bool], nn.Module]] = {
        ("resnet18", "linear_probe"): build_resnet18_linear_probe,
        ("resnet18", "lora"): build_resnet18_lora,
        ("resnet18", "tsa"): build_resnet18_adapters,
        ("resnet18", "bn_tuning"): build_resnet18_bn_tuning,
        ("resnet18", "norm_tuning"): lambda n, d, p: _build_norm_tuning_generic("resnet18", n, d, p),
        ("resnet18", "full_finetune"): build_resnet18_full_finetune,
        ("resnet18", "adapter"): lambda n, d, p: _build_adapter_generic("resnet18", n, d, p, target_keys=("layer3", "layer4", "classifier"), bottleneck_dim=32),
        ("resnet18", "bitfit"): lambda n, d, p: _build_bitfit_generic("resnet18", n, d, p),
        ("resnet18", "ssf"): lambda n, d, p: _build_ssf_generic("resnet18", n, d, p, target_keys=("layer3", "layer4", "classifier")),
        ("efficientnet", "linear_probe"): build_efficientnet_linear_probe,
        ("efficientnet", "lora"): build_efficientnet_lora,
        ("efficientnet", "dora"): build_efficientnet_dora,
        ("efficientnet", "tsa"): build_efficientnet_adapters,
        ("efficientnet", "bn_tuning"): build_efficientnet_bn_tuning,
        ("efficientnet", "norm_tuning"): lambda n, d, p: _build_norm_tuning_generic("efficientnet_v2_s", n, d, p),
        ("efficientnet", "bn_last1"): build_efficientnet_bn_last1,
        ("efficientnet", "bn_last2"): build_efficientnet_bn_last2,
        ("efficientnet", "full_finetune"): build_efficientnet_full_finetune,
        ("efficientnet", "adapter"): lambda n, d, p: _build_adapter_generic("efficientnet_v2_s", n, d, p, target_keys=("features.5", "features.6", "features.7", "classifier"), bottleneck_dim=32),
        ("efficientnet", "bitfit"): lambda n, d, p: _build_bitfit_generic("efficientnet_v2_s", n, d, p),
        ("efficientnet", "ssf"): lambda n, d, p: _build_ssf_generic("efficientnet_v2_s", n, d, p, target_keys=("features.5", "features.6", "features.7", "classifier")),
        ("efficientnet_v2_s", "linear_probe"): build_efficientnet_linear_probe,
        ("efficientnet_v2_s", "lora"): build_efficientnet_lora,
        ("efficientnet_v2_s", "dora"): build_efficientnet_dora,
        ("efficientnet_v2_s", "tsa"): build_efficientnet_adapters,
        ("efficientnet_v2_s", "bn_tuning"): build_efficientnet_bn_tuning,
        ("efficientnet_v2_s", "norm_tuning"): lambda n, d, p: _build_norm_tuning_generic("efficientnet_v2_s", n, d, p),
        ("efficientnet_v2_s", "bn_last1"): build_efficientnet_bn_last1,
        ("efficientnet_v2_s", "bn_last2"): build_efficientnet_bn_last2,
        ("efficientnet_v2_s", "full_finetune"): build_efficientnet_full_finetune,
        ("efficientnet_v2_s", "adapter"): lambda n, d, p: _build_adapter_generic("efficientnet_v2_s", n, d, p, target_keys=("features.5", "features.6", "features.7", "classifier"), bottleneck_dim=32),
        ("efficientnet_v2_s", "bitfit"): lambda n, d, p: _build_bitfit_generic("efficientnet_v2_s", n, d, p),
        ("efficientnet_v2_s", "ssf"): lambda n, d, p: _build_ssf_generic("efficientnet_v2_s", n, d, p, target_keys=("features.5", "features.6", "features.7", "classifier")),
        ("resnet50", "linear_probe"): build_resnet50_linear_probe,
        ("resnet50", "lora"): build_resnet50_lora,
        ("resnet50", "tsa"): build_resnet50_adapters,
        ("resnet50", "bn_tuning"): build_resnet50_bn_tuning,
        ("resnet50", "norm_tuning"): lambda n, d, p: _build_norm_tuning_generic("resnet50", n, d, p),
        ("resnet50", "full_finetune"): build_resnet50_full_finetune,
        ("resnet50", "adapter"): lambda n, d, p: _build_adapter_generic("resnet50", n, d, p, target_keys=("layer3", "layer4", "classifier"), bottleneck_dim=32),
        ("resnet50", "bitfit"): lambda n, d, p: _build_bitfit_generic("resnet50", n, d, p),
        ("resnet50", "ssf"): lambda n, d, p: _build_ssf_generic("resnet50", n, d, p, target_keys=("layer3", "layer4", "classifier")),
        ("convnext_tiny", "linear_probe"): build_convnext_tiny_linear_probe,
        ("convnext_tiny", "lora"): build_convnext_tiny_lora,
        ("convnext_tiny", "tsa"): build_convnext_tiny_adapters,
        ("convnext_tiny", "norm_tuning"): lambda n, d, p: _build_norm_tuning_generic("convnext_tiny", n, d, p),
        ("convnext_tiny", "full_finetune"): build_convnext_tiny_full_finetune,
        ("convnext_tiny", "adapter"): lambda n, d, p: _build_adapter_generic("convnext_tiny", n, d, p, target_keys=("stage3", "stage4", "classifier"), bottleneck_dim=32),
        ("convnext_tiny", "bitfit"): lambda n, d, p: _build_bitfit_generic("convnext_tiny", n, d, p),
        ("convnext_tiny", "ssf"): lambda n, d, p: _build_ssf_generic("convnext_tiny", n, d, p, target_keys=("stage3", "stage4", "classifier")),
        ("mobilenet_v3_large", "linear_probe"): build_mobilenet_v3_large_linear_probe,
        ("mobilenet_v3_large", "lora"): build_mobilenet_v3_large_lora,
        ("mobilenet_v3_large", "tsa"): build_mobilenet_v3_large_adapters,
        ("mobilenet_v3_large", "bn_tuning"): build_mobilenet_v3_large_bn_tuning,
        ("mobilenet_v3_large", "norm_tuning"): lambda n, d, p: _build_norm_tuning_generic("mobilenet_v3_large", n, d, p),
        ("mobilenet_v3_large", "full_finetune"): build_mobilenet_v3_large_full_finetune,
        ("mobilenet_v3_large", "adapter"): lambda n, d, p: _build_adapter_generic("mobilenet_v3_large", n, d, p, target_keys=("stage3", "stage4", "classifier"), bottleneck_dim=32),
        ("mobilenet_v3_large", "bitfit"): lambda n, d, p: _build_bitfit_generic("mobilenet_v3_large", n, d, p),
        ("mobilenet_v3_large", "ssf"): lambda n, d, p: _build_ssf_generic("mobilenet_v3_large", n, d, p, target_keys=("stage3", "stage4", "classifier")),
        ("densenet121", "linear_probe"): build_densenet121_linear_probe,
        ("densenet121", "lora"): build_densenet121_lora,
        ("densenet121", "tsa"): build_densenet121_adapters,
        ("densenet121", "bn_tuning"): build_densenet121_bn_tuning,
        ("densenet121", "norm_tuning"): lambda n, d, p: _build_norm_tuning_generic("densenet121", n, d, p),
        ("densenet121", "full_finetune"): build_densenet121_full_finetune,
        ("densenet121", "adapter"): lambda n, d, p: _build_adapter_generic("densenet121", n, d, p, target_keys=("denseblock3", "denseblock4", "classifier"), bottleneck_dim=32),
        ("densenet121", "bitfit"): lambda n, d, p: _build_bitfit_generic("densenet121", n, d, p),
        ("densenet121", "ssf"): lambda n, d, p: _build_ssf_generic("densenet121", n, d, p, target_keys=("denseblock3", "denseblock4", "classifier")),
    }
    return registry[(backbone, strategy)]
