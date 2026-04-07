from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn
from torchvision import models


def build_optimizer(model: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.Adam(trainable_params, lr=lr)


def _resolved_device(device: str | torch.device) -> str | torch.device:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_resnet18_classifier(num_classes: int) -> nn.Module:
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except TypeError:
        model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_efficientnet_v2_s_classifier(num_classes: int) -> nn.Module:
    try:
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    except TypeError:
        model = models.efficientnet_v2_s(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
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


class ConvAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck_ratio: float = 0.25) -> None:
        super().__init__()
        bottleneck_channels = max(8, int(channels * bottleneck_ratio))
        self.down = nn.Conv2d(channels, bottleneck_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck_channels, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(inputs)))


class AdapterWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, channels: int) -> None:
        super().__init__()
        self.base_module = base_module
        self.adapter = ConvAdapter(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.base_module(inputs)
        return outputs + self.adapter(outputs)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = copy.deepcopy(base)
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = rank
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
        self.lora_down = nn.Conv2d(
            in_channels=base.in_channels,
            out_channels=rank,
            kernel_size=1,
            stride=base.stride,
            padding=0,
            bias=False,
        )
        self.lora_up = nn.Conv2d(
            in_channels=rank,
            out_channels=base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_up(self.lora_down(inputs)) * self.scaling


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


def wrap_module_with_adapter(parent: nn.Module, child_name: str, channels: int) -> None:
    child = getattr(parent, child_name) if not isinstance(parent, (nn.Sequential, nn.ModuleList)) else parent[int(child_name)]
    _replace_child(parent, child_name, AdapterWrapper(child, channels))


def _finalize_model(model: nn.Module, device: str | torch.device) -> nn.Module:
    model.to(_resolved_device(device))
    return model


def build_resnet18_linear_probe(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_resnet18_classifier(num_classes)
    freeze_all(model)
    unfreeze_module(model.fc)
    return _finalize_model(model, device)


def build_resnet18_lora(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_resnet18_classifier(num_classes)
    freeze_all(model)
    apply_lora_recursively(model.layer4)
    model.fc = LoRALinear(model.fc, rank=8, alpha=16.0)
    return _finalize_model(model, device)


def build_resnet18_adapters(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_resnet18_classifier(num_classes)
    freeze_all(model)
    wrap_module_with_adapter(model.layer3, "0", channels=256)
    wrap_module_with_adapter(model.layer3, "1", channels=256)
    wrap_module_with_adapter(model.layer4, "0", channels=512)
    wrap_module_with_adapter(model.layer4, "1", channels=512)
    unfreeze_module(model.fc)
    return _finalize_model(model, device)


def build_resnet18_bn_tuning(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_resnet18_classifier(num_classes)
    enable_bn_tuning(model, [model.fc])
    return _finalize_model(model, device)


def build_resnet18_full_finetune(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_resnet18_classifier(num_classes)
    unfreeze_all(model)
    return _finalize_model(model, device)


def build_efficientnet_linear_probe(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes)
    freeze_all(model)
    unfreeze_module(model.classifier[1])
    return _finalize_model(model, device)


def _apply_lora_to_feature_slice(feature_slice: nn.Module) -> None:
    apply_lora_recursively(feature_slice)


def build_efficientnet_lora(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes)
    freeze_all(model)
    _apply_lora_to_feature_slice(model.features[6])
    _apply_lora_to_feature_slice(model.features[7])
    model.classifier[1] = LoRALinear(model.classifier[1], rank=8, alpha=16.0)
    return _finalize_model(model, device)


def build_efficientnet_adapters(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes)
    freeze_all(model)
    wrap_module_with_adapter(model.features, "5", channels=160)
    wrap_module_with_adapter(model.features, "6", channels=256)
    wrap_module_with_adapter(model.features, "7", channels=256)
    unfreeze_module(model.classifier[1])
    return _finalize_model(model, device)


def build_efficientnet_bn_tuning(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes)
    enable_bn_tuning(model, [model.classifier[1]])
    return _finalize_model(model, device)


def build_efficientnet_full_finetune(num_classes: int, device: str | torch.device) -> nn.Module:
    model = load_efficientnet_v2_s_classifier(num_classes)
    unfreeze_all(model)
    return _finalize_model(model, device)


def strategy_builder(backbone: str, strategy: str) -> Callable[[int, str | torch.device], nn.Module]:
    registry: dict[tuple[str, str], Callable[[int, str | torch.device], nn.Module]] = {
        ("resnet18", "linear_probe"): build_resnet18_linear_probe,
        ("resnet18", "lora"): build_resnet18_lora,
        ("resnet18", "tsa"): build_resnet18_adapters,
        ("resnet18", "bn_tuning"): build_resnet18_bn_tuning,
        ("resnet18", "full_finetune"): build_resnet18_full_finetune,
        ("efficientnet", "linear_probe"): build_efficientnet_linear_probe,
        ("efficientnet", "lora"): build_efficientnet_lora,
        ("efficientnet", "tsa"): build_efficientnet_adapters,
        ("efficientnet", "bn_tuning"): build_efficientnet_bn_tuning,
        ("efficientnet", "full_finetune"): build_efficientnet_full_finetune,
    }
    return registry[(backbone, strategy)]
