from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import scan2d


MODALITIES: Tuple[str, ...] = ("cqt", "logmel", "stft")

class ChannelAttention(nn.Module):
    """CBAM channel attention using shared average/max-pool projections."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_score = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_score = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg_score + max_score)


class SpatialAttention(nn.Module):
    """CBAM spatial attention from channel-wise average and maximum maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7.")
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)],
            dim=1,
        )
        return torch.sigmoid(self.conv(descriptors))


class ResidualCBAM(nn.Module):
    """CBAM refinement on a residual branch to preserve the baseline features."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        # A small residual scale preserves the initial encoder features in UATR-Triba.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = x * self.channel_attention(x)
        refined = refined * self.spatial_attention(refined)
        return x + self.residual_scale * refined


class UATRTriba(nn.Module):
    """
    Three VMamba encoders -> independent residual CBAM -> concatenation -> VSSM.

    Modality dropout and internal VSSM dropout regularise the high-capacity
    fusion path of UATR-Triba while retaining spatial modelling.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        encoder_dim: int = 64,
        encoder_depth: int = 2,
        encoder_patch_size: int = 4,
        feature_size: Optional[Sequence[int]] = (32, 128),
        attention_reduction: int = 8,
        modality_dropout: float = 0.1,
        fusion_dim: int = 64,
        fusion_depths: Sequence[int] = (2, 2, 5, 2),
        fusion_patch_size: int = 4,
        fusion_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if feature_size is not None:
            if len(feature_size) != 2 or min(feature_size) <= 0:
                raise ValueError("feature_size must contain two positive integers.")
            self.feature_size = (int(feature_size[0]), int(feature_size[1]))
        else:
            self.feature_size = None
        for name, probability in (
            ("modality_dropout", modality_dropout),
            ("fusion_dropout", fusion_dropout),
            ("classifier_dropout", classifier_dropout),
        ):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        self.modality_dropout = modality_dropout

        self.encoders = nn.ModuleDict(
            {
                modality: scan2d.MambaModalityEncoder(
                    in_channels=in_channels,
                    feature_dim=encoder_dim,
                    depth=encoder_depth,
                    patch_size=encoder_patch_size,
                    drop_path_rate=drop_path_rate,
                )
                for modality in MODALITIES
            }
        )
        self.attention = nn.ModuleDict(
            {
                modality: ResidualCBAM(encoder_dim, attention_reduction)
                for modality in MODALITIES
            }
        )

        vmamba_kwargs = scan2d._common_vmamba_kwargs()
        vmamba_kwargs.update(
            ssm_drop_rate=fusion_dropout,
            mlp_drop_rate=fusion_dropout,
        )
        self.fusion_vmamba = scan2d.VSSM(
            in_chans=len(MODALITIES) * encoder_dim,
            num_classes=num_classes,
            depths=list(fusion_depths),
            dims=fusion_dim,
            patch_size=fusion_patch_size,
            drop_path_rate=drop_path_rate,
            **vmamba_kwargs,
        )
        old_head = self.fusion_vmamba.classifier.head
        self.fusion_vmamba.classifier.head = nn.Sequential(
            nn.Dropout(classifier_dropout),
            old_head,
        )

    @staticmethod
    def _normalize_inputs(
        inputs: Union[Mapping[str, torch.Tensor], torch.Tensor],
        logmel: Optional[torch.Tensor],
        stft: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if isinstance(inputs, Mapping):
            missing = set(MODALITIES) - set(inputs)
            if missing:
                raise KeyError(f"Missing modalities: {sorted(missing)}")
            modality_inputs = {name: inputs[name] for name in MODALITIES}
        else:
            if logmel is None or stft is None:
                raise ValueError(
                    "Pass a modality mapping or all three tensors: cqt, logmel, stft."
                )
            modality_inputs = {"cqt": inputs, "logmel": logmel, "stft": stft}

        batch_size = next(iter(modality_inputs.values())).size(0)
        for modality, tensor in modality_inputs.items():
            if tensor.ndim != 4:
                raise ValueError(
                    f"{modality} must be [B, C, H, W], got {tuple(tensor.shape)}."
                )
            if tensor.size(0) != batch_size:
                raise ValueError("All modalities must have the same batch size.")
        return modality_inputs

    def _apply_modality_dropout(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not self.training or self.modality_dropout == 0.0:
            return features

        reference = features[MODALITIES[0]]
        batch_size = reference.size(0)
        keep = torch.rand(
            batch_size,
            len(MODALITIES),
            device=reference.device,
        ) >= self.modality_dropout
        # Never remove every modality from a sample.
        empty_rows = ~keep.any(dim=1)
        if empty_rows.any():
            fallback = torch.randint(
                len(MODALITIES),
                (int(empty_rows.sum().item()),),
                device=reference.device,
            )
            keep[empty_rows] = False
            keep[empty_rows, fallback] = True

        scale = 1.0 / (1.0 - self.modality_dropout)
        return {
            modality: features[modality]
            * keep[:, index, None, None, None].to(reference.dtype)
            * scale
            for index, modality in enumerate(MODALITIES)
        }

    def forward_features(
        self,
        inputs: Union[Mapping[str, torch.Tensor], torch.Tensor],
        logmel: Optional[torch.Tensor] = None,
        stft: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        modality_inputs = self._normalize_inputs(inputs, logmel, stft)
        features = {
            modality: self.attention[modality](
                self.encoders[modality](modality_inputs[modality])
            )
            for modality in MODALITIES
        }
        features = self._apply_modality_dropout(features)

        target_size = self.feature_size
        if target_size is None:
            target_size = (
                min(feature.shape[-2] for feature in features.values()),
                min(feature.shape[-1] for feature in features.values()),
            )
        aligned_features = [
            F.adaptive_avg_pool2d(features[modality], target_size)
            for modality in MODALITIES
        ]
        return torch.cat(aligned_features, dim=1)

    def forward(
        self,
        inputs: Union[Mapping[str, torch.Tensor], torch.Tensor],
        logmel: Optional[torch.Tensor] = None,
        stft: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.fusion_vmamba(self.forward_features(inputs, logmel, stft))

### create model ###
UATR_Triba = UATRTriba

def run_tests(device="cpu"):
    import io
    import unittest

    device = torch.device(device)
    torch.manual_seed(42)
    torch.set_num_threads(min(torch.get_num_threads(), 4))

    class ModelTests(unittest.TestCase):
        def setUp(self):
            self.kwargs = dict(
                num_classes=4, encoder_dim=8, encoder_depth=1,
                feature_size=(8, 12), fusion_dim=8, fusion_depths=(1, 1),
                modality_dropout=0.0, drop_path_rate=0.0,
            )
            self.model = UATRTriba(**self.kwargs).to(device)
            self.inputs = {
                "cqt": torch.randn(2, 1, 16, 24, device=device),
                "logmel": torch.randn(2, 1, 20, 28, device=device),
                "stft": torch.randn(2, 1, 24, 32, device=device),
            }

        def test_full_default_model(self):
            model = UATRTriba(num_classes=4).to(device).eval()
            with torch.no_grad():
                features = model.forward_features(self.inputs)
                self.assertEqual(tuple(features.shape), (2, 192, 32, 128))
                logits = model.fusion_vmamba(features)
            self.assertEqual(tuple(logits.shape), (2, 4))
            self.assertTrue(torch.isfinite(logits).all().item())
            print("model parameters：", sum(p.numel() for p in model.parameters()))

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ModelTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="测试设备，例如 cpu、cuda、cuda:0")
    args = parser.parse_args()
    run_tests(device=args.device)
