# submission/model.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


STATS = {
    "lat_mean": 39.95127270421107,
    "lat_std":  0.00045387404598668803,
    "lon_mean": -75.19183937087774,
    "lon_std":  0.0004151705814206056,
}


def build_eval_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CustomViTRegressor(nn.Module):
    def __init__(self, num_outputs: int = 2, dropout: float = 0.1):
        super().__init__()
        # self.backbone = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        self.backbone = models.vit_l_16(weights=None)
        hidden = self.backbone.hidden_dim
        self.backbone.heads = nn.Identity()

        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.regressor(feats)


def _normalize_state_dict_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        if kk.startswith("model."):
            kk = kk[len("model.") :]
        out[kk] = v
    return out


class Model(nn.Module):
    

    def __init__(self, weights_path: str = "__no_weights__.pth", img_size: int = 224):
        super().__init__()
        self.model = CustomViTRegressor(num_outputs=2, dropout=0.1)
        self.transform = build_eval_transform(img_size)
        self.stats = STATS

        self._device: Optional[torch.device] = None

        wp = Path(weights_path)
        if weights_path and weights_path != "__no_weights__.pth" and wp.exists():
            ckpt = torch.load(str(wp), map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                sd = _normalize_state_dict_keys(ckpt["state_dict"])
            elif isinstance(ckpt, dict):
                sd = _normalize_state_dict_keys(ckpt)
            else:
                raise RuntimeError("Checkpoint must be a state_dict or {'state_dict': ...}.")
            self.model.load_state_dict(sd, strict=False)

        self.eval()

    def eval(self) -> "Model":
        super().eval()
        self.model.eval()
        return self

    def _ensure_device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self._device)
        return self._device

    def _to_tensor(self, item: Any) -> torch.Tensor:
        if isinstance(item, torch.Tensor):
            if item.ndim == 3:
                return item.float()
            raise ValueError(f"Unexpected tensor shape: {tuple(item.shape)}")

        if isinstance(item, (str, Path)):
            p = Path(item)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            img = Image.open(str(p))
            return self.transform(img)

        if isinstance(item, Image.Image):
            return self.transform(item)

        raise TypeError(f"Unsupported input type in predict(): {type(item)}")

    def _denormalize(self, pred_norm: np.ndarray) -> np.ndarray:
        lat = pred_norm[:, 0] * self.stats["lat_std"] + self.stats["lat_mean"]
        lon = pred_norm[:, 1] * self.stats["lon_std"] + self.stats["lon_mean"]
        lat = np.clip(lat, -90.0, 90.0)
        lon = np.clip(lon, -180.0, 180.0)
        return np.stack([lat, lon], axis=1)

    @torch.inference_mode()
    def predict(self, batch: Iterable[Any]) -> List[Any]:
        
        device = self._ensure_device()
        xs = [self._to_tensor(x) for x in batch]
        x = torch.stack(xs, dim=0).to(device, non_blocking=True)

        out = self.model(x)                 # [B,2] normalized
        out_np = out.detach().float().cpu().numpy()
        pred = self._denormalize(out_np)    # [B,2] lat/lon

        return pred.tolist()


def get_model() -> Model:
    return Model()
