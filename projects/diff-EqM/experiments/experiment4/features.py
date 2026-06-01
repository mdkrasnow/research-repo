"""Feature extractors for Experiment 4.

Every extractor takes a uint8 NHWC array in [0,255] (the format produced by
eqm-upstream/sample_gd.py's .npz, key 'arr_0') and returns float32 features
(N, D). Each extractor exposes a `preprocessing_hash()` so the audit can assert
that generated samples and reference banks were embedded identically (failure
mode: feature-extractor preprocessing mismatch).

Backbones:
  inception_pool3 : reuses experiments/evaluate_fid.py InceptionV3Features. FID/KID.
  dinov2_vitl14   : torch.hub DINOv2 (primary NN feature, cosine).
  clip_vitl14     : open_clip (semantic NN cross-check, cosine).
  lpips_alex      : per-pair LPIPS (visual-copy check on suspicious pairs only).
  stub            : deterministic seeded linear projection, CPU, ZERO external
                    deps / no network. TEST-ONLY (local smoke). Never use stub
                    features for any reported metric.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# evaluate_fid.py lives one dir up; reuse its Inception extractor verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _to_nchw_float(images: np.ndarray) -> torch.Tensor:
    """uint8 NHWC [0,255] -> float NCHW [0,1]."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected uint8 NHWC RGB, got shape {images.shape}")
    x = torch.from_numpy(np.ascontiguousarray(images)).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


class _BaseExtractor:
    name = "base"
    dim = 0
    is_distance = False  # True for LPIPS-style pairwise-only backbones

    def preprocessing_hash(self) -> str:
        raise NotImplementedError

    @torch.no_grad()
    def extract(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        raise NotImplementedError


class InceptionExtractor(_BaseExtractor):
    name = "inception_pool3"
    dim = 2048

    def __init__(self, device="cuda"):
        from evaluate_fid import InceptionV3Features  # reuse, do not duplicate

        self.device = device
        self.model = InceptionV3Features(device=device)

    def preprocessing_hash(self) -> str:
        return "inception_pool3|in[-1,1]|bilinear299|torchvision_default_weights"

    @torch.no_grad()
    def extract(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        x = _to_nchw_float(images).mul_(2.0).sub_(1.0)  # [0,1] -> [-1,1]
        out = []
        for i in range(0, x.shape[0], batch_size):
            b = x[i : i + batch_size].to(self.device)
            out.append(self.model(b).cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


class DinoV2Extractor(_BaseExtractor):
    name = "dinov2_vitl14"
    dim = 1024

    def __init__(self, device="cuda", size=224):
        self.device = device
        self.size = size
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        self.model.eval().to(device)

    def preprocessing_hash(self) -> str:
        return f"dinov2_vitl14|imagenet_norm|bicubic{self.size}|cls_token|l2norm"

    @torch.no_grad()
    def extract(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        mean = _IMAGENET_MEAN.to(self.device)
        std = _IMAGENET_STD.to(self.device)
        out = []
        x = _to_nchw_float(images)
        for i in range(0, x.shape[0], batch_size):
            b = x[i : i + batch_size].to(self.device)
            b = F.interpolate(b, size=(self.size, self.size), mode="bicubic", align_corners=False)
            b = (b - mean) / std
            f = self.model(b)  # (B, 1024) cls token
            f = F.normalize(f, dim=1)
            out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


class ClipExtractor(_BaseExtractor):
    name = "clip_vitl14"
    dim = 768

    def __init__(self, device="cuda", size=224):
        import open_clip

        self.device = device
        self.size = size
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.model.eval().to(device)

    def preprocessing_hash(self) -> str:
        return f"clip_vitl14_openai|clip_norm|bicubic{self.size}|image_features|l2norm"

    @torch.no_grad()
    def extract(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        mean = _CLIP_MEAN.to(self.device)
        std = _CLIP_STD.to(self.device)
        out = []
        x = _to_nchw_float(images)
        for i in range(0, x.shape[0], batch_size):
            b = x[i : i + batch_size].to(self.device)
            b = F.interpolate(b, size=(self.size, self.size), mode="bicubic", align_corners=False)
            b = (b - mean) / std
            f = self.model.encode_image(b)
            f = F.normalize(f, dim=1)
            out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


class StubExtractor(_BaseExtractor):
    """Deterministic CPU projection. TEST-ONLY. No network, no model download.

    Downsamples to 16x16 grayscale, flattens, applies a fixed seeded linear map.
    Lets the full pipeline (config -> features -> FID/KID -> NN -> panels ->
    CSV) run on a laptop with random fake images. Results are meaningless.
    """

    name = "stub"

    def __init__(self, device="cpu", dim=64, seed=0):
        self.device = "cpu"
        self.dim = dim
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((16 * 16, dim)).astype(np.float32) / np.sqrt(16 * 16)

    def preprocessing_hash(self) -> str:
        return f"stub|gray16|linear{self.dim}|TEST_ONLY"

    @torch.no_grad()
    def extract(self, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
        x = _to_nchw_float(images)
        x = x.mean(dim=1, keepdim=True)  # grayscale
        x = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
        flat = x.view(x.shape[0], -1).cpu().numpy().astype(np.float32)
        return flat @ self._proj


class LpipsScorer:
    """Pairwise LPIPS for suspicious gen/train pairs only (not a bank feature)."""

    name = "lpips_alex"
    is_distance = True

    def __init__(self, device="cuda"):
        import lpips

        self.device = device
        self.net = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def distance(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        """img_a, img_b: single uint8 HWC [0,255]."""
        a = _to_nchw_float(img_a[None]).mul_(2.0).sub_(1.0).to(self.device)
        b = _to_nchw_float(img_b[None]).mul_(2.0).sub_(1.0).to(self.device)
        return float(self.net(a, b).item())


_FID_BACKBONES = {"inception_pool3": InceptionExtractor, "stub": StubExtractor}
_NN_BACKBONES = {
    "dinov2_vitl14": DinoV2Extractor,
    "clip_vitl14": ClipExtractor,
    "stub": StubExtractor,
}


def build_extractor(name: str, device: str = "cuda", **kw) -> _BaseExtractor:
    table = {**_FID_BACKBONES, **_NN_BACKBONES}
    if name not in table:
        raise ValueError(f"unknown backbone '{name}'. known: {sorted(table)}")
    return table[name](device=device, **kw)


def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
