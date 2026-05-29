from __future__ import annotations

from pathlib import Path
import logging
from typing import Any

from .util.keypoints import _build_features

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from ultralytics import YOLO  # type: ignore
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional deps
    cv2 = None  # type: ignore
    np = None  # type: ignore
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    YOLO = None  # type: ignore
    tqdm = None  # type: ignore

logger = logging.getLogger(__name__)


def _require_viewpoint_deps() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if YOLO is None:
        missing.append("ultralytics")
    if tqdm is None:
        missing.append("tqdm")
    if missing:
        raise ImportError(
            "Viewpoint estimation dependencies are missing. "
            "Install the optional extras to enable this feature: "
            f"{', '.join(missing)}."
        )

if torch is None or nn is None or F is None:
    class ViewpointPredictor:
        def __init__(self, input_dim: int, hidden_dims: list[int]):
            _require_viewpoint_deps()
else:
    class ViewpointPredictor(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: list[int]):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15)]
                prev = h
            layers.append(nn.Linear(prev, 2))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return F.normalize(self.net(x), p=2, dim=1, eps=1e-8)




def _load_state_dict(path: str, device: Any):
    
    ckpt = torch.load(path, map_location=device, weights_only=True)

    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def _infer_hidden_dims_from_state_dict(state_dict: dict) -> list[int]:
    hidden_dims: list[int] = []
    idx = 0
    while True:
        weight_key = f"net.{idx}.weight"
        if weight_key not in state_dict:
            break
        out_dim = int(state_dict[weight_key].shape[0])
        if out_dim == 2:
            break
        hidden_dims.append(out_dim)
        idx += 4
    if not hidden_dims:
        raise ValueError("Could not infer hidden dimensions from viewpoint checkpoint")
    return hidden_dims


class ViewpointInference:


    def __init__(self, cfg: dict):
        _require_viewpoint_deps()
        self.cfg = cfg
        requested_device = str(cfg["device"]).strip()
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested (%s) but not available. Falling back to CPU.", requested_device)
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.yolo = YOLO(cfg["keypoint_weights"])
        state = _load_state_dict(cfg["viewpoint_weights"], self.device)
        input_dim = int(state["net.0.weight"].shape[1])
        hidden_dims = list(cfg.get("hidden_dims") or _infer_hidden_dims_from_state_dict(state))
        inferred_hidden_dims = _infer_hidden_dims_from_state_dict(state)
        if hidden_dims != inferred_hidden_dims:
            hidden_dims = inferred_hidden_dims
        self.cfg["hidden_dims"] = hidden_dims
        self.model = ViewpointPredictor(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()


    def _predict_angle_from_image(self, image: Any) -> tuple[float | None, Any]:
        result = self.yolo.predict(source=image, conf=self.cfg["yolo_conf"], device=str(self.device), verbose=False)[0]
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            logger.warning("No keypoints detected by YOLO.")
            return None, None
        idx = int(np.argmax(result.boxes.conf.cpu().numpy())) if result.boxes is not None and result.boxes.conf is not None else 0
        xy = result.keypoints.xy[idx].cpu().numpy()
        conf = result.keypoints.conf[idx].cpu().numpy()
        if int((conf >= self.cfg["min_kpt_conf"]).sum()) < self.cfg["min_keypoints"]:
            logger.warning("Insufficient keypoints above confidence threshold.")
            return None, None
        h, w = image.shape[:2]
        kpts_xyc = np.column_stack((xy[:, 0] / w, xy[:, 1] / h, conf))
        feats = _build_features(kpts_xyc, self.cfg["feature_type"])
        out = self.model(torch.from_numpy(feats).unsqueeze(0).to(self.device))[0]
        angle = float(np.degrees(np.arctan2(out[0].item(), out[1].item())) % 360.0)
        return angle, xy
    

    @((torch.no_grad() if torch is not None else (lambda f: f)))
    def predict_angle(self, image_path: str | Path) -> float | None:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Failed to read image at %s", image_path)
            return None
        angle, _ = self._predict_angle_from_image(image)
        return angle

    @((torch.no_grad() if torch is not None else (lambda f: f)))
    def predict_angle_from_image(self, image: Any) -> float | None:
        angle, _ = self._predict_angle_from_image(image)
        return angle

    @((torch.no_grad() if torch is not None else (lambda f: f)))
    def predict_with_keypoints(self, image_path: str | Path) -> tuple[float | None, Any]:
        """Predict angle and return keypoints."""
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Failed to read image at %s", image_path)
            return None, None
        return self._predict_angle_from_image(image)
    
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_viewpoint_cache(img_dir: Path, predictor: ViewpointInference) -> dict:
    image_paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    cache = {}
    for p in tqdm(image_paths, desc=f"Estimating viewpoints in {img_dir.name}: "):
        cache[p.name] = predictor.predict_angle(p)
    return cache

def _circular_diff_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)
