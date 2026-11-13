import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

repo = Path.home() / "Rekep"
out = repo / "project2_evidence/02_mobile/rgbd_run01"
sys.path.insert(0, str(repo))

from utils import get_config
from keypoint_proposal import KeypointProposer

def rgb8(x):
    x = np.asarray(x)[..., :3]
    if np.issubdtype(x.dtype, np.floating):
        if np.nanmax(x) <= 1.0:
            x = x * 255.0
    return np.nan_to_num(x).clip(0, 255).astype(np.uint8)

data = np.load(out / "forward_camera.npz", allow_pickle=False)

rgb = torch.as_tensor(rgb8(data["rgb"]), dtype=torch.uint8)
points = torch.as_tensor(np.asarray(data["points"], dtype=np.float32), dtype=torch.float32)

# 关键修复：将 NumPy 分割标签转换为 PyTorch Tensor
masks = torch.as_tensor(
    np.asarray(data["seg"], dtype=np.int64)
)

print("RGB:", rgb.shape, rgb.dtype, flush=True)
print("POINTS:", points.shape, points.dtype, flush=True)
print("MASKS:", masks.shape, masks.dtype, flush=True)
print("MASK_IS_CUDA:", masks.is_cuda, flush=True)
print("SEGMENT_COUNT:", len(torch.unique(masks)), flush=True)

config = get_config(config_path=str(repo / "configs/config.yaml"))
proposer = KeypointProposer(config["keypoint_proposer"])

summary_path = out / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))

try:
    keypoints, overlay = proposer.get_keypoints(
        rgb, points, masks
    )

    keypoints = np.asarray(keypoints)
    np.save(out / "forward_keypoints.npy", keypoints)
    Image.fromarray(rgb8(overlay)).save(
        out / "forward_keypoints.png"
    )

    summary["keypoint_retry"] = {
        "status": "success",
        "count": int(len(keypoints)),
        "source": "forward_camera.npz",
        "mask_adapter": "numpy_to_torch"
    }

    print("KEYPOINT_SUCCESS", flush=True)
    print("KEYPOINT_COUNT =", len(keypoints), flush=True)
    print("KEYPOINTS =", keypoints.tolist(), flush=True)

except Exception:
    summary["keypoint_retry"] = {
        "status": "failed",
        "error": traceback.format_exc()
    }
    traceback.print_exc()
    raise

finally:
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
