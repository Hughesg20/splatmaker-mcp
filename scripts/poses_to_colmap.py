"""Build a COLMAP sparse model from hand-entered camera poses.

Input: a JSON file mapping image filename -> camera spec:

```json
{
  "fx": 960.0, "fy": 960.0, "cx": 400.0, "cy": 300.0,
  "width": 800, "height": 600,
  "cameras": {
    "138205.jpg": {"pos": [x, y, z], "look_at": [x, y, z]},
    ...
  }
}
```

All images share one PINHOLE camera (same convention as the backend's SfM
path). Writes <scene>/sparse/0/{cameras,images,points3D}.bin plus a small
random seed point cloud around the origin (the trainer needs a non-empty
point cloud for scale calibration). Images must live in <scene>/images/.

Usage:
  uv run python scripts/poses_to_colmap.py poses.json /path/to/scene
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pycolmap


def look_at_R(pos, target, up=(0.0, 0.0, 1.0)):
    """World->camera (R, t) for a camera at `pos` looking at `target`.

    Matches metalsplat.camera.Camera.look_at exactly: camera +Z is the
    *forward* direction, +Y is image-down (v grows downward), rows of R_wc
    are the camera axes in world space.
    """
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R_wc = np.stack([right, down, forward], axis=0)
    t_wc = -R_wc @ pos
    return R_wc, t_wc


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    spec = json.loads(Path(sys.argv[1]).read_text())
    scene = Path(sys.argv[2])
    # accept either <scene>/images/ or images directly at the scene root
    img_dir = scene / "images"
    if not img_dir.exists() and any(scene.glob("*.jpg")) or any(scene.glob("*.png")):
        img_dir = scene
    if not img_dir.exists():
        sys.exit(f"expected images in {scene}/images/ or {scene}/")
    sparse0 = scene / "sparse" / "0"
    sparse0.mkdir(parents=True, exist_ok=True)

    fx = float(spec.get("fx", 960.0))
    fy = float(spec.get("fy", fx))
    w = int(spec.get("width", 800))
    h = int(spec.get("height", 600))
    cx = float(spec.get("cx", w / 2))
    cy = float(spec.get("cy", h / 2))

    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera.create_from_model_id(
        1, pycolmap.CameraModelId.PINHOLE, fx, w, h
    )
    cam.params = np.array([fx, fy, cx, cy], dtype=np.float64)
    rec.add_camera(cam)
    # trivial-frame path requires a rig with exactly one sensor = the camera
    rig = pycolmap.Rig()
    rig.rig_id = 1
    sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 1)  # sensor id 1 == camera_id 1
    rig.add_ref_sensor(sensor)
    rec.add_rig(rig)

    names = sorted(spec["cameras"].keys())
    available = {p.name for p in img_dir.iterdir()}
    missing = [n for n in names if n not in available]
    if missing:
        sys.exit(f"images not found in {img_dir}: {missing}")

    for i, name in enumerate(names, start=1):
        c = spec["cameras"][name]
        R, t = look_at_R(c["pos"], c["look_at"])
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        img = pycolmap.Image(name=name, camera_id=1, image_id=i)
        rec.add_image_with_trivial_frame(img, pose)

    # seed point cloud: a small grid around the origin plus a ring at radius 1
    # so hand-placed cameras looking at the origin always have points in view
    rng = np.random.default_rng(0)
    n = 2000
    pts = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float64)
    ring = 500
    ang = rng.uniform(0, 2 * np.pi, ring)
    pts = np.vstack(
        [pts, np.stack([np.cos(ang), np.sin(ang), np.zeros(ring)], axis=1)]
    ).astype(np.float64)
    for j in range(pts.shape[0]):
        track = pycolmap.Track()
        rec.add_point3D(pts[j], track, color=np.array([128, 128, 128], dtype=np.uint8))

    rec.write(str(sparse0))
    print(f"wrote {sparse0}: {len(names)} cameras, {pts.shape[0]} seed points, PINHOLE fx={fx} {w}x{h}")


if __name__ == "__main__":
    main()
