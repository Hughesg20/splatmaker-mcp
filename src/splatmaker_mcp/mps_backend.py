"""Metal/MPS Gaussian-splat engine for splatmaker-mcp (macOS Apple Silicon).

Pipeline per job:
  1. COLMAP structure-from-motion via pycolmap's incremental mapper
     (images dir -> database.db + sparse/0), or ffmpeg frame extraction
     from a video first.
  2. 3DGS training with tchauffi/metalsplat on the torch `mps` backend,
     adapted from examples/train_garden.py (same loss, densify/seed/prune
     schedules, best-checkpoint PLY export).
  3. Export standard .ply via metalsplat.save_ply — the same format
     Nerfstudio exports and that MetalSplatter/SuperSplat viewers read.

Selected with SPLATMAKER_ENGINE=mps. Requires `uv sync --extra mps`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("splatmaker.mps")

# torch and pycolmap each bundle an OpenMP runtime; the process aborts on the
# second unless this is set BEFORE both are imported (metalsplat does it in
# its colmap loader, but only if that runs before our own import — set it here
# to be safe).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class MpsBackend:
    """Same duck-typed interface as SplatBackend: is_configured(),
    generate_from_video(), generate_from_images() -> EngineResult."""

    def __init__(self, max_iterations: int = 7000) -> None:
        self.engine = "mps"
        # MPS is ~10-20x slower than an RTX 4090 for this workload; 7000 iters
        # is the pragmatic default (nerfstudio path uses 15000).
        self.max_iterations = int(
            os.environ.get("SPLATMAKER_MAX_ITERATIONS", max_iterations)
        )
        self._work_root = Path.home() / ".splatmaker-mcp" / "jobs"
        self._work_root.mkdir(parents=True, exist_ok=True)
        # When the images dir already contains a COLMAP model (sparse/0/*.bin
        # next to the images), skip our own SfM and load it directly.
        self._precomputed_sparse: Path | None = None

    def is_configured(self) -> bool:
        try:
            import torch  # noqa: F401

            return torch.backends.mps.is_available()
        except Exception:
            return False

    # -- public async API (mirrors SplatBackend) -----------------------------

    async def generate_from_video(self, video_path: str, job_id: str):
        from .server import EngineResult  # local import: avoid circularity

        p = Path(video_path)
        if not p.exists():
            return EngineResult(ok=False, message=f"video not found: {video_path}")
        job_dir = self._work_root / job_id
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        # ~2 fps, capped at 80 frames — enough parallax for SfM without
        # exploding training time.
        rc = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-i", str(p), "-vf", "fps=2", "-frames:v", "80",
             str(frames_dir / "f%04d.jpg")],
            capture_output=True,
            text=True,
        )
        if rc.returncode != 0 or not list(frames_dir.glob("*.jpg")):
            return EngineResult(ok=False, message=f"ffmpeg frame extraction failed: {rc.stderr[-800:]}")
        return await self._run_job(str(frames_dir), job_id, source="video")

    async def generate_from_images(self, image_paths: list[str], job_id: str):
        from .server import EngineResult

        if not image_paths:
            return EngineResult(ok=False, message="image_paths is empty")
        job_dir = self._work_root / job_id
        img_dir = job_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for src in image_paths:
            src_p = Path(src)
            if not src_p.exists():
                return EngineResult(ok=False, message=f"image not found: {src}")
            shutil.copy2(src_p, img_dir / src_p.name)
        return await self._run_job(str(img_dir), job_id, source="images")

    # -- internals -----------------------------------------------------------

    async def _run_job(self, image_dir: str, job_id: str, source: str):
        from .server import EngineResult

        if not self.is_configured():
            return EngineResult(
                ok=False,
                message="mps engine not available: torch missing or MPS unavailable. "
                "Run `uv sync --extra mps` on Apple Silicon.",
            )
        job_dir = self._work_root / job_id
        scene_dir = job_dir / "colmap_scene"
        (scene_dir / "images").mkdir(parents=True, exist_ok=True)
        for f in Path(image_dir).glob("*"):
            shutil.copy2(f, scene_dir / "images" / f.name)
        # Pre-computed COLMAP model shipped alongside the images?
        for cand in (Path(image_dir).parent / "sparse" / "0", Path(image_dir) / "sparse" / "0"):
            if cand.exists() and (cand / "cameras.bin").exists():
                sparse_dst = scene_dir / "sparse"
                sparse_dst.mkdir(parents=True, exist_ok=True)
                shutil.copytree(cand, sparse_dst / "0", dirs_exist_ok=True)
                logger.info("using pre-computed COLMAP model at %s", cand)
                break

        def work() -> EngineResult:
            return self._train_sync(scene_dir, job_dir, source)

        return await asyncio.to_thread(work)

    def _map_scene(self, scene_dir: Path) -> None:
        """COLMAP incremental mapping via pycolmap into scene_dir/sparse/0.

        Notes from getting this to work with pycolmap 4.3 (COLMAP 4.3.0.dev0):
        - the database file must exist first (Database.open creates it);
          import_images/extract_features/match_* all hard-fail otherwise.
        - all images are assigned to ONE shared PINHOLE camera (f=960, 800x600)
          via camera_mode=SINGLE + a camera overwrite; with AUTO's per-image
          SIMPLE_RADIAL cameras the initial-pair triangulation kept failing
          ("Provided pair is unsuitable") because mixed image aspect ratios
          (some review photos are portrait) skewed the shared-focal estimate.
        - mapper init/registration thresholds are relaxed well below COLMAP
          defaults (init_min_num_inliers 100/tri 16 deg -> 8/0.5) because the
          photos are small (370-800px) review shots, not DSLR captures.
        - the initial pair is forced to the strongest verified pair from
          read_two_view_geometries() (id packing is uint64: id1<<32|id2,
          masked to 31 bits).
        """
        import pycolmap

        db = scene_dir / "database.db"
        sparse = scene_dir / "sparse"
        sparse.mkdir(parents=True, exist_ok=True)
        images = scene_dir / "images"
        pycolmap.Database.open(str(db))
        pycolmap.import_images(str(db), str(images), camera_mode=pycolmap.CameraMode.SINGLE)
        # normalize the single shared camera to PINHOLE 960/960/400/300
        import numpy as np

        d = pycolmap.Database.open(str(db))
        d.update_camera(
            pycolmap.Camera.create_from_model_id(
                1, pycolmap.CameraModelId.PINHOLE, 960.0, 800, 600
            )
        )
        d.close()
        pycolmap.extract_features(str(db), str(images))
        pycolmap.match_exhaustive(str(db))
        d2 = pycolmap.Database.open(str(db))
        ids, geoms = d2.read_two_view_geometries()
        d2.close()

        opts = pycolmap.IncrementalPipelineOptions()
        opts.min_num_matches = 8
        opts.min_model_size = 1
        opts.max_num_models = 1
        opts.multiple_models = False
        opts.init_num_trials = 500
        opts.triangulation.min_angle = 0.5
        mo = opts.mapper
        mo.init_min_num_inliers = 8
        mo.init_min_tri_angle = 0.5
        mo.init_max_error = 10.0
        mo.init_max_forward_motion = 1.0
        mo.init_max_reg_trials = 10
        mo.abs_pose_min_num_inliers = 8
        mo.abs_pose_min_inlier_ratio = 0.05
        mo.abs_pose_max_error = 10.0
        mo.filter_min_tri_angle = 0.5
        mo.filter_max_reproj_error = 10.0
        # Try each of the top verified pairs as the forced initial pair —
        # which pair initializes successfully is data-dependent (observed:
        # the strongest pair often fails, weaker ones succeed), so iterate.
        rows = []
        for pid, g in zip(ids, geoms):
            if g is None or g.inlier_matches is None:
                continue
            u = pid & 0xFFFFFFFFFFFFFFFF
            rows.append((len(g.inlier_matches), (u >> 32) & 0x7FFFFFFF, u & 0x7FFFFFFF))
        rows.sort(reverse=True)
        recons = {}
        for _n, id1, id2 in rows[:8]:
            opts.init_image_id1 = id1
            opts.init_image_id2 = id2
            trial = pycolmap.incremental_mapping(
                str(db), str(images), str(sparse), options=opts
            )
            if trial:
                recons = trial
                logger.info("colmap init succeeded on pair %d-%d", id1, id2)
                break
        if not recons:
            raise RuntimeError("COLMAP produced no reconstruction (no viable initial pair)")
        best = max(recons.values(), key=lambda r: r.num_images())
        target = sparse / "0"
        target.mkdir(parents=True, exist_ok=True)  # Reconstruction.write requires the dir to exist
        best.write(str(target))  # writes cameras/images/points3D .bin files
        n = best.num_images()
        if n < 3:
            raise RuntimeError(
                f"COLMAP registered only {n} images — need at least 3 with overlap"
            )
        logger.info("colmap registered %d images, %d points", n, best.num_points3D())

    def _train_sync(self, scene_dir: Path, job_dir: Path, source: str):
        from .server import EngineResult

        try:
            import torch
            from metalsplat import GaussianModel, render, save_ply
            from metalsplat.camera import camera_extent
            from metalsplat.data.colmap import load_colmap_scene
            from metalsplat.densify import densify_and_prune, prune_low_opacity, reset_opacity
            from metalsplat.filter3d import compute_3d_filter
            from metalsplat.losses import gaussian_splatting_loss, psnr
            from metalsplat.optim import SparseAdam, sh_lr_scale
            from metalsplat.seed import seed_uncovered_regions
            from metalsplat.training import (
                GaussianTrainingState,
                calibrate_initial_scale,
                estimate_scene_scale,
            )
        except ImportError as exc:
            return EngineResult(
                ok=False,
                message=f"metalsplat not installed ({exc}). Run `uv sync --extra mps`.",
            )

        device = "mps"
        num_iters = self.max_iterations
        try:
            if (scene_dir / "sparse" / "0" / "cameras.bin").exists():
                logger.info("skipping SfM: pre-computed COLMAP model present")
            else:
                self._map_scene(scene_dir)
            scene = load_colmap_scene(scene_dir, device=device, downscale=2.0)
        except Exception as exc:
            return EngineResult(ok=False, message=f"COLMAP stage failed: {exc}")

        n_images = len(scene.cameras)
        eval_idx = list(range(0, n_images, 8))
        train_idx = [i for i in range(n_images) if i not in set(eval_idx)]
        if not train_idx:
            train_idx = list(range(n_images))
            eval_idx = []
        train_cameras = [scene.cameras[i] for i in train_idx]

        scene_scale = estimate_scene_scale(scene.points)
        scene_scale = max(scene_scale, 1e-6)  # degenerate point cloud guard
        init_scale = calibrate_initial_scale(scene.points, train_cameras, scene_scale)
        n_points = scene.points.shape[0]
        model = GaussianModel(
            scene.points,
            scales=torch.full((n_points, 3), init_scale, device=device),
            colors=scene.colors,
            opacities=torch.full((n_points,), 0.1, device=device),
            sh_degree=3,
        ).to(device)

        lr_means_init = 0.02 * scene_scale
        lr_means_final = lr_means_init * 0.01
        extent = camera_extent(train_cameras)
        max_world_size = 0.1 * extent
        split_scale = 0.01 * extent
        background = torch.zeros(3, device=device)

        def make_optimizer(m):
            color_group = {
                "params": [m.raw_sh],
                "lr": 0.0025,
                "lr_scale": sh_lr_scale(m.raw_sh),
            }
            return SparseAdam(
                [
                    {"params": [m.means], "lr": lr_means_init},
                    {"params": [m.raw_scales], "lr": 0.005},
                    {"params": [m.raw_quats], "lr": 0.001},
                    color_group,
                    {"params": [m.raw_opacities], "lr": 0.05},
                ]
            )

        def compute_filter_3d(m):
            return compute_3d_filter(
                m.means.detach(), train_cameras, sampling_scale=0.2
            )

        state = GaussianTrainingState(
            model, make_optimizer, filter_3d=compute_filter_3d(model)
        )
        densify_threshold = None
        best = (float("-inf"), 0)
        out_dir = job_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        ply_best = out_dir / "splat_best.ply"
        ply_final = out_dir / "splat.ply"

        def eval_and_save(step):
            nonlocal best
            if not eval_idx:
                return
            with torch.no_grad():
                psnrs = []
                for idx in eval_idx:
                    pred = render(
                        state.model,
                        scene.cameras[idx],
                        background=background,
                        filter_3d=state.filter_3d,
                    )
                    psnrs.append(psnr(pred, scene.images[idx]).item())
                torch.mps.synchronize()
                mean_psnr = sum(psnrs) / len(psnrs)
            logger.info("step %d eval PSNR %.2f dB", step, mean_psnr)
            if mean_psnr > best[0]:
                best = (mean_psnr, step)
                save_ply(state.model, ply_best, filter_3d=state.filter_3d)

        import time

        start = time.time()
        for step in range(1, num_iters + 1):
            t = step / num_iters
            state.optimizer.param_groups[0]["lr"] = lr_means_init * (
                lr_means_final / lr_means_init
            ) ** t
            idx = train_idx[int(torch.randint(len(train_idx), (1,)).item())]
            cam = scene.cameras[idx]
            target = scene.images[idx]
            state.optimizer.zero_grad()
            aux = render(
                state.model,
                cam,
                background=background,
                return_aux=True,
                abs_grad_accum=state.grad_accum,
                filter_3d=state.filter_3d,
            )
            pred, valid, final_T = aux.image, aux.valid, aux.final_T
            loss = gaussian_splatting_loss(pred, target, lambda_dssim=0.2)
            loss.backward()
            visible = valid > 0.5
            state.optimizer.step(visible)
            state.record_visibility(visible)

            if step % 25 == 0 or step == 1:
                torch.mps.synchronize()
                logger.info(
                    "step %5d loss %.5f n %d (%.2fs/step)",
                    step, loss.item(), state.model.num_points,
                    (time.time() - start) / step,
                )

            if 1000 <= step <= min(15000, num_iters) and step % 250 == 0:
                model2, stats = densify_and_prune(
                    state.model,
                    state.grad_accum,
                    state.grad_count,
                    scene_scale=split_scale,
                    grad_percentile=0.9,
                    max_points=5_000_000,
                    grad_threshold=densify_threshold,
                    max_world_size=max_world_size if step > 1500 else None,
                )
                if densify_threshold is None and stats.grad_threshold is not None:
                    densify_threshold = stats.grad_threshold
                state.replace_model(model2, stats.source_index, stats.parent_index)
                state.reset_accumulators()

            if 100 <= step <= min(4000, num_iters) and step % 250 == 0:
                model2, seed_stats = seed_uncovered_regions(
                    state.model, cam, pred.detach(), target, final_T,
                    init_scale=init_scale, residual_thresh=0.15,
                    coverage_thresh=0.8, max_seeds_per_call=300, near=0.2,
                    max_points=5_000_000,
                )
                state.replace_model(model2, seed_stats.source_index, seed_stats.parent_index)

            if step <= 3000 and step % 1500 == 0:
                reset_opacity(state.model, value=0.01, optimizer=state.optimizer)

            if step % 100 == 0:
                model2, n_pruned, prune_index = prune_low_opacity(
                    state.model, prune_opacity_thresh=0.005
                )
                state.replace_model(model2, prune_index)

            if step % 1000 == 0:
                eval_and_save(step)

        eval_and_save(num_iters)
        save_ply(state.model, ply_final, filter_3d=state.filter_3d)
        final_ply = ply_best if ply_best.exists() else ply_final
        return EngineResult(
            ok=True,
            message=f"done ({source}, {num_iters} iters, best PSNR {best[0]:.2f} dB @ {best[1]})",
            asset_paths={
                "ply": str(final_ply),
                "job_dir": str(job_dir),
            },
        )
