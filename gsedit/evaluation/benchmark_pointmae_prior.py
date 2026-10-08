"""Evaluate GPGS's pretrained geometry prior on masked observed floor patches.

This measures operator compatibility and reconstructing an artificial hole with
known context. It does not establish the depth of the wall hidden by the bed.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import random
import resource
import sys
import time
from types import ModuleType

from gsedit.evaluation.pointmae_knn_compat import KNN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["upstream-dir", "checkpoint", "source-ply", "excluded-ids", "floor-fit", "output-dir"]:
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(status="preparing", approved=False, actual_bed_background_reconstructed=False,
                  scope="pretrained_prior_on_masked_observed_floor_not_hidden_wall", seed=a.seed,
                  knn_adaptation="torch.cdist/topk; native Point-MAE only consumes indices")
    with (a.output_dir / "run.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            import numpy as np
            import torch
            import yaml
            from easydict import EasyDict
            from plyfile import PlyData, PlyElement
            np.random.seed(a.seed)
            random.seed(a.seed)
            torch.manual_seed(a.seed)
            sys.path.insert(0, str(a.upstream_dir.resolve()))
            shim = ModuleType("knn_cuda")
            shim.KNN = KNN
            sys.modules["knn_cuda"] = shim
            from models.Point_MAE import Point_MAE
            config = EasyDict(yaml.safe_load((a.upstream_dir / "cfgs/custom.yaml").read_text())["model"])
            config.transformer_config.mask_ratio = .4
            config.transformer_config.mask_type = "rand"
            model = Point_MAE(config).cuda().eval()
            # Official numeric checkpoint, restricted loading (no pickle code).
            checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
            state = checkpoint.get("base_model", checkpoint.get("state_dict", checkpoint))
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            report["checkpoint_sha256"] = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
            vertices = PlyData.read(str(a.source_ply))["vertex"].data
            xyz = np.column_stack([vertices[k] for k in ["x", "y", "z"]])
            floor = json.loads(a.floor_fit.read_text())
            origin = np.asarray(floor["plane_origin"])
            normal = np.asarray(floor["plane_normal_toward_removed_object"])
            excluded = np.load(a.excluded_ids)
            valid = np.abs((xyz-origin) @ normal) < .03
            valid[excluded] = False
            ids = np.flatnonzero(valid)
            if len(ids) < 1024:
                raise ValueError("Insufficient intact floor context")
            # One observed local patch around the median clean floor location.
            center = np.median(xyz[ids], axis=0)
            ids = ids[np.argsort(np.linalg.norm(xyz[ids]-center, axis=1))[:8192]]
            rng = np.random.default_rng(a.seed)
            ids = np.sort(rng.choice(ids, 1024, replace=False))
            points = xyz[ids]
            centroid = points.mean(axis=0)
            scale = float(np.linalg.norm(points-centroid, axis=1).max())
            if not scale > 0:
                raise ValueError("Degenerate patch")
            tensor = torch.from_numpy(((points-centroid)/scale).astype(np.float32))[None].cuda()
            timings = []
            with torch.no_grad():
                for repeat in range(5):
                    random.seed(a.seed)
                    np.random.seed(a.seed)
                    torch.manual_seed(a.seed)
                    torch.cuda.synchronize()
                    before = time.perf_counter()
                    full, observed, centers, rebuilt = model(tensor, vis=True)
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter()-before)
                if not torch.isfinite(rebuilt).all():
                    raise ValueError("Non-finite reconstructed points")
                predicted = rebuilt.cpu().numpy()*scale+centroid
                # Distance to fitted plane is diagnostic, not hidden-geometry GT.
                distances = np.abs((predicted-origin) @ normal)
            dtype = [(k, "f4") for k in ["x", "y", "z"]]
            cloud = np.empty(len(predicted), dtype=dtype)
            for i, key in enumerate(["x", "y", "z"]):
                cloud[key] = predicted[:, i]
            PlyData([PlyElement.describe(cloud, "vertex")]).write(str(a.output_dir / "predicted-points-diagnostic.ply"))
            np.save(a.output_dir / "observed-source-ids.npy", ids)
            np.save(a.output_dir / "context-points.npy", points)
            report.update(status="completed_prior_probe", checkpoint_loaded_strictly=True,
                          output_points=len(predicted), median_inference_seconds=float(np.median(timings[1:])),
                          median_distance_to_observed_plane=float(np.median(distances)),
                          p95_distance_to_observed_plane=float(np.percentile(distances, 95)),
                          total_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                          gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                          warning="Masked-group centers are derived from intact context; this cannot validate unseen wall depth.")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            report.update(status="failed", error=repr(exc), total_seconds=time.perf_counter()-started)
        (a.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
