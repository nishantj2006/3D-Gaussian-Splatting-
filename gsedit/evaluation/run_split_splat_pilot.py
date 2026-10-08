"""Run native Split&Splat instance training with bounded steps and live timing.

No scene merge, approval or semantic-feature transfer is performed here.
"""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-dir", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    if a.steps < 30:
        raise ValueError("Need at least 30 steps for warmed timing")
    out.mkdir(parents=True)
    started = time.perf_counter()
    with (out / "train.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        sys.path.insert(0, str(a.upstream_dir.resolve()))
        spec = importlib.util.spec_from_file_location("upstream_split_training", a.upstream_dir / "train.py")
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        import torch
        import numpy as np
        import scene
        # Upstream Scene otherwise overwrites the RNG with wall-clock seconds.
        scene.time = SimpleNamespace(time=lambda: a.seed)
        native.safe_state(True)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        parser = argparse.ArgumentParser()
        lp = native.ModelParams(parser)
        op = native.OptimizationParams(parser)
        pp = native.PipelineParams(parser)
        args = parser.parse_args(["-s", str(a.dataset_dir.resolve()), "-m", str(out / "native-model"),
                                  "--is_instance", "--eval", "--resolution", "270",
                                  "--iterations", str(a.steps), "--optimizer_type", "default",
                                  "--position_lr_max_steps", "1000", "--densify_from_iter", "100",
                                  "--densify_until_iter", str(a.steps - 50),
                                  "--depth_l1_weight_init", "0", "--depth_l1_weight_final", "0"])
        native.args = args
        original_report = native.training_report
        previous = time.perf_counter()
        samples = []
        forward_ms = []

        def measured_report(writer, iteration, l1, loss, l1_fn, elapsed, testing, current_scene, *rest):
            nonlocal previous
            torch.cuda.synchronize()
            now = time.perf_counter()
            duration = now - previous
            previous = now
            if iteration > 20:
                samples.append(duration)
                forward_ms.append(float(elapsed))
            original_report(writer, iteration, l1, loss, l1_fn, elapsed, testing, current_scene, *rest)
            if iteration % 10 == 0 or iteration == a.steps:
                recent = samples[-50:]
                median = float(np.median(recent)) if recent else None
                report = dict(scope="native_instance_training_pilot_not_full_removal", status="training",
                              iteration=iteration, target_iterations=a.steps, total_seconds=now-started,
                              warmed_median_seconds_per_iteration=median,
                              warmed_mean_seconds_per_iteration=float(np.mean(recent)) if recent else None,
                              median_forward_backward_ms=float(np.median(forward_ms[-50:])) if recent else None,
                              estimated_pilot_remaining_seconds=(a.steps-iteration)*median if median else None,
                              gaussians=int(current_scene.gaussians.get_xyz.shape[0]), loss=float(loss.item()),
                              peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                              gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                              seed=a.seed, held_out=["frame_0134", "frame_0141", "frame_0143"],
                              unrelated_source_scene_modified=False, visual_approval=False)
                (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")

        native.training_report = measured_report
        try:
            native.training(lp.extract(args), op.extract(args), pp.extract(args), [],
                            [a.steps], [a.steps], None, -1)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            (out / "failure.json").write_text(json.dumps(dict(status="failed", error=str(exc),
                                                              total_seconds=time.perf_counter()-started), indent=2))
            raise
        report = json.loads((out / "progress.json").read_text())
        report.update(status="completed_native_instance_pilot", total_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      warning="Native object PLY lacks our 128D scene schema; do not merge or approve it.")
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
