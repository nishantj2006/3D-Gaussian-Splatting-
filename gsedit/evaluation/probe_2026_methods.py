"""Non-inference upstream startup probes; never a visual-quality benchmark.

Run in separate upstream working directories without replacing installed CUDA
extensions. Output is new-only and records commits, stdout, timing and RSS.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


METHODS = {
    "Split_and_Splat": "train.py",
    "Inpaint360GS": "edit_object_inpaint.py",
    "GPGS": "train_compose.py",
    "CoIn": "train_gs.py",
}


def probe(repo, entry, output, python, bindings_root=None):
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    if bindings_root is not None:
        env["PYTHONPATH"] = str(bindings_root.resolve() / repo.name)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["WANDB_MODE"] = "disabled"
    log = output / f"{repo.name}.log"
    command = [python, entry, "--help"]
    started = time.perf_counter()
    with log.open("w") as stream:
        proc = subprocess.Popen(
            command,
            cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        timed_out = False
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                code = os.waitstatus_to_exitcode(status)
                proc.returncode = code
                break
            elapsed = time.perf_counter() - started
            if elapsed > 45:
                import signal
                os.killpg(proc.pid, signal.SIGKILL if elapsed > 50 else signal.SIGTERM)
                timed_out = True
            time.sleep(.1)
    lines = log.read_text(errors="replace").splitlines()
    errors = [line for line in lines if "Error:" in line]
    rss = usage.ru_maxrss / 1024
    return {
        "method": repo.name, "commit": commit, "command": command,
        "exit_code": code, "timed_out": timed_out,
        "seconds": time.perf_counter() - started, "startup_peak_rss_mib": rss,
        "status": "startup_passed_not_inference" if code == 0 else "startup_blocked",
        "last_error": errors[-1] if errors else None,
        "log": str(log), "inference_run": False, "visual_quality_measured": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bindings-root", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, entry in METHODS.items():
        row = probe(args.upstream_root.resolve() / name, entry, output, args.python, args.bindings_root)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (output / "report.json").write_text(json.dumps({
            "scope": "startup_only_no_scene_edit_or_generation", "methods": rows,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
