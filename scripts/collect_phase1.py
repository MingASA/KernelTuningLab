"""Research protocol wrapper: existing serial v2 workflow + read-only telemetry.

No alternate execution engine or measurement store. Resume with the same command.
Use timeout --signal=INT for an optional whole-study wall deadline on Linux/WSL.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import uuid

from kernel_tuning_lab.research import run_research


def collect(spec, output, interval=2):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    run = uuid.uuid4().hex
    path = output.with_name(output.name + "-telemetry.jsonl")
    def sample():
        while not stop.is_set():
            row = {"collection_id": run, "utc": datetime.now(timezone.utc).isoformat()}
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
                    "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
                row.update(csv=result.stdout.strip(), returncode=result.returncode)
            except (OSError, subprocess.TimeoutExpired) as error:
                row["error"] = type(error).__name__
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            stop.wait(interval)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        result = run_research(spec, output)
    finally:
        stop.set()
        thread.join(timeout=6)
    print(json.dumps(result, indent=2))
    return result.get("exit_code", 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(collect(args.spec, args.output))
