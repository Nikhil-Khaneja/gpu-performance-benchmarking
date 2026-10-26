"""Every benchmark script calls log_run() once at the end so RUN_LOG.txt is
a UUID-labelled audit trail tying each output file to the command and
parameters that produced it, per the deliverables list in the assignment.
"""
import datetime
import os
import sys

RUN_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RUN_LOG.txt")


def log_run(script_name, gpu_uuid, params: dict, output_paths):
    if isinstance(output_paths, str):
        output_paths = [output_paths]
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    params_str = " ".join(f"{k}={v}" for k, v in params.items())
    outputs_str = ",".join(output_paths)
    line = f"[{ts}] uuid={gpu_uuid} script={script_name} {params_str} outputs={outputs_str}\n"
    with open(RUN_LOG_PATH, "a") as f:
        f.write(line)
    print(f"logged run to {RUN_LOG_PATH}", file=sys.stderr)
