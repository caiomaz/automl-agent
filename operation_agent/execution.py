import os
import subprocess
import selectors
import sys
from pathlib import Path
from typing import Optional

DIM   = "\033[2m"
CYAN  = "\033[96m"
RESET = "\033[0m"


def _tee(line: str, log_fh) -> None:
    """Write *line* (already has newline) to the open log file handle."""
    if log_fh is not None:
        log_fh.write(line)
        log_fh.flush()


def execute_script(script_name, work_dir=".", device="0", log_path: Optional[str] = None):
    """Execute *script_name* in *work_dir* and capture output.

    Parameters
    ----------
    log_path:
        Optional path to a file where all stdout+stderr lines are appended
        (in addition to printing them to the console).  The file is opened
        in append mode so multiple runs/retries accumulate in order.
    """
    if not os.path.exists(os.path.join(work_dir, script_name)):
        raise Exception(f"The file {script_name} does not exist.")
    try:
        script_path = script_name
        cmd = f"CUDA_VISIBLE_DEVICES={device} {sys.executable} -u {script_path}"
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, shell=True, cwd=work_dir,
        )

        stdout_lines = []
        stderr_lines = []

        print(f"\n{DIM}{'─' * 62}{RESET}")
        print(f"{CYAN}  ▶ Running: {script_name}{RESET}")
        print(f"{DIM}{'─' * 62}{RESET}")

        log_fh = None
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "a", encoding="utf-8")
            log_fh.write(f"\n{'─' * 62}\n  ▶ Running: {script_name}\n{'─' * 62}\n")
            log_fh.flush()

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        while process.poll() is None and selector.get_map():
            events = selector.select(timeout=1)
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    selector.unregister(key.fileobj)
                    continue
                if key.fileobj == process.stdout:
                    print(f"  {line}", end="", flush=True)
                    _tee(line, log_fh)
                    stdout_lines.append(line)
                else:
                    print(f"  {DIM}{line}{RESET}", end="", flush=True)
                    _tee(line, log_fh)
                    stderr_lines.append(line)

        # drain remaining output
        for line in process.stdout:
            print(f"  {line}", end="", flush=True)
            _tee(line, log_fh)
            stdout_lines.append(line)
        for line in process.stderr:
            print(f"  {DIM}{line}{RESET}", end="", flush=True)
            _tee(line, log_fh)
            stderr_lines.append(line)

        print(f"{DIM}{'─' * 62}{RESET}\n")

        if log_fh is not None:
            log_fh.write(f"{'─' * 62}\n")
            log_fh.close()

        return_code = process.returncode
        if return_code != 0:
            observation = "".join(stderr_lines)
        else:
            observation = "".join(stdout_lines)
        if observation == "" and return_code == 0:
            observation = "".join(stderr_lines)

        return return_code, "The script has been executed. Here is the output:\n" + observation

    except Exception as e:
        return -1, f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."
