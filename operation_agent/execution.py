import os
import subprocess
import selectors
import sys

DIM   = "\033[2m"
CYAN  = "\033[96m"
RESET = "\033[0m"

def execute_script(script_name, work_dir=".", device="0"):
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
                    stdout_lines.append(line)
                else:
                    print(f"  {DIM}{line}{RESET}", end="", flush=True)
                    stderr_lines.append(line)

        # drain remaining output
        for line in process.stdout:
            print(f"  {line}", end="", flush=True)
            stdout_lines.append(line)
        for line in process.stderr:
            print(f"  {DIM}{line}{RESET}", end="", flush=True)
            stderr_lines.append(line)

        print(f"{DIM}{'─' * 62}{RESET}\n")

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
