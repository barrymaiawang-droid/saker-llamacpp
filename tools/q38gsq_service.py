#!/usr/bin/env python3
"""Launch item 23 with native templates, full draft vocabulary and model defaults."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def launch_spec(config, env):
    profile = env.get("PROFILE", config.get("profile", "AR")).upper()
    if profile not in ("AR", "MTP2"):
        raise ValueError("PROFILE must be AR or MTP2")
    context = int(env.get("CTX", config.get("ctx", 262144)))
    budget = int(env.get("RBUDGET", 32768))
    if context < 512 or context > 262144 or budget == 0 or budget < -1:
        raise ValueError("CTX must be 512..262144; RBUDGET must be positive or -1")
    reasoning = env.get("REASONING", "on").lower()
    if reasoning not in ("on", "off", "true", "false", "1", "0"):
        raise ValueError("REASONING must be on or off")
    thinking = reasoning in ("on", "true", "1")
    kwargs = json.loads(env.get("LLAMA_CHAT_TEMPLATE_KWARGS", env.get("LLAMA_ARG_CHAT_TEMPLATE_KWARGS", "{}")))
    if not isinstance(kwargs, dict):
        raise ValueError("LLAMA_CHAT_TEMPLATE_KWARGS must be an object")
    effort = kwargs.get("reasoning_effort", env.get("LLAMA_ARG_REASONING_EFFORT", config.get("effort", "xhigh")))
    if effort not in ("low", "medium", "xhigh"):
        raise ValueError("reasoning_effort must be low, medium or xhigh")
    kwargs.update(enable_thinking=thinking, reasoning_effort=effort)
    kwargs.setdefault("preserve_thinking", True)
    child_env = dict(env)
    # The dynamic shortlist is measured to change target output. Keep full-head MTP.
    child_env.pop("SAKER_SHORTLIST", None)
    child_env.pop("LLAMA_CHAT_TEMPLATE_KWARGS", None)
    child_env["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"] = json.dumps(kwargs)
    child_env["LLAMA_ARG_REASONING_EFFORT"] = effort
    library_dir = str(Path(config["binary"]).resolve().parent)
    child_env["LD_LIBRARY_PATH"] = ":".join(p for p in [library_dir, env.get("LD_LIBRARY_PATH", "")] if p)
    cmd = [config["binary"], "-m", config["model"], "--alias", "qwen3.8-27b-gsq-rco-iq3s-mtp",
           "--host", env.get("HOST", "0.0.0.0"), "--port", str(int(env.get("PORT", 18151))),
           "-ngl", "99", "-c", str(context), "-b", "2048", "-ub", "256", "-np", "1", "-fa", "on",
           "-t", str(int(env.get("THREADS", config.get("threads", 4)))),
           "-tb", str(int(env.get("THREADS_BATCH", config.get("threads_batch", 4)))),
           "-n", str(budget), "--reasoning", "on" if thinking else "off", "--reasoning-format", "deepseek",
           "--temp", env.get("TEMP", "1.0" if thinking else "0.7"),
           "--top-p", env.get("TOP_P", "0.95" if thinking else "0.8"), "--top-k", env.get("TOP_K", "20"),
           "--min-p", env.get("MIN_P", "0"), "--presence-penalty", env.get("PRESENCE_PENALTY", "0" if thinking else "1.5"),
           "--repeat-penalty", "1", "--cache-ram", str(config.get("cache_ram_mib", 2048)), "--no-webui"]
    if profile == "MTP2":
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    return cmd, child_env


def process_start(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if stat[0] == "Z" else stat[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_state(path):
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    return state if state.get("start") and process_start(state["pid"]) == state["start"] else None


def main():
    config_path = Path(os.environ.get("Q38GSQ_CONFIG", Path(__file__).with_name("q38gsq-runtime.json")))
    config = json.loads(config_path.read_text())
    state_path = config_path.with_suffix(".pid.json")
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action not in ("start", "restart", "stop", "kill", "status", "logs", "dry-run"):
        raise ValueError("expected start|restart|stop|kill|status|logs|dry-run")
    # Validate before stopping an existing process on restart.
    cmd, child_env = launch_spec(config, os.environ) if action in ("start", "restart", "dry-run") else ([], {})
    if action == "dry-run":
        print(json.dumps({"command": cmd, "template_kwargs": json.loads(child_env["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"]),
                          "shortlist": child_env.get("SAKER_SHORTLIST")}, indent=2))
        return
    if action == "logs":
        os.execvp("tail", ["tail", "-n", "80", "-f", config["log"]])
    state = owned_state(state_path)
    if action == "status":
        print(json.dumps({"running": state is not None, "state": state}))
        return
    if action in ("stop", "kill", "restart") and state:
        os.kill(state["pid"], signal.SIGKILL if action == "kill" else signal.SIGTERM)
        deadline = time.monotonic() + 15
        while process_start(state["pid"]) == state["start"]:
            if time.monotonic() > deadline:
                raise RuntimeError("server did not stop; use kill explicitly")
            time.sleep(.1)
        state_path.unlink(missing_ok=True)
        state = None
    if action in ("stop", "kill"):
        print("q38gsq stopped")
        return
    if state:
        print("q38gsq already running")
        return
    with Path(config["log"]).open("ab") as log:
        proc = subprocess.Popen(cmd, env=child_env, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    state_path.write_text(json.dumps({"pid": proc.pid, "start": process_start(proc.pid), "command": cmd})+"\n")
    print(f"BOOT_ISSUED q38gsq pid={proc.pid}")


if __name__ == "__main__":
    main()
