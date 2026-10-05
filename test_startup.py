#!/usr/bin/env python3
"""Run the ordered startup checks, keeping only a successful final server alive."""
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
SERVER = ROOT / "llama.cpp-runtime/llama-server"
ENV = dict(os.environ, LD_LIBRARY_PATH=str(SERVER.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
RESULTS = []


def save():
    (ROOT / "logs/startup-results.json").write_text(json.dumps(RESULTS, indent=2))


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def start(label, context, mtp, kv=None):
    command = [str(SERVER), "-m", str(MODEL), "-ngl", "999", "-fa", "on",
               "-c", str(context), "-np", "1", "--host", "127.0.0.1", "--port", "1235",
               "--jinja", "--alias", "swift-1.5-qwen3.8-27b-iq3s-mtp", "--metrics"]
    if kv:
        command += ["--cache-type-k", kv, "--cache-type-v", kv]
    if mtp:
        command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    log = ROOT / f"logs/{label}.log"
    print("Starting:", " ".join(command), flush=True)
    with log.open("w") as output:
        process = subprocess.Popen(command, env=ENV, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    record = {"label": label, "context": context, "mtp": mtp, "kv": kv or "f16",
              "command": command, "log": str(log), "pid": process.pid}
    RESULTS.append(record)
    deadline = time.monotonic() + 180
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with urlopen("http://127.0.0.1:1235/health", timeout=2) as response:
                health = json.load(response)
            if health.get("status") == "ok":
                record["health"] = health
                record["gpu_after_load"] = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
                    text=True).strip()
                save()
                print(label, "healthy; VRAM:", record["gpu_after_load"], flush=True)
                return process, record
        except Exception:
            pass
        time.sleep(1)
    if process.poll() is None:
        stop(process)
    record["exit_code"] = process.returncode
    record["error_log"] = log.read_text()[-18000:]
    save()
    print(record["error_log"], flush=True)
    return None, record


def chat(record):
    payload = {"model": "swift-1.5-qwen3.8-27b-iq3s-mtp",
               "messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number only."}],
               "max_tokens": 32, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    request = Request("http://127.0.0.1:1235/v1/chat/completions", json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        record["chat_response"] = json.load(response)
    save()
    print("Chat response:", json.dumps(record["chat_response"]), flush=True)


def main():
    assert MODEL.is_file(), "Verified model missing"
    for label, mtp in [("baseline-8k", False), ("mtp-8k", True)]:
        process, record = start(label, 8192, mtp)
        if process is None:
            raise RuntimeError(f"{label} failed; see {record['log']}")
        try:
            chat(record)
        finally:
            stop(process)
    for context in [131072, 114688, 98304, 81920]:
        process, record = start(f"mtp-{context}", context, True, "q8_0")
        if process is not None:
            try:
                chat(record)
            except Exception:
                stop(process)
                raise
            (ROOT / "logs/server.pid").write_text(str(process.pid) + "\n")
            print("Final server remains running:", process.pid, flush=True)
            return
        error = record["error_log"].lower()
        if not any(s in error for s in ["out of memory", "failed to allocate", "cudamalloc"]):
            raise RuntimeError("Startup failed for a reason other than allocation; no fallback attempted")
        print("Allocation failed; next attempt retains q8_0 and lowers context.", flush=True)
    raise RuntimeError("All requested context sizes failed")


if __name__ == "__main__":
    main()
