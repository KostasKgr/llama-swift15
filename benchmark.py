#!/usr/bin/env python3
"""Small local throughput probes; synthetic prompts are not quality benchmarks."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:1235"


def post(route, payload):
    request = Request(BASE + route, json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=900) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 8192, 32768])
    parser.add_argument("--label", default="mtp")
    args = parser.parse_args()
    text = ("The observatory keeps a record of weather, instruments, and maintenance. "
            "Each entry includes a date, an observation, and a recommended action.\n") * 1600
    tokens = post("/tokenize", {"content": text, "add_special": True})["tokens"]
    results = []
    for count in args.sizes:
        assert len(tokens) >= count
        payload = {"prompt": tokens[:count], "n_predict": 128,
                   "temperature": 0.0, "seed": 42, "stream": True,
                   "cache_prompt": False, "ignore_eos": True}
        request = Request(BASE + "/completion", json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        start = time.monotonic()
        first = None
        final = None
        content = []
        with urlopen(request, timeout=900) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    continue
                event = json.loads(raw)
                if event.get("content"):
                    first = first or time.monotonic()
                    content.append(event["content"])
                if event.get("stop"):
                    final = event
        assert final is not None, "No final timing event received"
        result = {"prompt_tokens_requested": count,
                  "ttft_seconds": None if first is None else first - start,
                  "wall_seconds": time.monotonic() - start,
                  "timings": final.get("timings"),
                  "tokens_predicted": final.get("tokens_predicted"),
                  "tokens_evaluated": final.get("tokens_evaluated"),
                  "sample": "".join(content)[:300]}
        results.append(result)
        print(json.dumps(result), flush=True)
        (ROOT / f"logs/benchmark-{args.label}.json").write_text(json.dumps(results, indent=2))
        with urlopen(BASE + "/metrics", timeout=10) as response:
            (ROOT / f"logs/metrics-{args.label}-{count}.txt").write_bytes(response.read())


if __name__ == "__main__":
    main()
