# Swift 1.5 IQ3_S with llama.cpp in WSL

Run Swift 1.5 Qwen3.8-27B GSQ-RCO IQ3_S with its embedded MTP head, a
131,072-token context, and an OpenAI-compatible API at `http://127.0.0.1:1235/v1`.
Everything lives under `~/ai/llama-swift15/` in WSL.

The tested setup is Ubuntu 24.04.3 WSL2, an RTX 5090 Laptop GPU with 24 GB VRAM,
Windows driver 592.27, and the official llama.cpp **b11393 CUDA 12.8** prebuilt.
It uses about 18.6 GiB of whole-GPU VRAM after loading. See [HANDOFF.md](HANDOFF.md)
for the exact versions, memory measurements, and initial benchmark results.

## Prepare the runtime

Run the following commands in **Bash inside WSL**. You need `curl`, `tar`,
`sha256sum`, and working GPU passthrough (`nvidia-smi`). These commands extract
the prebuilt engine and CUDA runtime libraries locally; no source build or CUDA
toolkit installation is needed. Keep the Windows GPU driver unchanged and do
not install a Linux NVIDIA driver inside WSL.

Skip this section if `llama.cpp-runtime/llama-server` and its bundled libraries
are already present.

```bash
set -euo pipefail
ROOT="$HOME/ai/llama-swift15"
mkdir -p "$ROOT/downloads" "$ROOT/llama.cpp-runtime"
cd "$ROOT/downloads"

RELEASE="https://github.com/ggml-org/llama.cpp/releases/download/b11393"
curl -fL --retry 3 -C - -O "$RELEASE/llama-b11393-bin-ubuntu-cuda-12.8-x64.tar.gz"
curl -fL --retry 3 -C - -O "$RELEASE/cudart-llama-b11393-bin-ubuntu-cuda-12.8-x64.tar.gz"

sha256sum --check <<'SUMS'
bd8a074ee086e954eab41159de4b0c1c4d4b7204d79291a3d49e0ac5b642e93f  llama-b11393-bin-ubuntu-cuda-12.8-x64.tar.gz
1acadcd2ffd86b566940d36a14be4adbd83b1e04f36f54b9e1790bbffadcb864  cudart-llama-b11393-bin-ubuntu-cuda-12.8-x64.tar.gz
SUMS

tar -xzf llama-b11393-bin-ubuntu-cuda-12.8-x64.tar.gz \
  -C "$ROOT/llama.cpp-runtime" --strip-components=1
tar -xzf cudart-llama-b11393-bin-ubuntu-cuda-12.8-x64.tar.gz \
  -C "$ROOT/llama.cpp-runtime" --strip-components=1

LD_LIBRARY_PATH="$ROOT/llama.cpp-runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROOT/llama.cpp-runtime/llama-server" --list-devices
```

The device list should include the RTX 5090. Archive checksums above are from
the [official b11393 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11393).

## Download and verify the model

Authoritative source:
[UkisAI Swift GSQ-RCO GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF).
This setup pins revision `d74895bbe5db4bec1e0024e7cc87d59c02d7631a`.

| Property | Value |
| --- | --- |
| Filename | `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` |
| Size | 12,120,016,896 bytes / 11.2876 GiB |
| SHA-256 | `9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5` |

The pinned repository's `release-manifest.json` supplies the exact size, and its
`SHA256SUMS` supplies the hash. Download only the IQ3_S **`-mtp.gguf`** file.
No `.ninfer` artifact, separate draft model, or conversion is required.

```bash
set -euo pipefail
ROOT="$HOME/ai/llama-swift15"
FILE="Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
REV="d74895bbe5db4bec1e0024e7cc87d59c02d7631a"
SOURCE="https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/resolve/$REV"
mkdir -p "$ROOT/models" "$ROOT/downloads"

curl -fL --retry 3 "$SOURCE/SHA256SUMS" -o "$ROOT/downloads/SHA256SUMS"
curl -fL --retry 3 "$SOURCE/release-manifest.json" -o "$ROOT/downloads/release-manifest.json"
EXPECTED_SHA=$(awk -v file="$FILE" '$2 == file { print $1 }' "$ROOT/downloads/SHA256SUMS")
test "$EXPECTED_SHA" = "9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5"

MODEL="$ROOT/models/$FILE"
if [[ ! -f "$MODEL" ]]; then
  curl -fL --retry 3 -C - "$SOURCE/$FILE" -o "$MODEL.part"
  test "$(stat -c %s "$MODEL.part")" = "12120016896"
  printf '%s  %s\n' "$EXPECTED_SHA" "$MODEL.part" | sha256sum --check
  mv "$MODEL.part" "$MODEL"
fi

test "$(stat -c %s "$MODEL")" = "12120016896"
printf '%s  %s\n' "$EXPECTED_SHA" "$MODEL" | sha256sum --check
```

Interrupted downloads resume from `.part`. A valid existing model is verified
without downloading again. Any size or checksum failure stops the commands;
do not start the server with that file.

## Start and stop the server

Ensure port 1235 is free and GPU memory is available. From WSL:

```bash
nvidia-smi
chmod +x ~/ai/llama-swift15/start_swift15_iq3s_128k.sh
~/ai/llama-swift15/start_swift15_iq3s_128k.sh
```

Leave this terminal open. Stop the server with **Ctrl+C**. If the server from
an earlier session is still running, stop that instance before starting another.

The launcher enables all GPU layers, Flash Attention, one server slot, Jinja
chat templates, native MTP with up to three draft tokens, and a 131,072-token
context. Main K/V cache uses `q8_0`; the MTP draft cache uses upstream F16 defaults.
Its model ID is `swift-1.5-qwen3.8-27b-iq3s-mtp`. It binds only to `127.0.0.1`.

The launcher defaults to `--load-mode dio` (Direct I/O where supported) to
avoid filling WSL's filesystem page cache with the model during loading.
It still needs RAM for CPU tensors, metadata, and staging buffers; this does
not eliminate RAM use or guarantee against OOM. The previous default, `mmap`,
uses reclaimable file-backed pages and does not lock the model in RAM;
`mlock` is the mode that forces residency. Direct I/O may change loading speed
and depends on filesystem support. To restore the original loading behavior:

```bash
SWIFT_LOAD_MODE=auto ~/ai/llama-swift15/start_swift15_iq3s_128k.sh
```

To save logs while keeping Ctrl+C available:

```bash
mkdir -p ~/ai/llama-swift15/logs
~/ai/llama-swift15/start_swift15_iq3s_128k.sh 2>&1 \
  | tee ~/ai/llama-swift15/logs/server.log
```

If 128K fails because of VRAM allocation, keep IQ3_S and `q8_0` K/V and try
contexts of 114688, 98304, then 81920. For example:

```bash
SWIFT_CONTEXT=114688 ~/ai/llama-swift15/start_swift15_iq3s_128k.sh
```

## Check the API and use OpenCode

In a second WSL terminal:

```bash
curl -fsS http://127.0.0.1:1235/health
curl -fsS http://127.0.0.1:1235/v1/models
curl -fsS http://127.0.0.1:1235/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"swift-1.5-qwen3.8-27b-iq3s-mtp","messages":[{"role":"user","content":"What is 2 + 2? Answer with the number only."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Health should return `{"status":"ok"}` once loading finishes. Windows localhost
access was also verified; from PowerShell, use:

```powershell
curl.exe http://localhost:1235/health
```

The live global OpenCode config on this machine already has provider `llama.cpp`
pointing to `http://127.0.0.1:1235/v1`, with a 131072-token context limit and an
8192-token output limit. Select it with `/models`, or run:

```bash
opencode models llama.cpp
opencode -m llama.cpp/swift-1.5-qwen3.8-27b-iq3s-mtp
```

The server must be running before OpenCode sends requests. The OpenCode config
is separate from this repository and is not installed by these commands.

## Optional probes

With the server running, `python3 benchmark.py` measures synthetic short, 8K,
and 32K prompts and saves results under `logs/`. Repetitive prompts can produce
optimistic MTP acceptance; this is not a coding-quality benchmark.

`python3 test_startup.py` runs separate ordered startup checks and starts/stops
its own server processes. Stop the existing server first, since these checks
also use port 1235. The script leaves its successful final server running.

`.gitignore` excludes models, runtime binaries/libraries, downloads, logs,
process IDs, and Python bytecode. Keep the launcher, probe scripts, and
documentation in Git.
