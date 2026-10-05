# Swift 1.5 cold-prefill report — WSL2, 2026-10-05

The batch sweep did not show a clear throughput breakthrough. 4096/1024 had
the highest measured median (854.6 tok/s), 2.7% above the 2048/512 baseline,
with overlapping run ranges and 434 MiB more peak VRAM. Keep the default
2048/512 for now; these data do not justify spending more VRAM for a small,
uncertain gain. The production launcher defaults have not been changed.

One matched MTP-disabled run reached 889.6 tok/s, 4.1% above the MTP-on median
at 4096/1024, and reduced sampled peak VRAM by 1,680 MiB (1.64 GiB). This is
preliminary evidence of a modest prefill difference in this build, not a
repeatable speed claim or a recommendation to disable MTP for decoding.

| Batch/microbatch | Speculation | Completed runs | Median prompt tok/s | Range tok/s | Peak VRAM MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 2048/512 | MTP3 | 3 | 832.3 | 802.7–854.9 | 19,273.0 |
| 4096/1024 | MTP3 | 3 | 854.6 | 832.9–861.9 | 19,707.0 |
| 4096/2048 | MTP3 | 2 | 847.2 | 846.0–848.5 | 20,577.0 |
| 8192/2048 | MTP3 | 2 | 833.5 | 808.1–858.9 | 20,577.0 |
| 4096/1024 | Off | 1 | 889.6 | 889.6–889.6 | 18,027.0 |

All 11 completed requests were stable: full 50,000-token evaluation, zero
cached tokens, no truncation, and healthy server afterward. No completed run
reported a CUDA allocation or host-memory-limit failure. One additional
4096/2048 trial was interrupted at the user's request and is excluded. The
batch sweep was stopped before finishing its third round, followed by one
MTP-off request; no further permutations or repetitions were performed.

## Method and fixed settings

- RTX 5090 Laptop GPU, 24,463 MiB; Windows driver 592.27 exposed to WSL2.
- Official llama.cpp b11393 CUDA 12.8 runtime, commit dbe4c3ed4.
- Server binary SHA-256:
  `bdffebb4edaf60522f82eca49f58ca142901ea80f938ef8455a7ddf95dc8b9fa`.
- Swift IQ3_S MTP GGUF; model SHA-256
  `9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5`.
- 131072 context, q8_0/q8_0 main KV, default F16 draft KV when enabled,
  full GPU offload, Flash Attention, one slot, Direct I/O loading.
- Production scope: MemoryHigh 6 GiB, MemoryMax 8 GiB, no scope swap;
  two context checkpoints, 512 MiB RAM prompt cache.
- Every request started in a freshly loaded server, with `cache_prompt:false`,
  greedy sampling, seed 42, and one generated token. Tokenization did not
  submit an inference request. No inference warmup was performed.
- Identical frozen token IDs across both runs. Prompt SHA-256:
  `a177408aee56cfb4a402217151989914cde293d1df89e4fe4118862a781d9bfd`.
- Prompt is varied synthetic Python service modules, truncated to exactly
  50,000 token IDs. This measures prefill throughput, not a production trace,
  coding quality, MTP acceptance, or decode performance.
- Server timings supply prompt tok/s. Whole-GPU memory, temperature and power
  were sampled every approximately 200 ms plus nvidia-smi query overhead.
  Peak VRAM is a sampled whole-GPU maximum, not allocator telemetry.

## Limits and interpretation

Completed prefill times were approximately 56.2–62.3 seconds. Larger batches
and microbatches increased allocation without a substantial demonstrated
speedup. Temperature reached 68°C; runs were not thermally normalized, and
later results were often slower. Power/temperature traces are retained, but
these measurements cannot establish whether thermal or power limits caused
that variation. Repeat counts are unequal because testing was stopped on
request. No statistical significance claim is made.

MTP-off used the same GGUF with `--spec-type none`. Its lower allocation is
part of that configuration change; the test does not isolate a particular
kernel or establish why prompt processing differed. The model was not
replaced. One output token intentionally provides no useful decode benchmark.

The target range of 750–900 tok/s was reached by all completed configurations.
There was no native Windows testing or configuration work. All benchmark
servers were stopped afterward; no server was running before this session.

## Reproduction and evidence

Harness: [prefill_benchmark.py](prefill_benchmark.py). Launcher overrides:
`SWIFT_BATCH`, `SWIFT_UBATCH`, `SWIFT_SPEC_TYPE`; defaults remain unchanged.

```bash
# Full sweep; stop any existing server first.
python3 prefill_benchmark.py

# Only a single matched MTP-off check using the frozen prompt.
python3 prefill_benchmark.py --mtp-only --repeats 1 --batch 4096 --ubatch 1024 \
  --prompt-tokens logs/prefill-20261005-134711/prompt-tokens.json
```

- Batch evidence: `logs/prefill-20261005-134711/`.
- MTP-off evidence: `logs/prefill-20261005-135846/`.
- Combined summary: `logs/prefill-20261005-134711/summary.json`.
- Each folder preserves metadata, token array, per-trial server logs, GPU CSVs,
  raw timing responses and results. These local logs are ignored by Git.

Validation: Python compilation, Bash syntax and Git whitespace checks passed;
the live requests validated full prompt evaluation and cold cache behavior.
