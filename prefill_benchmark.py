#!/usr/bin/env python3
"""Cold 50K prefill sweep, using the production WSL launcher and stdlib only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:1235'
MATRIX = [(2048, 512), (4096, 1024), (4096, 2048), (8192, 2048)]


def post(route, payload):
    with urlopen(Request(BASE + route, json.dumps(payload).encode(),
                         headers={'Content-Type': 'application/json'}), timeout=900) as r:
        return json.load(r)


def stop():
    subprocess.run(['systemctl', '--user', 'stop', 'swift15.scope'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def gpu():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free,temperature.gpu,power.draw',
                                   '--format=csv,noheader,nounits'], text=True)
    return [float(x.strip()) for x in raw.splitlines()[0].split(',')]


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mtp-only', action='store_true', help='only run MTP-disabled trials')
    parser.add_argument('--batch', type=int, default=4096, help='batch for --mtp-only')
    parser.add_argument('--ubatch', type=int, default=1024, help='microbatch for --mtp-only')
    parser.add_argument('--prompt-tokens', type=Path, help='reuse a saved 50K token array')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    # Refuse to interrupt an unrelated existing listener.
    try:
        urlopen(BASE + '/health', timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError('Port 1235 already has a server; stop it before benchmarking')
    out = args.output or ROOT / 'logs' / time.strftime('prefill-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'workload': 'synthetic varied Python service modules; not a production trace',
                'prompt_tokens': 50000, 'repeats': args.repeats,
                'launcher': (ROOT / 'start_swift15_iq3s_128k.sh').read_text(),
                'environment': subprocess.check_output(['nvidia-smi'], text=True)}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    results, tokens = [], None
    if args.prompt_tokens:
        tokens = json.loads(args.prompt_tokens.read_text())
        if len(tokens) != 50000 or not all(isinstance(t, int) for t in tokens):
            raise ValueError('saved prompt must contain exactly 50000 integer token IDs')
        data = json.dumps(tokens).encode()
        (out / 'prompt-tokens.json').write_bytes(data)
        metadata['prompt_sha256'] = hashlib.sha256(data).hexdigest()
        metadata['prompt_source'] = str(args.prompt_tokens.resolve())
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    def save():
        (out / 'results.json').write_text(json.dumps(results, indent=2))

    def run(batch, ubatch, mtp, repeat):
        nonlocal tokens
        label = f'b{batch}-ub{ubatch}-{"mtp" if mtp else "none"}-{repeat}'
        record = {'label': label, 'batch': batch, 'ubatch': ubatch, 'mtp': mtp,
                  'repeat': repeat, 'stable': False}
        results.append(record)
        env = dict(os.environ, SWIFT_BATCH=str(batch), SWIFT_UBATCH=str(ubatch),
                   SWIFT_SPEC_TYPE='draft-mtp' if mtp else 'none', SWIFT_CONTEXT='131072')
        done = threading.Event()
        samples = []
        def monitor():
            with (out / f'{label}-gpu.csv').open('w') as f:
                f.write('monotonic,used_mib,free_mib,temp_c,power_w\n')
                while not done.is_set():
                    try:
                        values = gpu()
                        samples.append(values)
                        f.write(','.join(map(str, [time.monotonic()] + values)) + '\n')
                        f.flush()
                    except Exception as exc:
                        record['monitor_error'] = str(exc)
                    done.wait(0.2)
        process = None
        thread = None
        try:
            with (out / f'{label}-server.log').open('w') as log:
                process = subprocess.Popen([str(ROOT / 'start_swift15_iq3s_128k.sh')],
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited during startup: {process.returncode}')
                try:
                    with urlopen(BASE + '/health', timeout=2) as r:
                        if json.load(r).get('status') == 'ok':
                            break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server startup timed out')
                time.sleep(1)
            if tokens is None:
                text = 'Review the following Python service modules for correctness and propose fixes.\n'
                for i in range(2200):
                    text += (f'\n# services/worker_{i}.py\n'
                             f'def process_{i}(records, limit={i % 97 + 1}):\n'
                             f'    """Validate shard {i} and return accepted records."""\n'
                             f'    accepted = []\n    for index, record in enumerate(records):\n'
                             f'        if index >= limit:\n            break\n'
                             f'        value = record.get("value", {i % 31})\n'
                             f'        if isinstance(value, int) and value % {i % 11 + 2} == 0:\n'
                             f'            accepted.append(("shard_{i}", value + {i}))\n'
                             f'    return accepted\n')
                all_tokens = post('/tokenize', {'content': text, 'add_special': True})['tokens']
                assert len(all_tokens) >= 50000
                tokens = all_tokens[:50000]
                data = json.dumps(tokens).encode()
                (out / 'prompt-tokens.json').write_bytes(data)
                (out / 'prompt-source.txt').write_text(text)
                metadata['prompt_sha256'] = hashlib.sha256(data).hexdigest()
                (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
            record['gpu_before'] = gpu()
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            start = time.monotonic()
            request = Request(BASE + '/completion', json.dumps({
                'prompt': tokens, 'n_predict': 1, 'temperature': 0, 'seed': 42,
                'cache_prompt': False, 'ignore_eos': True, 'stream': True}).encode(),
                headers={'Content-Type': 'application/json'})
            final = None
            with urlopen(request, timeout=900) as response:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        continue
                    event = json.loads(raw)
                    if event.get('content') and 'ttft_seconds' not in record:
                        record['ttft_seconds'] = time.monotonic() - start
                    if event.get('stop'):
                        final = event
            record['wall_seconds'] = time.monotonic() - start
            if final is None:
                raise RuntimeError('missing final timing event')
            (out / f'{label}-response.json').write_text(json.dumps(final, indent=2))
            record['timings'] = final['timings']
            assert final['timings']['prompt_n'] == 50000, 'full prompt not evaluated'
            assert final['timings'].get('cache_n', 0) == 0, 'cached tokens detected'
            assert not final.get('truncated', False), 'prompt truncated'
            with urlopen(BASE + '/health', timeout=5) as r:
                assert json.load(r)['status'] == 'ok'
            record['stable'] = True
        except KeyboardInterrupt:
            record['interrupted'] = True
            raise
        except Exception as exc:
            record['error'] = repr(exc)
        finally:
            done.set()
            if thread:
                thread.join(timeout=10)
            if samples:
                record['sampled_peak_vram_mib'] = max(v[0] for v in samples)
                record['sampled_min_free_mib'] = min(v[1] for v in samples)
                record['max_temperature_c'] = max(v[2] for v in samples)
            record['scope_memory'] = subprocess.run(
                ['systemctl', '--user', 'show', 'swift15.scope', '-p', 'MemoryPeak',
                 '-p', 'MemoryCurrent', '-p', 'Result'], capture_output=True, text=True).stdout
            stop()
            if process:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            save()
            print(json.dumps(record), flush=True)
            time.sleep(3)
        return record['stable']

    try:
        viable = [] if args.mtp_only else [pair for pair in MATRIX if run(*pair, True, 1)]
        for repeat in range(2, args.repeats + 1):
            for pair in viable[:]:
                if not run(*pair, True, repeat):
                    viable.remove(pair)
        if not viable and not args.mtp_only:
            raise RuntimeError('no stable batch configuration')
        def speed(pair):
            return statistics.median(r['timings']['prompt_per_second'] for r in results
                                     if r['stable'] and (r['batch'], r['ubatch']) == pair and r['mtp'])
        winner = (args.batch, args.ubatch) if args.mtp_only else max(viable, key=speed)
        for repeat in range(1, args.repeats + 1):
            if not run(*winner, False, repeat):
                break
        summary = []
        for batch, ubatch, mtp in dict.fromkeys((r['batch'], r['ubatch'], r['mtp']) for r in results):
            group = [r for r in results if (r['batch'], r['ubatch'], r['mtp']) == (batch, ubatch, mtp)]
            valid = [r for r in group if r['stable']]
            rates = [r['timings']['prompt_per_second'] for r in valid]
            summary.append({'batch': batch, 'ubatch': ubatch, 'mtp': mtp, 'valid_runs': len(valid),
                            'failed_runs': len(group) - len(valid),
                            'median_tps': statistics.median(rates) if rates else None,
                            'range_tps': [min(rates), max(rates)] if rates else None,
                            'peak_vram_mib': max((r.get('sampled_peak_vram_mib', 0) for r in group), default=0)})
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('SUMMARY', json.dumps(summary), flush=True)
    finally:
        stop()
    print('Artifacts:', out, flush=True)


if __name__ == '__main__':
    main()
