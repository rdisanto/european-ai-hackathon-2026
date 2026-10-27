"""Finite, synchronized English SSE bursts; no API keys in output.

Measures client latency through the supplied URL and independently differences
server Prometheus queue histograms. This is a burst test, not steady-state load.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import math
from pathlib import Path
import re
import threading
import time
import urllib.request


LEVELS = [1, 4, 8, 16, 32, 50, 100]


def distribution(values):
    values = sorted(x for x in values if x is not None and math.isfinite(x))
    if not values:
        return {'n': 0}
    def percentile(q):
        index = (len(values) - 1) * q
        low = int(index)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)
    return {'n': len(values), 'mean': sum(values) / len(values),
            'p50': percentile(.50), 'p95': percentile(.95), 'max': values[-1]}


def metrics(text):
    result = {}
    for line in text.splitlines():
        match = re.match(r'^([a-zA-Z_:][a-zA-Z_0-9:]*)(\{.*\})?\s+([^\s]+)', line)
        if not match:
            continue
        name, labels, number = match.groups()
        if not name.startswith('vllm:'):
            continue
        try:
            value = float(number)
        except ValueError:
            continue
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', labels or ''))
        label = labels.get('le', '')
        result[(name, label)] = result.get((name, label), 0) + value
    return result


def queue_delta(before, after):
    result = {}
    for (name, label), count in after.items():
        if 'queue' not in name or not name.endswith('_seconds_count') or label:
            continue
        base = name[:-6]
        count -= before.get((name, ''), 0)
        total = after.get((base + '_sum', ''), 0) - before.get((base + '_sum', ''), 0)
        buckets = []
        for (metric, boundary), value in after.items():
            if metric == base + '_bucket':
                buckets.append((float(boundary), value - before.get((metric, boundary), 0)))
        buckets.sort()
        bound = next((edge for edge, value in buckets if value >= .95 * count), None) if count > 0 else None
        result[base] = {'count': count, 'sum_seconds': total,
                        'mean_seconds': total / count if count > 0 else None,
                        'p95_bucket_upper_seconds': bound if bound is not None and math.isfinite(bound) else None,
                        'p95_infinite_bucket': bound is not None and not math.isfinite(bound),
                        'buckets': [{'le': edge if math.isfinite(edge) else '+Inf', 'count': value} for edge, value in buckets]}
    return result


def request(base, key, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    return urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=timeout)


def stream_one(base, key, profile, prompt, cap, timeout, barrier=None, epoch=None):
    body = {'model': profile['served_model'], 'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': cap, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True},
            'chat_template_kwargs': profile['client']['chat_template_kwargs']}
    if barrier is not None:
        barrier.wait(timeout=60)
    start = time.monotonic()
    first = visible = None
    text, reasoning = [], []
    usage, finish, done = None, None, False
    obj = {'ok': False, 'dispatch_delay_seconds': start - epoch[0] if epoch else 0}
    try:
        with request(base, key, '/v1/chat/completions', body, timeout) as response:
            obj['http_status'] = response.status
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if 'error' in event:
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    content = delta.get('content') or ''
                    thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                    now = time.monotonic() - start
                    if (content or thought) and first is None:
                        first = now
                    if content and visible is None:
                        visible = now
                    text.append(content)
                    reasoning.append(thought)
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
        obj.update({'done': done, 'finish_reason': finish, 'usage': usage,
                    'ttft_seconds': first, 'first_visible_content_seconds': visible,
                    'content_sample': ''.join(text)[:800], 'reasoning_sample': ''.join(reasoning)[:200]})
        if not done or not finish or first is None or not usage or not isinstance(usage.get('completion_tokens'), int):
            raise RuntimeError('Incomplete SSE stream, finish reason, text, or token usage')
        obj['ok'] = True
    except Exception as error:
        obj['error'] = (type(error).__name__ + ': ' + str(error)).replace(key, '[REDACTED]')
    obj['e2e_seconds'] = time.monotonic() - start
    if obj['ok']:
        tokens = usage['completion_tokens']
        generation_time = obj['e2e_seconds'] - first
        obj['decode_tokens_per_second'] = (tokens - 1) / generation_time if tokens > 1 and generation_time > 0 else None
    return obj


def save(path, obj):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--api-key-file', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--levels', nargs='+', type=int, default=LEVELS)
    args = p.parse_args()
    profile = json.loads(args.profile.read_text())
    key = args.api_key_file.read_text().strip()
    base = args.base_url.rstrip('/')
    args.output.mkdir(parents=True, exist_ok=True)
    def scrape():
        with request(base, key, '/metrics') as response:
            return response.read().decode()
    # Different identifiers prevent completely identical prompts within a burst.
    def prompt(tag):
        return ('Workshop request ' + tag + '. Write only in English. '
                'Prepare a detailed practical checklist of at least 300 words for a participant '
                'using an HPC-hosted AI assistant. Cover API authentication, streaming, retries, '
                'privacy, and responsible use. Start the checklist immediately without an introduction.')
    warm = stream_one(base, key, profile, prompt('warmup'), args.max_tokens, args.timeout)
    save(args.output / 'warmup.json', warm)
    if not warm['ok']:
        raise RuntimeError('Warm-up failed; inspect warmup.json')
    time.sleep(6)
    result = {'profile': profile, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'design': 'Two synchronized finite bursts per concurrency; not a steady-state benchmark',
              'max_output_tokens': args.max_tokens, 'request_timeout_seconds': args.timeout,
              'repeats': args.repeats, 'levels': args.levels, 'prompt_example': prompt('example'), 'bursts': []}
    save(args.output / 'results.json', result)
    for level in args.levels:
        for repeat in range(1, args.repeats + 1):
            tag = '{}-{}'.format(level, repeat)
            before_text = scrape()
            (args.output / (tag + '-metrics-before.txt')).write_text(before_text)
            samples, stop = [], threading.Event()
            def monitor():
                while not stop.is_set():
                    try:
                        data = metrics(scrape())
                        samples.append({'seconds': time.monotonic() - epoch[0],
                                        'running': data.get(('vllm:num_requests_running', '')),
                                        'waiting': data.get(('vllm:num_requests_waiting', ''))})
                    except Exception as error:
                        samples.append({'error': str(error).replace(key, '[REDACTED]')})
                    stop.wait(1)
            barrier, epoch = threading.Barrier(level + 1), [0.0]
            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(stream_one, base, key, profile, prompt(tag + '-' + str(i)),
                                           args.max_tokens, args.timeout, barrier, epoch) for i in range(level)]
                epoch[0] = time.monotonic()
                watcher = threading.Thread(target=monitor, daemon=True)
                watcher.start()
                barrier.wait(timeout=60)
                responses = [future.result() for future in futures]
            elapsed = time.monotonic() - epoch[0]
            stop.set()
            watcher.join(timeout=35)
            # Wait for periodic engine metrics before taking histogram differences.
            time.sleep(6)
            after_text = scrape()
            (args.output / (tag + '-metrics-after.txt')).write_text(after_text)
            good = [r for r in responses if r['ok']]
            tokens = sum(r['usage']['completion_tokens'] for r in good)
            burst = {'concurrency': level, 'repeat': repeat, 'successful': len(good), 'failed': level - len(good),
                     'wall_seconds': elapsed, 'output_tokens': tokens,
                     'output_tokens_per_second': tokens / elapsed, 'requests_per_second': len(good) / elapsed,
                     'ttft_seconds': distribution([r.get('ttft_seconds') for r in good]),
                     'first_visible_content_seconds': distribution([r.get('first_visible_content_seconds') for r in good]),
                     'e2e_seconds': distribution([r['e2e_seconds'] for r in good]),
                     'decode_tokens_per_second': distribution([r.get('decode_tokens_per_second') for r in good]),
                     'dispatch_delay_seconds': distribution([r['dispatch_delay_seconds'] for r in responses]),
                     'queue_histogram_delta': queue_delta(metrics(before_text), metrics(after_text)),
                     'samples': samples, 'responses': responses}
            save(args.output / (tag + '.json'), burst)
            result['bursts'].append(burst)
            save(args.output / 'results.json', result)
            print('BURST', profile['id'], tag, 'success={}/{}'.format(len(good), level),
                  'p95_ttft={:.3f}'.format(burst['ttft_seconds'].get('p95', -1)),
                  'p95_e2e={:.3f}'.format(burst['e2e_seconds'].get('p95', -1)),
                  'output_tok_s={:.2f}'.format(tokens / elapsed), flush=True)
            after_metrics = metrics(after_text)
            if any(after_metrics.get(('vllm:num_requests_' + name, ''), 0) for name in ('waiting', 'running')):
                raise RuntimeError('Unexpected active requests after burst')
            with request(base, key, '/health') as health:
                if health.status != 200:
                    raise RuntimeError('Server unhealthy; stopping escalation')
    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(args.output / 'results.json', result)
    print('ALL_BURSTS_COMPLETE', profile['id'], flush=True)


if __name__ == '__main__':
    main()
