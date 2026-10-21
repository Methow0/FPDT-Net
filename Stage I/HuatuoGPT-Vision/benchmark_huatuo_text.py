#!/usr/bin/env python3
"""Run beside HuatuoGPT-Vision/cli.py in its existing Python environment.

Timing includes image IO, preprocessing, full autoregressive generation and
decoding. Loading/warmup are reported separately. No timing outliers are removed.
Optional profiler FLOPs count only supported operators, NOT total model FLOPs.
"""
import argparse
import csv
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def caption_text(output):
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise ValueError('Expected exactly one generated description')
        output = output[0]
    if not isinstance(output, str) or not output.strip():
        raise ValueError('Empty or unexpected model output')
    return output.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-folder', default='/home/rxk/YJ/FullNet/data/GlaS/images/train')
    parser.add_argument('--model', default='FreedomIntelligence/HuatuoGPT-Vision-7B')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--query', default='What does the picture show?')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-new-tokens', type=int, default=None,
                        help='Omit to preserve your cli.py generation settings')
    parser.add_argument('--profile-flops-images', type=int, default=0,
                        help='Separate diagnostic pass on first N successful images; partial FLOPs only')
    args = parser.parse_args()
    if args.warmup < 0 or args.profile_flops_images < 0:
        parser.error('Counts must be nonnegative')
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    root = Path(args.root_folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        raise RuntimeError(f'No images found in {root}')
    out = Path(args.output_dir or ('huatuo_stats_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(parents=True, exist_ok=True)
    for filename in ['captions.csv', 'per_image_stats.csv', 'summary.json', 'summary.csv', 'partial_flops.json']:
        if (out / filename).exists():
            raise FileExistsError(f'{out / filename}: choose a new --output-dir')

    # Official cli.py silently skips unreadable images; detect them explicitly.
    valid, invalid = [], []
    for path in paths:
        try:
            with Image.open(path) as im:
                im.convert('RGB').load()
            valid.append(path)
        except Exception as exc:
            invalid.append({'filename': str(path), 'status': 'invalid_image', 'error': str(exc)})
    if not valid:
        raise RuntimeError('No readable images')

    from cli import HuatuoChatbot
    seed_all(args.seed)
    sync()
    t0 = time.perf_counter()
    bot = HuatuoChatbot(args.model)
    model = bot.model
    model.eval()
    sync()
    load_s = time.perf_counter() - t0
    # parameters() deduplicates shared Parameter objects; includes registered vision tower.
    parameters = sum(p.numel() for p in model.parameters())
    if args.max_new_tokens is not None:
        bot.gen_kwargs['max_new_tokens'] = args.max_new_tokens
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise RuntimeError('This benchmark expects the official single-CUDA-device chatbot')
    torch.cuda.set_device(device)

    def generate(path):
        with torch.inference_mode():
            return caption_text(bot.inference(args.query, [str(path)]))

    print(f'Images: {len(paths)}; readable: {len(valid)}; invalid: {len(invalid)}')
    print(f'Parameters (M): {parameters / 1e6:.3f}')
    print(f'Generation settings: {bot.gen_kwargs}')
    sync()
    t0 = time.perf_counter()
    for _ in range(args.warmup):
        generate(valid[0])
    sync()
    warmup_s = time.perf_counter() - t0
    seed_all(args.seed)  # Warmup must not consume the measured run's RNG sequence.
    torch.cuda.reset_peak_memory_stats(device)
    rows, times, token_counts, successful_paths = [], [], [], []
    sync()
    batch_start = time.perf_counter()
    with (out / 'captions.csv').open('w', newline='', encoding='utf-8') as cf, \
         (out / 'per_image_stats.csv').open('w', newline='', encoding='utf-8') as sf:
        captions = csv.writer(cf)
        captions.writerow(['Caption', 'filename'])
        stats = csv.DictWriter(sf, fieldnames=['filename', 'status', 'processing_time_s',
                                             'caption_retokenized_tokens', 'error'])
        stats.writeheader()
        for row in invalid:
            stats.writerow(row)
        for index, path in enumerate(valid, 1):
            sync()
            start = time.perf_counter()
            try:
                caption = generate(path)
                sync()
                elapsed = time.perf_counter() - start
                # Re-tokenized decoded text, not exact generation steps (EOS excluded).
                count = len(bot.tokenizer.encode(caption, add_special_tokens=False))
                captions.writerow([caption, str(path)])
                times.append(elapsed)
                token_counts.append(count)
                successful_paths.append(path)
                row = dict(filename=str(path), status='ok', processing_time_s=elapsed,
                           caption_retokenized_tokens=count, error='')
                print(f'[{index}/{len(valid)}] {path.name}: {elapsed:.3f} s')
            except Exception as exc:
                sync()
                row = dict(filename=str(path), status='error',
                           processing_time_s=time.perf_counter() - start,
                           caption_retokenized_tokens='', error=str(exc))
                print(f'ERROR {path.name}: {exc}')
            rows.append(row)
            stats.writerow(row)
            cf.flush()
            sf.flush()
    sync()
    batch_s = time.perf_counter() - batch_start
    summary = {
        'component': 'Text generation', 'model': args.model,
        'parameters_M': parameters / 1e6,
        'processing_time_mean_s_per_image': statistics.mean(times) if times else None,
        'processing_time_std_s': statistics.stdev(times) if len(times) > 1 else (0.0 if times else None),
        'processing_time_median_s': statistics.median(times) if times else None,
        'successful_inference_sum_min': sum(times) / 60,
        'total_processing_time_min': batch_s / 60,
        'model_load_time_s': load_s, 'warmup_time_s': warmup_s,
        'load_warmup_processing_sum_min': (load_s + warmup_s + batch_s) / 60,
        'peak_allocated_GPU_memory_GB': torch.cuda.max_memory_allocated(device) / 1e9,
        'peak_reserved_GPU_memory_GB': torch.cuda.max_memory_reserved(device) / 1e9,
        'images_found': len(paths), 'images_succeeded': len(times),
        'images_failed': len(paths) - len(times),
        'mean_caption_retokenized_tokens': statistics.mean(token_counts) if token_counts else None,
        'GPU': torch.cuda.get_device_name(device), 'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda, 'parameter_dtypes': sorted({str(p.dtype) for p in model.parameters()}),
        'image_processor_crop_size': getattr(bot.processor, 'crop_size', None),
        'generation_settings': dict(bot.gen_kwargs), 'batch_size': 1,
        'warmup_calls': args.warmup, 'seed': args.seed, 'query': args.query,
        'FLOPs_G': None,
        'timing_scope': 'Single-image inference() including image IO/preprocessing/generation/decoding. '
                        'Mean over all successes, no outlier filtering. Total processing includes failed '
                        'attempts, CSV IO, re-tokenization and logging; excludes loading, warmup, validation '
                        'scan and optional profiler pass.',
        'memory_scope': 'PyTorch CUDA allocator peak after warmup; includes resident model. GB=10^9 bytes. '
                        'Excludes non-PyTorch allocations; reserved is allocator cache, not tensor usage.'
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f'Results saved to: {out.resolve()}')

    if args.profile_flops_images:
        print('Separate FLOPs diagnostic: supported operators only; NOT full-model FLOPs.')
        from torch.profiler import ProfilerActivity, profile
        estimates = []
        for index, path in enumerate(successful_paths[:args.profile_flops_images]):
            seed_all(args.seed + index)
            try:
                # CPU dispatcher events contain supported CUDA-operation FLOP formulas.
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
                    caption = generate(path)
                    sync()
                value = sum(event.flops or 0 for event in prof.key_averages()) / 1e9
                estimates.append({'filename': str(path), 'partial_supported_ops_GFLOPs': value,
                                  'caption_retokenized_tokens': len(bot.tokenizer.encode(caption, add_special_tokens=False))})
            except Exception as exc:
                estimates.append({'filename': str(path), 'error': str(exc)})
        values = [r['partial_supported_ops_GFLOPs'] for r in estimates if 'partial_supported_ops_GFLOPs' in r]
        result = {'mean_partial_supported_ops_GFLOPs': statistics.mean(values) if values else None,
                  'warning': 'Partial operator estimate, NOT complete generation FLOPs. Fused attention and '
                             'other unsupported operations can be omitted. Separate stochastic generations '
                             'may have different lengths. Do not place this number in a full FLOPs column.',
                  'samples': estimates}
        (out / 'partial_flops.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if not times:
        raise RuntimeError('All generations failed; see per_image_stats.csv')


if __name__ == '__main__':
    main()
