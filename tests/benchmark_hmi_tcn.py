"""Bounded GPU benchmark of original vs equivalent TCN; no simulator/training run.

Run from the repository root in srlgym:
    python tests/benchmark_hmi_tcn.py --output /tmp/hmi_tcn_benchmark.json
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_srl_hmi_tcn import as_conv_reference  # imports isaacgym before torch
import torch
from isaacgymenvs.learning.SRLEvo.hmi_encoder.config import EncoderConfig
from isaacgymenvs.learning.SRLEvo.hmi_encoder.model import HMIEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    fast = HMIEncoder(EncoderConfig(enabled=True)).cuda()
    old = as_conv_reference(copy.deepcopy(fast))
    report = dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
                  cudnn_available=torch.backends.cudnn.is_available(),
                  cudnn_version=torch.backends.cudnn.version(), dtype='float32',
                  tf32=False, repeats=args.repeats, measurements=[])
    # Validate full-network GPU outputs, input gradients and every parameter gradient.
    x = torch.randn(32, 64, 41, device='cuda')
    x[..., -1] = 1
    x[0, :23] = 0
    records = []
    for model in (old, fast):
        a = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        z, pred = model(a)
        (z.square().mean() + pred.square().mean()).backward()
        records.append((z.detach(), pred.detach(), a.grad.detach(),
                        {n: p.grad.detach().clone() for n, p in model.named_parameters()}))
    errors = {}
    for name, i in (('latent', 0), ('prediction', 1), ('input_gradient', 2)):
        torch.testing.assert_close(records[0][i], records[1][i], rtol=2e-4, atol=1e-5)
        errors[name + '_max_abs'] = (records[0][i] - records[1][i]).abs().max().item()
    errors['parameter_gradient_max_abs'] = 0.
    for name, value in records[0][3].items():
        other = records[1][3][name]
        torch.testing.assert_close(value, other, rtol=2e-4, atol=1e-5, msg=name)
        errors['parameter_gradient_max_abs'] = max(errors['parameter_gradient_max_abs'],
                                                   (value - other).abs().max().item())
    report['equivalence'] = errors
    print('GPU equivalence:', json.dumps(errors), flush=True)
    del records, x

    for batch, backward in ((8192, False), (2048, True)):
        data = torch.randn(batch, 64, 41, device='cuda')
        data[..., -1] = 1
        row = dict(batch=batch, history=64, features=41, backward=backward)

        def run(model):
            model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(backward):
                z, pred = model(data)
                if backward:
                    (z.square().mean() + pred.square().mean()).backward()

        for name, model in (('conv1d', old), ('linear', fast)):
            run(model)  # same-shape warmup, excluded from timings
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            timings = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                run(model)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)
            row[name] = dict(seconds=timings, median_seconds=statistics.median(timings),
                             peak_extra_mib=(torch.cuda.max_memory_allocated() - baseline) / 2**20)
            print(json.dumps(dict(batch=batch, backward=backward, implementation=name,
                                  **row[name])), flush=True)
        row['speedup'] = row['conv1d']['median_seconds'] / row['linear']['median_seconds']
        report['measurements'].append(row)
        del data
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('Report:', args.output, flush=True)


if __name__ == '__main__':
    main()
