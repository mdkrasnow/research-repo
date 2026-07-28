"""GPU dense scalar-versus-line validation on a subset of the frozen bank only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from energy_monotonicity.evaluate_energy_monotonicity import (
    CheckpointRecord, dense_validation,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--monotonicity-output-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--subset', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    source, output = args.monotonicity_output_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = [CheckpointRecord(**x) for x in json.loads((source/'checkpoint_manifest.json').read_text())]
    bank = torch.load(source/'evaluation_bank.pt', map_location='cpu', weights_only=True)
    # Preserve the original frozen examples/noise tensors; never create a new bank.
    device = torch.device(args.device)
    report = {'source': str(source), 'frozen_bank_sha256': json.loads((source/'evaluation_bank.json').read_text())['bank_sha256'],
              'subset_images': args.subset, 'trajectories_per_variant': args.subset, 'grids': [21, 101], 'results': {}}
    for variant in ('dot', 'direct'):
        record = next(r for r in manifest if r.variant == variant and r.epoch == 8)
        report['results'][variant] = dense_validation(record, bank, output, args.batch_size, device, torch.float32, args.subset, False)
    (output/'numerical_validation_dense.json').write_text(json.dumps(report, indent=2) + '\n')
    if not all(r['convergence_pass'] for r in report['results'].values()):
        raise RuntimeError('dense scalar-line convergence failed')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
