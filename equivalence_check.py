"""
equivalence_check.py  --  run BEFORE trusting any placement result (R3 guard).

Checks:
  1. placement='none' produces the SAME output as the original baseline Model
     for identical weights and input (bit-for-bit up to fp noise).
  2. placement='parallel_to_tcn' with gate=0 (init) ALSO equals baseline on the
     first forward -> proves the parallel wiring is a true additive zero at init.
  3. All five placements run a forward+backward without shape errors and produce
     finite gradients (catches the V x V transport wiring bugs early).

Run on CPU with a tiny tensor; this is a wiring test, not a speed test.
Use NTU layout (num_point=25) or SHREC (num_point=22, num_person=1) as needed.
"""

import copy
import torch

from model.ctrgcn import Model as BaseModel
from model.ctrgcn_placement import PlacementModel

GRAPH = 'graph.ntu_rgb_d.Graph'   # swap to your SHREC graph if testing SHREC
GA = dict(labeling_mode='spatial')
COMMON = dict(num_class=60, num_point=25, num_person=2,
              graph=GRAPH, graph_args=GA)


def _toy(num_point=25, num_person=2):
    return torch.randn(2, 3, 16, num_point, num_person)  # small T for speed


def check_none_equals_baseline():
    torch.manual_seed(0)
    base = BaseModel(**COMMON).eval()
    plc = PlacementModel(**COMMON, placement='none').eval()
    plc.load_state_dict(base.state_dict(), strict=True)  # MUST match exactly
    x = _toy()
    with torch.no_grad():
        a, b = base(x), plc(x)
    diff = (a - b).abs().max().item()
    print(f"[none==baseline] max|diff| = {diff:.3e}  -> {'PASS' if diff < 1e-5 else 'FAIL'}")
    return diff < 1e-5


def check_parallel_gate0_equals_baseline():
    torch.manual_seed(0)
    base = BaseModel(**COMMON).eval()
    plc = PlacementModel(**COMMON, placement='parallel_to_tcn',
                         mixer='cfot_adaptive').eval()
    # copy the shared baseline weights; slot/gate params are extra
    missing, unexpected = plc.load_state_dict(base.state_dict(), strict=False)
    x = _toy()
    with torch.no_grad():
        a, b = base(x), plc(x)
    diff = (a - b).abs().max().item()
    # gates init at 0 -> slot contributes nothing on first forward
    print(f"[parallel gate=0] max|diff| = {diff:.3e}  -> {'PASS' if diff < 1e-5 else 'FAIL'}")
    print(f"    (extra slot params not in baseline: {len(missing)} tensors)")
    return diff < 1e-5


def check_all_placements_run():
    for p in ['before_backbone', 'after_each_block', 'parallel_to_tcn',
              'deep_only', 'replace_tcn_all']:
        torch.manual_seed(0)
        m = PlacementModel(**COMMON, placement=p, mixer='cfot_adaptive').train()
        x = _toy()
        y = m(x)
        loss = y.sum()
        loss.backward()
        gnorm = sum(g.grad.abs().sum().item()
                    for g in m.parameters() if g.grad is not None)
        ok = torch.isfinite(y).all().item() and (gnorm == gnorm)  # not NaN
        print(f"[{p:18s}] out={tuple(y.shape)} grad_norm={gnorm:.3e} "
              f"-> {'PASS' if ok else 'FAIL'}")


if __name__ == '__main__':
    print("=== 1. none == baseline ===")
    check_none_equals_baseline()
    print("\n=== 2. parallel gate=0 == baseline ===")
    check_parallel_gate0_equals_baseline()
    print("\n=== 3. all placements forward+backward ===")
    check_all_placements_run()