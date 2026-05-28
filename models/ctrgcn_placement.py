"""
model/ctrgcn_placement.py

Placement study host. SUBCLASSES the original CTR-GCN Model so the baseline
forward path is never edited in place (protects R3 in the week-2 brief).

placement enum:
  'none'              -> no slot constructed at all -> bit-for-bit baseline
  'before_backbone'   -> x = x + slot(x) once, after data_bn, before l1
  'after_each_block'  -> x = x + slot_i(x) after each of l1..l10 (residual add)
  'parallel_to_tcn'   -> inside each block: y = relu(tcn1(g) + gate_i*slot_i(g) + res(x))
                         gate_i is a learned scalar init 0  -> starts == baseline
  'deep_only'         -> parallel_to_tcn, but only on blocks 5..10
  'replace_tcn_all'   -> y = relu(slot_i(g) + res(x))  (drops tcn1; see caveat)

NOTE on 'replace_tcn_all': CFOT outputs a *correction*, not a full feature map.
Replacing the multi-scale TCN with only the CFOT residual is the weakest-motivated
placement and is included only for completeness. Expect it to underperform; that
is itself a reportable result, not a bug.

Equivalence guarantees:
  * placement='none'            -> identical module tree to baseline.
  * placement='parallel_to_tcn' -> gate=0 at init, so first forward == baseline.
Both are checkable (see equivalence_check.py).
"""

import torch
import torch.nn as nn

from model.ctrgcn import Model as BaseModel, TCN_GCN_unit
from models.mixers.temporal_mixer_slot import TemporalMixerSlot


# channel width of each block in the base model (base_channel=64)
BLOCK_OUT_CH = [64, 64, 64, 64, 128, 128, 128, 256, 256, 256]


class PlacementModel(BaseModel):
    def __init__(self, *args,
                 placement='none',
                 mixer='cfot_adaptive',
                 mixer_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.placement = placement
        self.mixer = mixer
        mixer_kwargs = mixer_kwargs or {}

        self.blocks = [self.l1, self.l2, self.l3, self.l4, self.l5,
                       self.l6, self.l7, self.l8, self.l9, self.l10]

        if placement == 'none':
            self.slots = None
            return

        def make_slot(ch):
            return TemporalMixerSlot(in_channels=ch, mixer=mixer, **mixer_kwargs)

        if placement == 'before_backbone':
            in_ch = kwargs.get('in_channels', 3)
            self.slots = nn.ModuleList([make_slot(in_ch)])

        elif placement in ('after_each_block',):
            self.slots = nn.ModuleList([make_slot(c) for c in BLOCK_OUT_CH])

        elif placement in ('parallel_to_tcn', 'deep_only', 'replace_tcn_all'):
            idxs = range(10) if placement != 'deep_only' else range(4, 10)
            self.slots = nn.ModuleDict(
                {str(i): make_slot(BLOCK_OUT_CH[i]) for i in idxs}
            )
            # learned per-block gate (scalar), init 0 -> starts as baseline
            if placement in ('parallel_to_tcn', 'deep_only'):
                self.gates = nn.ParameterDict(
                    {str(i): nn.Parameter(torch.zeros(1)) for i in idxs}
                )
        else:
            raise ValueError(f"unknown placement {placement}")

    # ---- helper: run one block with optional parallel/replace slot ----
    def _run_block(self, i, block, x):
        # NOTE on stride: CTR-GCN blocks l5 (idx 4) and l8 (idx 7) use stride=2,
        # so tcn1 HALVES T while gcn1 preserves it. The slot returns the same T
        # it receives. To keep shapes aligned with the (strided) tcn output, the
        # slot operates on `tcn` (already at the correct output T), not on `g`.
        if self.placement in ('parallel_to_tcn', 'deep_only') and str(i) in self.slots:
            g = block.gcn1(x)
            tcn = block.tcn1(g)                       # may be stride-2 -> T/2
            res = block.residual(x)
            slot = self.slots[str(i)](tcn)            # operate on tcn -> T matches
            gate = self.gates[str(i)]
            return block.relu(tcn + gate * slot + res)
        elif self.placement == 'replace_tcn_all':
            # Replace tcn with a stride-aware path. CFOT alone cannot downsample,
            # so for strided blocks we still need tcn1 for the stride; replacing
            # it is only well-defined for stride-1 blocks. We run gcn1 then slot,
            # and for stride!=1 blocks fall back to tcn1 to handle downsampling.
            g = block.gcn1(x)
            res = block.residual(x)
            # detect stride via residual type is unreliable; use tcn1 output T:
            tcn = block.tcn1(g)
            if tcn.size(2) != g.size(2):
                # strided block: CFOT can't downsample -> keep tcn, add slot on it
                slot = self.slots[str(i)](tcn)
                return block.relu(tcn + slot + res)
            else:
                slot = self.slots[str(i)](g)
                return block.relu(slot + res)         # true tcn replacement
        else:
            return block(x)                            # untouched baseline block

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        if self.placement == 'before_backbone':
            x = x + self.slots[0](x)

        for i, block in enumerate(self.blocks):
            x = self._run_block(i, block, x)
            if self.placement == 'after_each_block':
                x = x + self.slots[i](x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        return self.fc(x)

    def collect_reg_loss(self):
        if self.slots is None:
            return torch.tensor(0.0)
        slots = self.slots.values() if isinstance(self.slots, (nn.ModuleDict,)) else self.slots
        total = torch.tensor(0.0)
        for s in slots:
            total = total + s.reg_loss()
        return total