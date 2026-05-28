"""
models/mixers/temporal_mixer_slot.py

Adapter around the EXISTING CFOT modules in cfot.py for the placement study.

Design fact that drives everything: CFOT returns a RESIDUAL R of shape
[B,C,T,V] meant to be ADDED to x (it builds R=zeros_like(x) and fills R[:,:,d:]).
So this slot is additive by nature. The host model decides WHERE to add it
(placement); the slot decides WHAT the residual is (mixer type).

mixer types:
  'cfot_multi'    -> MultiDeltaCFOT          (parallel fixed deltas, conv-fused)
  'cfot_adaptive' -> AdaptiveDeltaCFOTv2     (learned temporal gate over deltas)
  'cfot_signed'   -> SignedMultiDeltaCFOT    (bidirectional)
  'softmax'       -> a CFOT branch with transport='softmax' (attention control)
  'identity'      -> returns zeros (no residual) -> host adds 0 -> exact baseline

The 'identity' branch returning a zero residual is what makes placement=none /
gate=0 reduce to baseline cleanly.
"""

import torch
import torch.nn as nn

from model.cfot import MultiDeltaCFOT, AdaptiveDeltaCFOTv2, SignedMultiDeltaCFOT


class TemporalMixerSlot(nn.Module):
    def __init__(self, in_channels, mixer='cfot_adaptive',
                 deltas=(1, 2, 4), emb_channels=64,
                 sinkhorn_iters=5, sinkhorn_tau=0.5, topk=4,
                 store_links=False, **cfot_kwargs):
        super().__init__()
        self.mixer = mixer
        self.in_channels = in_channels

        common = dict(in_channels=in_channels, emb_channels=emb_channels,
                      deltas=deltas, sinkhorn_iters=sinkhorn_iters,
                      sinkhorn_tau=sinkhorn_tau, topk=topk,
                      store_links=store_links)

        if mixer == 'identity':
            self.core = None  # returns zeros; host add is a no-op
        elif mixer == 'cfot_multi':
            self.core = MultiDeltaCFOT(**common, **cfot_kwargs)
        elif mixer == 'cfot_adaptive':
            self.core = AdaptiveDeltaCFOTv2(**common, **cfot_kwargs)
        elif mixer == 'cfot_signed':
            self.core = SignedMultiDeltaCFOT(**common, **cfot_kwargs)
        elif mixer == 'softmax':
            # attention control: same CFOT machinery, row-stochastic transport
            self.core = MultiDeltaCFOT(**common, transport='softmax', **cfot_kwargs)
        else:
            raise ValueError(f"unknown mixer {mixer}")

    def forward(self, x):
        # returns a RESIDUAL (same shape as x). Host decides how to combine.
        if self.core is None:
            return torch.zeros_like(x)
        return self.core(x)

    def reg_loss(self):
        if self.core is not None and hasattr(self.core, 'regularization_loss'):
            return self.core.regularization_loss()
        return torch.tensor(0.0)