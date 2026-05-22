\# Day 1 — Thursday May 21



Audit complete: 6 existing CFOT IPN configs reviewed, deviations from published recipes

catalogued in `docs/recipe\_audit.md`. Biggest finding: the existing MS-G3D-on-IPN config

was using the 4-GPU recipe (lr=0.1, batch=64) on a single A5000 — this is the recipe

deviation the MS-G3D README explicitly warns against. Corrected to lr=0.05, batch=32 in

`configs/published/msg3d\_ntu60\_xsub.yaml` with a comment block citing the README.



Published configs copied for ST-GCN, CTR-GCN (NTU + UCLA), MS-G3D. InfoGCN YAML

constructed manually from `\~/external/infogcn/main.py` argparse defaults — λ₁ = 1.0e-4,

λ₂ = 1.0e-1, otherwise SGD+Nesterov 0.9 / lr 0.1 / cosine 110ep / wd 5e-4 / batch 64 /

warmup 5ep as expected.



NTU-60 X-Sub data verified: 4 files, total <X> GB, md5s recorded in `docs/data\_inventory.md`.

One-batch sanity load passed through all four backbone adapters. ST-GCN and InfoGCN adapters

stubbed from CTR-GCN's adapter as templates — needed only minor permutation adjustments.



ST-GCN NTU-60 X-Sub joint seed 0 launched at <time>, work\_dir

`logs/week1/stgcn\_ntu60\_xsub\_joint\_s0`. Expected finish Friday \~<time>. Will check on it

in the morning before launching the next baseline.



Gotchas: <whatever surprised you, or "none" if smooth>.

