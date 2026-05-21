# \# Recipe Audit — Existing CFOT IPN Configs

# 

#### | config | optimizer | base\_lr | schedule | weight\_decay | momentum/nesterov | batch | epochs | warmup | label\_smoothing | augmentation | vs published |

#### |---|---|---|---|---|---|---|---|---|---|---|---|

#### | config/ipn\_ctrgcn\_joint.yaml | SGD | 0.1 | step \[60,80] | 5e-4 | 0.9 / true | 64 | 100 | none | none | rotation, scale | CTR-GCN paper uses cosine, not step; otherwise matches |

#### | config/ipn\_msg3d\_joint.yaml | SGD | 0.1 | step \[45,55] | 5e-4 | 0.9 / true | 64 | 65 | none | none | none | \*\*MS-G3D paper requires lr=0.05/batch=32 on 1 GPU — you used 4-GPU recipe on a single A5000. This is wrong.\*\* |

#### 

#### (rows for every IPN config you have)

