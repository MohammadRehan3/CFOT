window_size: 180
normalziation:  subtracting a center joint (joint index 1: palm)
repeat: false
random_choose: false 
num_epoch : 65
warm_up_epoch: 5

SHREC-14 CTR-GCN repaired: 84–86% → 96.54% (seed 0), exceeding published 96.10%. Root causes, isolated: (1) training schedule ~6× too short — 65→150 epochs, step [35,55]→[70,100], repeat=5 added → +7–9 pp; (2) palm-centering zeroed joint 1, discarding ~4.5% of spatial signal — removed → +3.3 pp. Brief's recipe (T=150, palm-centered, 170ep) did not match verified DSTSA reference (T=180 interp, uncentered, 150ep); reference recipe was correct.

