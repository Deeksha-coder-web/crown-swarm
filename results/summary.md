# Results summary

Train plans: office, hall, mine. Test plans (never seen by the optimiser): apartment, rubble.

## Main comparison — unseen test plans (apartment, rubble)

60 agents, 800 steps, 30 seeds per plan. Metrics from intact runs; 'coverage kept' pairs each run with the same seed where 20% of agents are removed at step 400. Mean [95% bootstrap CI]. Static coverage ceiling (greedy placement, line of sight): apartment 0.998, rubble 0.990.

| Controller | Coverage (2nd half) | Area explored | Spacing CV | Collisions / agent | Path / agent (m) | Coverage kept after 20% loss |
|---|---|---|---|---|---|---|
| Random walk | 0.610 [0.602, 0.617] | 0.895 [0.880, 0.909] | 0.615 [0.589, 0.640] | 8.606 [8.461, 8.760] | 65.864 [64.433, 67.242] | 0.887 [0.868, 0.907] |
| A: abrasion | 0.596 [0.588, 0.604] | 0.893 [0.876, 0.909] | 0.624 [0.595, 0.650] | 5.935 [5.847, 6.033] | 59.808 [58.556, 61.046] | 0.892 [0.872, 0.913] |
| B: light | 0.886 [0.867, 0.905] | 0.957 [0.947, 0.967] | 0.220 [0.200, 0.238] | 1.054 [1.025, 1.082] | 49.631 [48.394, 50.870] | 0.941 [0.933, 0.948] |
| C: hybrid | 0.892 [0.875, 0.908] | 0.950 [0.939, 0.961] | 0.182 [0.165, 0.199] | 1.233 [1.178, 1.290] | 54.010 [53.081, 54.974] | 0.943 [0.935, 0.950] |
| Potential field | 0.807 [0.790, 0.823] | 0.871 [0.857, 0.885] | 0.206 [0.196, 0.217] | 1.165 [1.132, 1.198] | 34.135 [33.876, 34.380] | 0.935 [0.925, 0.944] |
| Boids | 0.790 [0.779, 0.802] | 0.856 [0.849, 0.863] | 0.190 [0.175, 0.205] | 1.344 [1.301, 1.388] | 93.463 [90.620, 96.071] | 0.922 [0.914, 0.931] |
| Lloyd (privileged) | 0.700 [0.655, 0.744] | 0.746 [0.699, 0.791] | 0.423 [0.395, 0.451] | 2.007 [1.934, 2.090] | 38.992 [38.407, 39.616] | 0.980 [0.969, 0.990] |

### Generalisation (coverage, 2nd half): train plans vs unseen test plans

| Controller | Train | Test |
|---|---|---|
| Random walk | 0.749 [0.715, 0.783] | 0.610 [0.602, 0.617] |
| A: abrasion | 0.734 [0.698, 0.768] | 0.596 [0.588, 0.604] |
| B: light | 0.928 [0.906, 0.949] | 0.886 [0.867, 0.905] |
| C: hybrid | 0.931 [0.911, 0.950] | 0.892 [0.875, 0.908] |
| Potential field | 0.893 [0.862, 0.925] | 0.807 [0.790, 0.823] |
| Boids | 0.921 [0.902, 0.939] | 0.790 [0.779, 0.802] |
| Lloyd (privileged) | 0.759 [0.707, 0.813] | 0.700 [0.655, 0.744] |

### Paired tests on test plans (same seeds → same start), Wilcoxon signed-rank

| Comparison | Metric | Mean diff | p |
|---|---|---|---|
| A vs RW | Coverage (2nd half) | -0.013 | 0.0082 |
| A vs RW | Area explored | -0.002 | 0.85 |
| A vs RW | Spacing CV | +0.008 | 0.62 |
| A vs RW | Collisions / agent | -2.672 | 1.6e-11 |
| B vs PF | Coverage (2nd half) | +0.079 | 1.6e-11 |
| B vs PF | Area explored | +0.086 | 1.6e-11 |
| B vs PF | Spacing CV | +0.013 | 0.089 |
| B vs PF | Collisions / agent | -0.111 | 4.5e-08 |
| C vs A | Coverage (2nd half) | +0.295 | 1.6e-11 |
| C vs A | Area explored | +0.057 | 3e-09 |
| C vs A | Spacing CV | -0.441 | 1.6e-11 |
| C vs A | Collisions / agent | -4.701 | 1.6e-11 |
| C vs B | Coverage (2nd half) | +0.005 | 0.1 |
| C vs B | Area explored | -0.007 | 8.6e-06 |
| C vs B | Spacing CV | -0.037 | 2.3e-09 |
| C vs B | Collisions / agent | +0.179 | 5.4e-09 |
| A vs B | Coverage (2nd half) | -0.290 | 1.6e-11 |
| A vs B | Area explored | -0.064 | 6.4e-10 |
| A vs B | Spacing CV | +0.404 | 1.6e-11 |
| A vs B | Collisions / agent | +4.881 | 1.6e-11 |
| B vs Boids | Coverage (2nd half) | +0.096 | 1.6e-11 |
| B vs Boids | Area explored | +0.101 | 1.6e-11 |
| B vs Boids | Spacing CV | +0.030 | 1.6e-06 |
| B vs Boids | Collisions / agent | -0.290 | 1.7e-11 |

## Sweeps (test plans; coverage in 2nd half)

All sweep runs use 60 agents with 20% removed at step 400 unless that is the swept variable, so values sit below the intact runs of the main table.

**Wind sway (m/step)**

| variant   |   0.0 |   0.02 |   0.04 |   0.07 |   0.1 |
|:----------|------:|-------:|-------:|-------:|------:|
| RW        | 0.53  |  0.548 |  0.569 |  0.573 | 0.578 |
| A         | 0.519 |  0.533 |  0.545 |  0.547 | 0.536 |
| B         | 0.833 |  0.834 |  0.832 |  0.812 | 0.81  |
| C         | 0.841 |  0.839 |  0.832 |  0.822 | 0.806 |
| PF        | 0.752 |  0.763 |  0.771 |  0.773 | 0.774 |
| Boids     | 0.737 |  0.735 |  0.738 |  0.732 | 0.735 |
| Lloyd     | 0.676 |  0.681 |  0.707 |  0.733 | 0.719 |

**Sensor noise**

| variant   |   0.0 |   0.1 |   0.2 |   0.4 |   0.8 |
|:----------|------:|------:|------:|------:|------:|
| RW        | 0.558 | 0.555 | 0.546 | 0.543 | 0.53  |
| A         | 0.537 | 0.536 | 0.533 | 0.542 | 0.548 |
| B         | 0.835 | 0.833 | 0.814 | 0.789 | 0.752 |
| C         | 0.838 | 0.835 | 0.814 | 0.784 | 0.746 |
| PF        | 0.769 | 0.768 | 0.766 | 0.758 | 0.724 |
| Boids     | 0.738 | 0.736 | 0.73  | 0.717 | 0.675 |
| Lloyd     | 0.681 | 0.681 | 0.681 | 0.681 | 0.681 |

**Swarm size**

| variant   |    10 |    30 |    60 |    90 |   120 |
|:----------|------:|------:|------:|------:|------:|
| RW        | 0.133 | 0.36  | 0.548 | 0.64  | 0.689 |
| A         | 0.133 | 0.351 | 0.533 | 0.617 | 0.651 |
| B         | 0.243 | 0.596 | 0.834 | 0.897 | 0.92  |
| C         | 0.239 | 0.593 | 0.839 | 0.893 | 0.909 |
| PF        | 0.232 | 0.567 | 0.763 | 0.874 | 0.951 |
| Boids     | 0.207 | 0.513 | 0.735 | 0.871 | 0.933 |
| Lloyd     | 0.24  | 0.548 | 0.681 | 0.72  | 0.744 |

**Fraction removed mid-run**

| variant   |   0.0 |   0.2 |   0.4 |   0.6 |
|:----------|------:|------:|------:|------:|
| RW        | 0.61  | 0.548 | 0.46  | 0.342 |
| A         | 0.596 | 0.533 | 0.46  | 0.353 |
| B         | 0.886 | 0.834 | 0.755 | 0.603 |
| C         | 0.892 | 0.839 | 0.757 | 0.596 |
| PF        | 0.807 | 0.763 | 0.706 | 0.574 |
| Boids     | 0.79  | 0.735 | 0.666 | 0.542 |
| Lloyd     | 0.7   | 0.681 | 0.631 | 0.524 |


## Ablations (test plans, 20% removed at step 400)

| Variant | Coverage (2nd half) | Explored | Spacing CV | Collisions / agent |
|---|---|---|---|---|
| A | 0.533 [0.525, 0.540] | 0.868 [0.851, 0.884] | 0.628 [0.603, 0.654] | 5.324 [5.223, 5.425] |
| A-no-regrowth | 0.303 [0.294, 0.312] | 0.427 [0.410, 0.443] | 0.736 [0.656, 0.816] | 3.284 [3.159, 3.417] |
| A-no-damage | 0.549 [0.541, 0.556] | 0.891 [0.878, 0.903] | 0.652 [0.628, 0.676] | 6.022 [5.901, 6.142] |
| B | 0.834 [0.813, 0.855] | 0.932 [0.919, 0.946] | 0.191 [0.173, 0.209] | 1.052 [1.024, 1.080] |
| B-total-stop | 0.252 [0.238, 0.266] | 0.370 [0.352, 0.387] | 0.861 [0.811, 0.910] | 1.469 [1.441, 1.496] |
| B-no-occlusion | 0.839 [0.821, 0.858] | 0.940 [0.928, 0.953] | 0.179 [0.163, 0.196] | 0.996 [0.966, 1.024] |
| B-no-stop | 0.850 [0.830, 0.869] | 0.943 [0.931, 0.955] | 0.181 [0.163, 0.199] | 1.292 [1.259, 1.324] |
| RW | 0.548 [0.540, 0.556] | 0.886 [0.872, 0.900] | 0.655 [0.630, 0.680] | 7.621 [7.503, 7.737] |

## Biology check: gap width vs sway (open arena, 150 agents)

Gap = nearest-neighbour distance minus two body radii. Spearman ρ between sway and gap over all runs; the field prediction is ρ > 0.

| Controller | ρ(sway, gap) | p | gap at min sway | gap at max sway |
|---|---|---|---|---|
| A: abrasion | +0.22 | 0.0007 | 0.342 [0.334, 0.350] | 0.355 [0.344, 0.366] |
| B: light | -0.37 | 2.6e-09 | 0.531 [0.523, 0.539] | 0.496 [0.489, 0.503] |
| C: hybrid | -0.46 | 6.5e-14 | 0.602 [0.595, 0.610] | 0.556 [0.548, 0.564] |
| Random walk | +0.51 | 1.9e-17 | 0.298 [0.291, 0.306] | 0.342 [0.333, 0.350] |

## Scaling (open arena, constant density)

**Coverage (2nd half)**

| variant   |    10 |    30 |   100 |   300 |   1000 |
|:----------|------:|------:|------:|------:|-------:|
| A         | 0.707 | 0.735 | 0.799 | 0.825 |  0.821 |
| B         | 0.992 | 0.999 | 1     | 1     |  1     |
| Boids     | 0.901 | 0.897 | 0.923 | 0.943 |  0.961 |
| C         | 0.986 | 0.996 | 0.999 | 0.999 |  0.999 |
| PF        | 0.998 | 1     | 1     | 1     |  0.999 |
| RW        | 0.71  | 0.747 | 0.804 | 0.832 |  0.843 |

**Wall-clock ms per step (single core)**

| variant   |   10 |   30 |   100 |   300 |   1000 |
|:----------|-----:|-----:|------:|------:|-------:|
| A         |  0.6 |  0.5 |   0.8 |   1.8 |    6.5 |
| B         |  0.5 |  0.8 |   2.3 |   7.9 |   40.8 |
| Boids     |  0.5 |  0.8 |   2.2 |   7.8 |   26.6 |
| C         |  0.5 |  0.9 |   2.3 |   7.6 |   44.2 |
| PF        |  0.4 |  0.7 |   2   |   6.9 |   26.8 |
| RW        |  0.4 |  0.5 |   0.8 |   1.9 |    5.8 |

