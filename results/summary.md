# Evaluation summary

- evaluated at **2026-10-12T11:33:58+0800** on `cuda`, commit `73b7939d432d`
- splits: `val` (1880), `train_clean` (2000), `test` (1780), `stress` (1180)
- **Final evaluation**: test and stress included.

## Sanity check

Val reprojection error recomputed from each checkpoint against the MLflow record (must agree within 0.01 px).

| run | recomputed | MLflow | abs diff | step |
|---|---:|---:|---:|---:|
| `resnet34_pre_p100_geo_lr3e-3_s0` | 4.2156 | 4.2156 | 0.00000 | 5999 |
| `resnet34_pre_p100_geo_lr3e-3_s1` | 4.2646 | 4.2646 | 0.00000 | 5999 |
| `resnet34_pre_p100_geo_lr3e-3_s2` | 4.2155 | 4.2155 | 0.00000 | 5999 |
| `vit_small_pre_p100_geo_lr5e-5_s0` | 3.8552 | 3.8552 | 0.00000 | 5999 |
| `vit_small_pre_p100_geo_lr5e-5_s1` | 3.8505 | 3.8505 | 0.00000 | 5499 |
| `vit_small_pre_p100_geo_lr5e-5_s2` | 3.8515 | 3.8515 | 0.00000 | 4999 |
| `resnet34_pre_p100_l1_lr3e-3_s0` | 4.8679 | 4.8679 | 0.00000 | 5999 |

## a. Main table

Reprojection error in px and as % of the reference radius `r_ref` (112 px circular, 158 px full-frame). Pixels are not comparable across image shapes; the percentage is. Multi-seed groups show the mean ± std over seeds of each statistic.

### val (1880 images)

| group | runs | mean px | mean % | median px | median % | p90 px | p90 % | f MAE (px) | ξ MAE | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet34 | 3 | 4.232 ± 0.028 | 3.12 ± 0.02 | 3.373 ± 0.090 | 2.50 ± 0.06 | 8.530 ± 0.075 | 6.28 ± 0.05 | 34.62 ± 0.87 | 0.263 ± 0.008 | 0.0000 ± 0.0000 |
| vit_small | 3 | 3.852 ± 0.002 | 2.88 ± 0.00 | 3.066 ± 0.093 | 2.27 ± 0.04 | 7.755 ± 0.147 | 5.74 ± 0.06 | 27.18 ± 0.46 | 0.196 ± 0.004 | 0.0000 ± 0.0000 |
| resnet34_l1 | 1 | 4.868 | 3.59 | 3.908 | 2.88 | 10.398 | 7.54 | 21.12 | 0.119 | 0.0000 |
| baseline_mean_params | 1 | 17.111 | 12.81 | 14.537 | 10.91 | 34.282 | 24.21 | 54.33 | 0.305 | 0.0000 |
| baseline_best_constant | 1 | 14.556 | 10.77 | 11.936 | 8.96 | 28.436 | 20.62 | 200.55 | 2.009 | 0.0000 |

### train_clean (2000 images)

| group | runs | mean px | mean % | median px | median % | p90 px | p90 % | f MAE (px) | ξ MAE | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet34 | 3 | 2.888 ± 0.046 | 2.16 ± 0.04 | 2.352 ± 0.055 | 1.79 ± 0.04 | 5.676 ± 0.177 | 4.18 ± 0.06 | 31.56 ± 0.68 | 0.245 ± 0.006 | 0.0000 ± 0.0000 |
| vit_small | 3 | 2.368 ± 0.063 | 1.80 ± 0.05 | 1.958 ± 0.066 | 1.47 ± 0.04 | 4.543 ± 0.157 | 3.53 ± 0.09 | 23.66 ± 0.35 | 0.184 ± 0.003 | 0.0000 ± 0.0000 |
| resnet34_l1 | 1 | 4.022 | 2.99 | 3.063 | 2.32 | 8.489 | 6.31 | 15.07 | 0.093 | 0.0000 |
| baseline_mean_params | 1 | 17.507 | 13.20 | 15.015 | 11.28 | 34.691 | 26.26 | 54.11 | 0.315 | 0.0000 |
| baseline_best_constant | 1 | 14.782 | 11.01 | 12.430 | 9.26 | 27.907 | 21.18 | 203.05 | 2.020 | 0.0000 |

### test (1780 images)

| group | runs | mean px | mean % | median px | median % | p90 px | p90 % | f MAE (px) | ξ MAE | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet34 | 3 | 4.375 ± 0.042 | 3.24 ± 0.04 | 3.338 ± 0.068 | 2.52 ± 0.02 | 9.011 ± 0.037 | 6.74 ± 0.07 | 34.47 ± 0.74 | 0.255 ± 0.006 | 0.0000 ± 0.0000 |
| vit_small | 3 | 4.078 ± 0.032 | 3.05 ± 0.03 | 3.226 ± 0.053 | 2.43 ± 0.02 | 8.296 ± 0.052 | 6.14 ± 0.12 | 27.76 ± 0.37 | 0.198 ± 0.003 | 0.0000 ± 0.0000 |
| resnet34_l1 | 1 | 4.659 | 3.45 | 3.594 | 2.66 | 9.903 | 7.22 | 20.42 | 0.118 | 0.0000 |
| baseline_mean_params | 1 | 17.341 | 13.10 | 14.877 | 11.08 | 34.318 | 25.22 | 53.33 | 0.319 | 0.0000 |
| baseline_best_constant | 1 | 14.656 | 10.94 | 12.091 | 9.14 | 28.386 | 20.85 | 205.77 | 2.040 | 0.0000 |

### stress (1180 images)

| group | runs | mean px | mean % | median px | median % | p90 px | p90 % | f MAE (px) | ξ MAE | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet34 | 3 | 7.168 ± 0.091 | 5.38 ± 0.09 | 5.193 ± 0.133 | 3.89 ± 0.02 | 15.119 ± 0.201 | 11.40 ± 0.40 | 72.39 ± 5.73 | 0.525 ± 0.031 | 0.0000 ± 0.0000 |
| vit_small | 3 | 7.479 ± 0.708 | 5.61 ± 0.53 | 5.577 ± 0.431 | 4.17 ± 0.24 | 15.935 ± 1.953 | 11.87 ± 1.47 | 46.17 ± 3.43 | 0.298 ± 0.011 | 0.0000 ± 0.0000 |
| resnet34_l1 | 1 | 8.210 | 6.12 | 5.789 | 4.30 | 17.063 | 12.40 | 29.94 | 0.187 | 0.0000 |
| baseline_mean_params | 1 | 17.061 | 12.85 | 14.809 | 11.13 | 34.152 | 25.10 | 53.60 | 0.319 | 0.0000 |
| baseline_best_constant | 1 | 14.523 | 10.81 | 12.216 | 9.16 | 27.926 | 20.47 | 204.15 | 2.022 | 0.0000 |

Baselines are fixed on train: mean params f=161.2, ξ=0.920; best constant f=375.1, ξ=2.939.

## b. Paired comparison: vit_small − resnet34

Per-image error averaged over each group's seeds, then differenced on the same 1780 images. 95% CI from a bootstrap over the 89 panoramas (1000 replicates, seed 0): views of one panorama are correlated, so the panorama is the resampling unit.

| metric | mean difference | 95% CI |
|---|---:|---|
| px | -0.298 | [-0.454, -0.150] |
| % of r_ref | -0.20% | [-0.32%, -0.09%] |

Both intervals lie below 0: `vit_small` is better. (Rule from SPEC 3.14.2; negative favours `vit_small`.)

## c. Generalisation gap

Mean error, px (% of r_ref).

| group | train_clean | val | test | val − train_clean |
|---|---:|---:|---:|---:|
| resnet34 | 2.888 (2.16%) | 4.232 (3.12%) | 4.375 (3.24%) | +1.343 (+0.96%) |
| vit_small | 2.368 (1.80%) | 3.852 (2.88%) | 4.078 (3.05%) | +1.485 (+1.08%) |
| resnet34_l1 | 4.022 (2.99%) | 4.868 (3.59%) | 4.659 (3.45%) | +0.846 (+0.60%) |
| baseline_mean_params | 17.507 (13.20%) | 17.111 (12.81%) | 17.341 (13.10%) | -0.396 (-0.39%) |
| baseline_best_constant | 14.782 (11.01%) | 14.556 (10.77%) | 14.656 (10.94%) | -0.226 (-0.24%) |

## d. Stratification on test

Mean reprojection error per bin, px (% of r_ref); multi-seed groups use the per-image error averaged over seeds. **Read `panoramas`, not `n`, as the sample size**: views of one panorama are correlated, and the test split has only 89 panoramas in total.

### Texture

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5 | 3 | 1 | 4.331 (2.99%) | 2.532 (1.96%) | 6.823 (4.98%) | 18.642 (11.99%) | 17.986 (11.71%) |
| Q1 [5.0-25.5] | 356 | 43 | 4.671 (3.33%) | 3.984 (2.83%) | 5.076 (3.64%) | 18.162 (13.08%) | 15.685 (11.16%) |
| Q2 (25.5-34.1] | 355 | 61 | 4.404 (3.29%) | 3.884 (2.92%) | 4.707 (3.49%) | 16.439 (12.56%) | 14.023 (10.58%) |
| Q3 (34.1-42.2] | 355 | 59 | 4.382 (3.26%) | 4.240 (3.18%) | 4.536 (3.39%) | 17.929 (13.73%) | 14.950 (11.29%) |
| Q4 (42.2-56.1] | 355 | 53 | 4.027 (2.98%) | 3.941 (2.95%) | 4.404 (3.25%) | 17.822 (13.64%) | 15.018 (11.33%) |
| Q5 (56.1-124.8] | 356 | 32 | 4.392 (3.36%) | 4.352 (3.35%) | 4.552 (3.48%) | 16.341 (12.52%) | 13.576 (10.35%) |

### Line score (quintiles)

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 [0.0-8.0] | 356 | 44 | 4.898 (3.65%) | 4.056 (3.05%) | 5.158 (3.87%) | 17.442 (13.46%) | 14.498 (11.07%) |
| Q2 (8.0-15.1] | 356 | 66 | 4.463 (3.34%) | 3.991 (2.98%) | 4.715 (3.50%) | 17.965 (13.69%) | 14.975 (11.28%) |
| Q3 (15.1-22.9] | 356 | 75 | 4.407 (3.23%) | 4.242 (3.15%) | 4.881 (3.57%) | 17.568 (13.17%) | 15.085 (11.16%) |
| Q4 (22.9-32.7] | 356 | 63 | 4.251 (3.16%) | 4.121 (3.06%) | 4.569 (3.40%) | 16.992 (12.58%) | 14.619 (10.69%) |
| Q5 (32.7-96.0] | 356 | 49 | 3.858 (2.83%) | 3.977 (2.98%) | 3.970 (2.92%) | 16.736 (12.63%) | 14.103 (10.51%) |

### FOV (deg)

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100-130 | 620 | 89 | 4.768 (3.53%) | 4.340 (3.23%) | 5.014 (3.71%) | 18.325 (12.67%) | 16.529 (11.36%) |
| 130-160 | 493 | 89 | 4.579 (3.40%) | 4.275 (3.21%) | 4.794 (3.56%) | 10.587 (7.70%) | 10.879 (7.91%) |
| 160-190 | 379 | 89 | 4.096 (3.07%) | 3.868 (2.91%) | 4.317 (3.22%) | 14.629 (12.30%) | 13.334 (10.85%) |
| 190-220 | 288 | 87 | 3.547 (2.60%) | 3.451 (2.54%) | 4.112 (3.03%) | 30.351 (24.35%) | 18.829 (15.36%) |

### Image shape

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-frame | 884 | 89 | 5.105 (3.22%) | 4.585 (2.89%) | 5.449 (3.44%) | 18.313 (11.56%) | 16.491 (10.41%) |
| circular | 896 | 89 | 3.655 (3.26%) | 3.576 (3.19%) | 3.879 (3.46%) | 16.381 (14.63%) | 12.845 (11.47%) |

### primary_category

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| Abandoned & Ruins | 100 | 5 | 3.669 (2.73%) | 3.578 (2.66%) | 3.580 (2.62%) | 16.659 (12.40%) | 14.252 (10.50%) |
| Coast & Water | 120 | 6 | 5.316 (3.91%) | 4.412 (3.27%) | 5.894 (4.36%) | 18.368 (14.01%) | 15.172 (11.44%) |
| Desert & Arid | 20 | 1 | 4.705 (3.70%) | 4.195 (3.15%) | 5.720 (4.42%) | 16.137 (12.14%) | 13.393 (10.11%) |
| Fields & Countryside | 200 | 10 | 4.211 (3.12%) | 3.851 (2.93%) | 4.363 (3.22%) | 17.124 (13.13%) | 14.619 (11.02%) |
| Forest & Woodland | 100 | 5 | 4.379 (3.29%) | 4.306 (3.29%) | 4.823 (3.62%) | 17.577 (12.93%) | 14.536 (10.65%) |
| Industrial | 20 | 1 | 3.436 (2.51%) | 2.641 (1.90%) | 3.071 (2.33%) | 22.609 (17.08%) | 17.588 (13.19%) |
| Interiors | 280 | 14 | 4.243 (3.13%) | 4.110 (3.04%) | 4.401 (3.26%) | 17.364 (13.13%) | 14.831 (11.05%) |
| Mountains & Hills | 180 | 9 | 4.131 (3.11%) | 3.619 (2.73%) | 3.980 (2.99%) | 16.848 (12.95%) | 14.039 (10.70%) |
| Parks & Gardens | 180 | 9 | 4.161 (3.18%) | 3.920 (3.01%) | 4.478 (3.40%) | 17.019 (13.04%) | 14.237 (10.82%) |
| Streets & Town | 380 | 19 | 4.634 (3.42%) | 4.454 (3.31%) | 5.014 (3.69%) | 17.418 (13.14%) | 14.922 (11.10%) |
| Studio | 200 | 10 | 4.493 (3.22%) | 4.164 (3.03%) | 5.182 (3.76%) | 17.311 (12.77%) | 14.662 (10.67%) |

Excluding texture < 5 (3 images from 1 panoramas):

| group | all | texture ≥ 5 |
|---|---:|---:|
| resnet34 | 4.375 (3.24%) | 4.375 (3.24%) |
| vit_small | 4.078 (3.05%) | 4.080 (3.05%) |
| resnet34_l1 | 4.659 (3.45%) | 4.655 (3.45%) |
| baseline_mean_params | 17.341 (13.10%) | 17.338 (13.11%) |
| baseline_best_constant | 14.656 (10.94%) | 14.650 (10.94%) |

### Texture × line score

Line score split at its median on test (18.91): `low` is strictly below. Checks whether the most textured frames are the ones short of straight lines (SPEC 3.14.3 item 5b). Cells: mean error px (% of r_ref).

| texture bin | n | panoramas | low line % | panoramas low / high | resnet34 low | resnet34 high | vit_small low | vit_small high | resnet34_l1 low | resnet34_l1 high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5 | 3 | 1 | 100% | 1 / 0 | 4.331 (2.99%) | — | 2.532 (1.96%) | — | 6.823 (4.98%) | — |
| Q1 [5.0-25.5] | 356 | 43 | 92% | 38 / 14 | 4.752 (3.40%) | 3.719 (2.46%) | 3.959 (2.83%) | 4.276 (2.84%) | 5.174 (3.72%) | 3.926 (2.64%) |
| Q2 (25.5-34.1] | 355 | 61 | 70% | 50 / 36 | 4.383 (3.35%) | 4.453 (3.17%) | 3.705 (2.86%) | 4.309 (3.08%) | 4.544 (3.44%) | 5.094 (3.60%) |
| Q3 (34.1-42.2] | 355 | 59 | 50% | 45 / 44 | 4.704 (3.60%) | 4.066 (2.94%) | 4.559 (3.50%) | 3.926 (2.86%) | 4.596 (3.53%) | 4.476 (3.25%) |
| Q4 (42.2-56.1] | 355 | 53 | 22% | 28 / 45 | 4.508 (3.36%) | 3.889 (2.86%) | 4.102 (3.09%) | 3.895 (2.91%) | 4.985 (3.72%) | 4.238 (3.12%) |
| Q5 (56.1-124.8] | 356 | 32 | 15% | 11 / 31 | 4.796 (3.87%) | 4.320 (3.27%) | 4.262 (3.38%) | 4.368 (3.35%) | 5.114 (4.10%) | 4.452 (3.37%) |

Line score: OpenCV LSD on the grey image; segments ≥ 8 px with the midpoint inside the valid mask (eroded 2× 3×3); non-radial = midpoint ≥ 0.2·r_ref from the principal point and ≥ 30° off the radial direction; length per 1000 valid pixels. Mask check (images with non-zero pixels outside the rebuilt mask): val 0/1880, train_clean 0/2000, test 0/1780, stress 0/1180.

## e. Stress (Pure Skies)

Sibling in train: 49 panoramas / 980 images. Sibling not in train: 10 panoramas / 200 images — **too few for a conclusion; reference only.**

Mean error, px (% of r_ref).

| group | all | sibling in train | sibling not in train |
|---|---:|---:|---:|
| resnet34 | 7.168 (5.38%) | 7.623 (5.75%) | 4.942 (3.58%) |
| vit_small | 7.479 (5.61%) | 7.699 (5.81%) | 6.399 (4.65%) |
| resnet34_l1 | 8.210 (6.12%) | 8.508 (6.37%) | 6.751 (4.86%) |
| baseline_mean_params | 17.061 (12.85%) | 16.977 (12.85%) | 17.474 (12.80%) |
| baseline_best_constant | 14.523 (10.81%) | 14.430 (10.80%) | 14.976 (10.84%) |

## f. FOV bias on test

Predicted FOV from `geometry.ucm.fov_from_f` at the sample's r_ref, minus the true FOV, in degrees; per image averaged over the group's seeds. `range` is the 2.5–97.5th percentile of that per-image bias (spread, not a CI). Regression to the mean shows as positive bias at narrow FOVs, negative at wide ones, i.e. a negative slope (SPEC 3.14.3 item 6). Slope CI: panorama bootstrap.

### resnet34

Mean bias +3.78°; slope against true FOV -0.089 deg/deg, 95% CI [-0.104, -0.076]; 0 of 1780 images with an undefined predicted FOV.

| true FOV | n | panoramas | mean bias (°) | range (°) |
|---|---:|---:|---:|---|
| 100-110 | 227 | 87 | +8.60 | [-2.23, +29.51] |
| 110-120 | 193 | 80 | +6.71 | [-5.22, +26.86] |
| 120-130 | 200 | 81 | +3.66 | [-12.98, +25.70] |
| 130-140 | 176 | 84 | +4.14 | [-12.85, +28.58] |
| 140-150 | 168 | 75 | +5.66 | [-10.73, +26.66] |
| 150-160 | 149 | 69 | +3.90 | [-15.87, +26.88] |
| 160-170 | 135 | 72 | +3.85 | [-10.36, +23.68] |
| 170-180 | 128 | 67 | +2.33 | [-15.29, +24.40] |
| 180-190 | 116 | 66 | +1.70 | [-17.70, +18.91] |
| 190-200 | 104 | 63 | +1.95 | [-17.94, +14.83] |
| 200-210 | 87 | 58 | -2.24 | [-20.44, +9.78] |
| 210-220 | 97 | 65 | -5.45 | [-24.32, +4.18] |

### vit_small

Mean bias +2.21°; slope against true FOV -0.067 deg/deg, 95% CI [-0.082, -0.054]; 0 of 1780 images with an undefined predicted FOV.

| true FOV | n | panoramas | mean bias (°) | range (°) |
|---|---:|---:|---:|---|
| 100-110 | 227 | 87 | +6.17 | [-5.01, +21.57] |
| 110-120 | 193 | 80 | +3.97 | [-8.53, +24.11] |
| 120-130 | 200 | 81 | +1.81 | [-11.21, +22.52] |
| 130-140 | 176 | 84 | +2.57 | [-13.03, +23.51] |
| 140-150 | 168 | 75 | +3.41 | [-9.97, +21.45] |
| 150-160 | 149 | 69 | +2.91 | [-12.38, +22.23] |
| 160-170 | 135 | 72 | +1.96 | [-14.50, +17.26] |
| 170-180 | 128 | 67 | +2.15 | [-14.75, +18.97] |
| 180-190 | 116 | 66 | -0.06 | [-20.41, +18.25] |
| 190-200 | 104 | 63 | +0.86 | [-18.13, +16.39] |
| 200-210 | 87 | 58 | -3.12 | [-19.80, +12.22] |
| 210-220 | 97 | 65 | -4.16 | [-17.95, +11.08] |

### resnet34_l1

Mean bias +0.58°; slope against true FOV -0.095 deg/deg, 95% CI [-0.112, -0.079]; 0 of 1780 images with an undefined predicted FOV.

| true FOV | n | panoramas | mean bias (°) | range (°) |
|---|---:|---:|---:|---|
| 100-110 | 227 | 87 | +6.92 | [-5.10, +27.05] |
| 110-120 | 193 | 80 | +3.52 | [-9.85, +23.84] |
| 120-130 | 200 | 81 | +0.25 | [-15.30, +20.83] |
| 130-140 | 176 | 84 | +0.01 | [-14.99, +20.29] |
| 140-150 | 168 | 75 | +1.43 | [-15.26, +24.26] |
| 150-160 | 149 | 69 | +0.56 | [-18.34, +23.46] |
| 160-170 | 135 | 72 | +0.74 | [-17.95, +21.94] |
| 170-180 | 128 | 67 | +0.18 | [-18.87, +19.74] |
| 180-190 | 116 | 66 | -1.33 | [-24.09, +19.45] |
| 190-200 | 104 | 63 | -2.52 | [-22.02, +12.83] |
| 200-210 | 87 | 58 | -6.35 | [-32.56, +11.00] |
| 210-220 | 97 | 65 | -7.72 | [-28.87, +5.38] |

## Figures

- `error_vs_texture`: [figures/test_error_vs_texture.png](figures/test_error_vs_texture.png)
- `error_vs_fov`: [figures/test_error_vs_fov.png](figures/test_error_vs_fov.png)
- `error_vs_line_score`: [figures/test_error_vs_line_score.png](figures/test_error_vs_line_score.png)
- `fov_bias`: [figures/test_fov_bias.png](figures/test_fov_bias.png)
- `worst_resnet34`: [figures/test_worst_resnet34.png](figures/test_worst_resnet34.png)
- `worst_vit_small`: [figures/test_worst_vit_small.png](figures/test_worst_vit_small.png)
- `worst_resnet34_l1`: [figures/test_worst_resnet34_l1.png](figures/test_worst_resnet34_l1.png)
