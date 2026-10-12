# Evaluation summary

- evaluated at **2026-10-12T00:00:49+0800** on `cuda`, commit `4d0a0bbf690b`
- splits: `val` (1880), `train_clean` (2000)
- **Not a final evaluation.** test and stress were not read. Sections marked *preview* run on val, which chose the learning rates and checkpoints, so they are optimistic -- they validate the analysis, not the conclusions. Rerun with `--final` for test.

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

Baselines are fixed on train: mean params f=161.2, ξ=0.920; best constant f=375.1, ξ=2.939.

## b. Paired comparison: vit_small − resnet34 *(preview on val)*

Per-image error averaged over each group's seeds, then differenced on the same 1880 images. 95% CI from a bootstrap over the 94 panoramas (1000 replicates, seed 0): views of one panorama are correlated, so the panorama is the resampling unit.

| metric | mean difference | 95% CI |
|---|---:|---|
| px | -0.380 | [-0.545, -0.222] |
| % of r_ref | -0.24% | [-0.36%, -0.13%] |

Both intervals lie below 0: `vit_small` is better. (Rule from SPEC 3.14.2; negative favours `vit_small`.)

## c. Generalisation gap

Mean error, px (% of r_ref).

| group | train_clean | val | val − train_clean |
|---|---:|---:|---:|
| resnet34 | 2.888 (2.16%) | 4.232 (3.12%) | +1.343 (+0.96%) |
| vit_small | 2.368 (1.80%) | 3.852 (2.88%) | +1.485 (+1.08%) |
| resnet34_l1 | 4.022 (2.99%) | 4.868 (3.59%) | +0.846 (+0.60%) |
| baseline_mean_params | 17.507 (13.20%) | 17.111 (12.81%) | -0.396 (-0.39%) |
| baseline_best_constant | 14.782 (11.01%) | 14.556 (10.77%) | -0.226 (-0.24%) |

## d. Stratification on val *(preview)*

Mean reprojection error per bin, px (% of r_ref); multi-seed groups use the per-image error averaged over seeds. **Read `panoramas`, not `n`, as the sample size**: views of one panorama are correlated, and the val split has only 94 panoramas in total.

### Texture

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 5 | 14 | 5 | 5.678 (4.16%) | 5.284 (4.11%) | 9.339 (6.39%) | 15.277 (10.40%) | 14.497 (9.85%) |
| Q1 [5.0-22.2] | 374 | 48 | 4.539 (3.21%) | 3.905 (2.78%) | 5.017 (3.56%) | 17.869 (12.73%) | 15.407 (10.86%) |
| Q2 (22.2-28.6] | 373 | 68 | 4.155 (3.12%) | 3.461 (2.62%) | 4.869 (3.64%) | 16.015 (12.01%) | 13.830 (10.26%) |
| Q3 (28.6-36.7] | 373 | 64 | 4.025 (2.94%) | 3.669 (2.73%) | 4.608 (3.38%) | 16.662 (12.54%) | 14.133 (10.50%) |
| Q4 (36.7-48.8] | 373 | 54 | 4.215 (3.15%) | 4.042 (3.08%) | 4.692 (3.50%) | 17.134 (13.06%) | 14.350 (10.83%) |
| Q5 (48.8-109.6] | 373 | 38 | 4.170 (3.14%) | 4.131 (3.13%) | 4.985 (3.76%) | 17.943 (13.79%) | 15.059 (11.44%) |

### Line score (quintiles)

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 [0.0-6.9] | 376 | 56 | 4.712 (3.44%) | 3.871 (2.86%) | 5.269 (3.82%) | 15.873 (11.82%) | 13.703 (10.10%) |
| Q2 (6.9-11.4] | 376 | 70 | 4.599 (3.40%) | 4.121 (3.08%) | 5.092 (3.76%) | 17.107 (12.79%) | 14.491 (10.72%) |
| Q3 (11.4-17.6] | 376 | 78 | 4.144 (3.07%) | 3.975 (2.99%) | 5.078 (3.74%) | 17.702 (13.22%) | 14.968 (11.08%) |
| Q4 (17.6-26.7] | 376 | 75 | 3.992 (2.90%) | 3.810 (2.84%) | 4.424 (3.23%) | 17.917 (13.26%) | 15.254 (11.14%) |
| Q5 (26.7-83.9] | 376 | 46 | 3.713 (2.79%) | 3.485 (2.62%) | 4.477 (3.39%) | 16.958 (12.95%) | 14.364 (10.81%) |

### FOV (deg)

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100-130 | 639 | 94 | 4.523 (3.29%) | 3.813 (2.83%) | 5.179 (3.75%) | 18.909 (13.00%) | 17.035 (11.63%) |
| 130-160 | 565 | 94 | 4.363 (3.26%) | 4.073 (3.08%) | 4.992 (3.72%) | 10.535 (7.64%) | 10.901 (7.90%) |
| 160-190 | 385 | 92 | 4.050 (3.03%) | 3.822 (2.89%) | 4.609 (3.48%) | 14.515 (12.19%) | 13.126 (10.67%) |
| 190-220 | 291 | 90 | 3.579 (2.59%) | 3.550 (2.57%) | 4.287 (3.11%) | 29.369 (23.25%) | 18.101 (14.59%) |

### Image shape

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| full-frame | 951 | 94 | 4.968 (3.14%) | 4.250 (2.68%) | 5.727 (3.62%) | 18.673 (11.79%) | 16.826 (10.62%) |
| circular | 929 | 94 | 3.478 (3.11%) | 3.446 (3.08%) | 3.988 (3.56%) | 15.512 (13.85%) | 12.233 (10.92%) |

### primary_category

| bin | n | panoramas | resnet34 | vit_small | resnet34_l1 | baseline_mean_params | baseline_best_constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| Abandoned & Ruins | 100 | 5 | 3.617 (2.62%) | 3.620 (2.68%) | 4.273 (3.09%) | 16.737 (12.59%) | 14.723 (10.94%) |
| Coast & Water | 140 | 7 | 4.388 (3.19%) | 3.872 (2.87%) | 5.080 (3.66%) | 17.100 (12.65%) | 14.505 (10.61%) |
| Desert & Arid | 20 | 1 | 4.439 (3.30%) | 3.461 (2.65%) | 5.226 (3.78%) | 18.443 (14.03%) | 15.304 (11.44%) |
| Fields & Countryside | 240 | 12 | 4.452 (3.25%) | 3.644 (2.71%) | 4.983 (3.65%) | 16.350 (12.12%) | 13.843 (10.17%) |
| Forest & Woodland | 120 | 6 | 4.571 (3.43%) | 4.202 (3.21%) | 5.112 (3.78%) | 18.044 (13.45%) | 15.589 (11.47%) |
| Industrial | 40 | 2 | 3.834 (2.87%) | 3.198 (2.50%) | 4.331 (3.32%) | 14.651 (11.53%) | 12.550 (9.70%) |
| Interiors | 280 | 14 | 3.509 (2.56%) | 3.480 (2.57%) | 4.080 (3.04%) | 17.307 (13.05%) | 14.884 (11.08%) |
| Mountains & Hills | 160 | 8 | 4.091 (3.03%) | 3.534 (2.65%) | 4.688 (3.49%) | 16.440 (12.24%) | 14.289 (10.50%) |
| Parks & Gardens | 200 | 10 | 5.114 (3.86%) | 4.601 (3.47%) | 5.620 (4.19%) | 16.183 (12.05%) | 13.957 (10.32%) |
| Streets & Town | 300 | 15 | 4.173 (3.07%) | 4.088 (3.06%) | 4.920 (3.65%) | 17.988 (13.57%) | 14.791 (11.04%) |
| Studio | 280 | 14 | 4.319 (3.18%) | 3.844 (2.84%) | 5.119 (3.73%) | 17.672 (13.18%) | 14.924 (10.99%) |

Excluding texture < 5 (14 images from 5 panoramas):

| group | all | texture ≥ 5 |
|---|---:|---:|
| resnet34 | 4.232 (3.12%) | 4.221 (3.11%) |
| vit_small | 3.852 (2.88%) | 3.842 (2.87%) |
| resnet34_l1 | 4.868 (3.59%) | 4.834 (3.57%) |
| baseline_mean_params | 17.111 (12.81%) | 17.125 (12.83%) |
| baseline_best_constant | 14.556 (10.77%) | 14.556 (10.78%) |

### Texture × line score

Line score split at its median on val (14.44): `low` is strictly below. Checks whether the most textured frames are the ones short of straight lines (SPEC 3.14.3 item 5b). Cells: mean error px (% of r_ref).

| texture bin | n | panoramas | low line % | panoramas low / high | resnet34 low | resnet34 high | vit_small low | vit_small high | resnet34_l1 low | resnet34_l1 high |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 5 | 14 | 5 | 100% | 5 / 0 | 5.678 (4.16%) | — | 5.284 (4.11%) | — | 9.339 (6.39%) | — |
| Q1 [5.0-22.2] | 374 | 48 | 78% | 43 / 24 | 4.700 (3.29%) | 3.975 (2.95%) | 4.063 (2.87%) | 3.353 (2.48%) | 5.235 (3.66%) | 4.252 (3.21%) |
| Q2 (22.2-28.6] | 373 | 68 | 68% | 45 / 43 | 4.263 (3.23%) | 3.920 (2.88%) | 3.396 (2.61%) | 3.602 (2.66%) | 4.819 (3.64%) | 4.978 (3.64%) |
| Q3 (28.6-36.7] | 373 | 64 | 49% | 42 / 49 | 4.459 (3.30%) | 3.612 (2.60%) | 3.799 (2.87%) | 3.545 (2.59%) | 5.097 (3.77%) | 4.142 (3.00%) |
| Q4 (36.7-48.8] | 373 | 54 | 35% | 30 / 49 | 4.696 (3.55%) | 3.958 (2.94%) | 4.551 (3.54%) | 3.770 (2.83%) | 5.200 (3.87%) | 4.421 (3.31%) |
| Q5 (48.8-109.6] | 373 | 38 | 18% | 15 / 35 | 5.144 (3.87%) | 3.953 (2.98%) | 5.347 (3.96%) | 3.860 (2.95%) | 5.685 (4.27%) | 4.829 (3.65%) |

Line score: OpenCV LSD on the grey image; segments ≥ 8 px with the midpoint inside the valid mask (eroded 2× 3×3); non-radial = midpoint ≥ 0.2·r_ref from the principal point and ≥ 30° off the radial direction; length per 1000 valid pixels. Mask check (images with non-zero pixels outside the rebuilt mask): val 0/1880, train_clean 0/2000.

## e. Stress (Pure Skies)

Not evaluated — stress is read only with `--final`.

## f. FOV bias on val *(preview)*

Predicted FOV from `geometry.ucm.fov_from_f` at the sample's r_ref, minus the true FOV, in degrees; per image averaged over the group's seeds. `range` is the 2.5–97.5th percentile of that per-image bias (spread, not a CI). Regression to the mean shows as positive bias at narrow FOVs, negative at wide ones, i.e. a negative slope (SPEC 3.14.3 item 6). Slope CI: panorama bootstrap.

### resnet34

Mean bias +3.90°; slope against true FOV -0.085 deg/deg, 95% CI [-0.099, -0.073]; 0 of 1880 images with an undefined predicted FOV.

| true FOV | n | panoramas | mean bias (°) | range (°) |
|---|---:|---:|---:|---|
| 100-110 | 218 | 89 | +9.25 | [-1.10, +29.07] |
| 110-120 | 228 | 91 | +5.81 | [-7.74, +23.26] |
| 120-130 | 193 | 85 | +5.39 | [-10.16, +25.56] |
| 130-140 | 193 | 80 | +3.49 | [-11.12, +22.53] |
| 140-150 | 195 | 81 | +4.09 | [-14.44, +24.81] |
| 150-160 | 177 | 83 | +2.52 | [-13.21, +24.75] |
| 160-170 | 123 | 68 | +4.27 | [-13.25, +28.30] |
| 170-180 | 140 | 74 | +3.89 | [-12.38, +21.57] |
| 180-190 | 122 | 65 | +3.65 | [-14.12, +17.26] |
| 190-200 | 110 | 69 | +2.59 | [-15.62, +14.82] |
| 200-210 | 88 | 63 | -2.46 | [-17.96, +9.55] |
| 210-220 | 93 | 51 | -5.93 | [-20.89, +4.54] |

### vit_small

Mean bias +2.15°; slope against true FOV -0.060 deg/deg, 95% CI [-0.072, -0.046]; 1 of 1880 images with an undefined predicted FOV.

| true FOV | n | panoramas | mean bias (°) | range (°) |
|---|---:|---:|---:|---|
| 100-110 | 217 | 89 | +5.75 | [-2.97, +20.26] |
| 110-120 | 228 | 91 | +3.32 | [-8.07, +22.37] |
| 120-130 | 193 | 85 | +2.66 | [-10.81, +18.11] |
| 130-140 | 193 | 80 | +1.64 | [-12.55, +21.69] |
| 140-150 | 195 | 81 | +2.89 | [-13.38, +19.28] |
| 150-160 | 177 | 83 | +1.90 | [-14.28, +19.26] |
| 160-170 | 123 | 68 | +2.29 | [-15.43, +20.61] |
| 170-180 | 140 | 74 | +2.80 | [-16.11, +21.03] |
| 180-190 | 122 | 65 | +2.35 | [-13.70, +21.29] |
| 190-200 | 110 | 69 | +1.57 | [-19.36, +18.55] |
| 200-210 | 88 | 63 | -3.09 | [-18.26, +10.24] |
| 210-220 | 93 | 51 | -5.92 | [-26.89, +8.92] |

### resnet34_l1

Mean bias +0.44°; slope against true FOV -0.094 deg/deg, 95% CI [-0.112, -0.077]; 2 of 1880 images with an undefined predicted FOV.

| true FOV | n | panoramas | mean bias (°) | range (°) |
|---|---:|---:|---:|---|
| 100-110 | 218 | 89 | +7.75 | [-5.38, +27.94] |
| 110-120 | 226 | 91 | +3.12 | [-10.19, +20.94] |
| 120-130 | 193 | 85 | +0.66 | [-15.57, +17.19] |
| 130-140 | 193 | 80 | -0.72 | [-17.95, +22.87] |
| 140-150 | 195 | 81 | -0.59 | [-20.22, +17.82] |
| 150-160 | 177 | 83 | -0.13 | [-21.79, +21.94] |
| 160-170 | 123 | 68 | +0.41 | [-19.75, +25.48] |
| 170-180 | 140 | 74 | +0.61 | [-26.23, +21.50] |
| 180-190 | 122 | 65 | +0.05 | [-19.24, +18.42] |
| 190-200 | 110 | 69 | -1.29 | [-18.23, +15.03] |
| 200-210 | 88 | 63 | -5.54 | [-22.68, +11.87] |
| 210-220 | 93 | 51 | -10.05 | [-25.34, +4.44] |

## Figures

- `error_vs_texture`: [figures/val_error_vs_texture.png](figures/val_error_vs_texture.png)
- `error_vs_fov`: [figures/val_error_vs_fov.png](figures/val_error_vs_fov.png)
- `error_vs_line_score`: [figures/val_error_vs_line_score.png](figures/val_error_vs_line_score.png)
- `fov_bias`: [figures/val_fov_bias.png](figures/val_fov_bias.png)
- `worst_resnet34`: [figures/val_worst_resnet34.png](figures/val_worst_resnet34.png)
- `worst_vit_small`: [figures/val_worst_vit_small.png](figures/val_worst_vit_small.png)
- `worst_resnet34_l1`: [figures/val_worst_resnet34_l1.png](figures/val_worst_resnet34_l1.png)
