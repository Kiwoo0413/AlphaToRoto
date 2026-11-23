# AlphaToRoto Quantitative Benchmark Results (Tables 1, 2, and 3)

## Table 1: Parametric Vectorization Comparison (Control Point Sparsity, Accuracy & Speed)

| Method | Points (N) | J (%) | F (%) | J&F (%) | Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| RDP Polygon Vectorization | 33.2 | 99.06 | 100.00 | 99.53 | 1.02 |
| SAM 2 Auto-Trace (Nuke) | 58.2 | 99.20 | 99.78 | 99.49 | 1.48 |
| Least-Squares B-Spline | 26.0 | 99.64 | 100.00 | 99.82 | 7.36 |
| Proposed Hermite-Bezier | 18.4 | 96.56 | 91.06 | 93.81 | 4.07 |

## Table 2: Dynamic Articulated Sequence Editability (NoM), Temporal Stability & Drift

| Method | NoM | T (Flicker) | dJ_100 | Gap (%) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Frame-wise SAM 2 + RDP | 104 | 33.28 | 4.11 | 0.00 | 1220.5 |
| Single-Shape Unidirectional | 35 | 1.39 | 18.46 | 0.00 | 437.3 |
| Single-Shape Bidirectional | 35 | 1.39 | 17.48 | 0.00 | 457.2 |
| Proposed Part + Single-Ref | 22 | 1.60 | 9.64 | 0.00 | 263.8 |
| Proposed AlphaToRoto (Multi-Ref) | 3 | 1.60 | 9.74 | 0.00 | 255.6 |

## Table 3: Ablation Study of Core AlphaToRoto Components

| Configuration | NoM | J&F (%) | Gap (%) | RMSE (px) |
| :--- | :---: | :---: | :---: | :---: |
| w/o Kinematic-Geometric Part Decomp. | 25 | 85.39 | 0.00 | 3.84 |
| w/o Seam Overlap Margin (0 px) | 11 | 87.86 | 0.39 | 24.20 |
| w/o Visibility Weighting (Linear) | 35 | 92.02 | 0.00 | 2.43 |
| w/o Chord-Aligned Tangent Rotation | 35 | 91.20 | 0.00 | 2.49 |
| Full Proposed AlphaToRoto (10-15 px) | 15 | 92.69 | 0.00 | 3.33 |
