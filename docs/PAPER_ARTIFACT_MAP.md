# arXiv experiment artifact map

This map records the **active experimental** tables and figures in the separate
`main-arxiv.tex` manuscript, inspected on 2026-10-10. It is a release audit,
not a claim that every historical run has been reproduced. Theoretical diagrams
and material inside comments or the excluded translation `\iffalse` block are
outside this map. Paper labels below are stable LaTeX labels, not file paths in
this repository.

| Manuscript item | Candidate source/data | Reconstruction and evidence | Release status |
| --- | --- | --- | --- |
| Figure `fig:normalization-domains` | `figures/plot_normalization_domain_schematic.py` | Writes `figures/generated/fig_normalization_domains.pdf`; no experimental data | Available as a schematic |
| Main Table `tab:best_mse` | `experiments/pinn/run_helmholtz.py`; `experiments/pinn/data/table2_selected_runs.csv` | `python experiments/pinn/verify_data.py` checks its selected cells against independent printed-value constants | Historical selected results checked; two selected full extracted runs match printed precision, but the table-wide search is not reproduced |
| Main Table `tab:cnn_operator_comparison` | CNN runners in `experiments/cnn/`; `figures/data/cnn_runs.csv` | `python figures/verify_data.py` checks all four printed LN/GN/PLN fitting values and the full factor grid; [one full random-label PLN[4] cell](CNN_FULL_VALIDATION_20261010.md) matched its archived final fitting accuracy | Aggregation available; one saturated cell checked, but the cross-method comparison and three-seed table have not been freshly reproduced |
| Main Table `tab:vit_grouping_summary` | `experiments/vit/`; `figures/data/vit_runs.csv` | `python figures/verify_data.py` checks all 15 printed mean/maximum pairs, independently selected within each row, plus the complete per-seed grid; [one full random-20k run](VIT_FULL_VALIDATION_20261010.md) closely matched its archived best-fit value | Aggregation available; one cell checked, but the table's three-seed family summaries have not been freshly reproduced |
| Appendix Table `tab:function_fitting` | `experiments/dense_fourier/run_dense_fourier_mlp.py`; `figures/data/dense_fourier_d4_runs.csv`, `dense_fourier_d8_runs.csv` | `python figures/verify_fourier_data.py` reconstructs both dimensions using the manuscript's shared-learning-rate selection and checks all 32 printed mean/SD pairs independently of the summary CSVs; [two full PLN[4] cells](FOURIER_FULL_VALIDATION_20261010.md) closely matched archived rows | Historical aggregation available; two cells checked, but the full grid and table statistics have not been freshly reproduced |
| Appendix Table `tab:best_mse_lossweight` | Same PINN source and selected-run CSV as the main PINN table | `python experiments/pinn/verify_data.py` checks all 42 printed cells | Historical selected results checked; two selected extracted full runs completed, with 40 selected cells and the full search pending |
| Appendix Table `tab:pinn_output_ntk` | `experiments/pinn/run_helmholtz.py`, `output_ntk.py`, `data/ntk_matched_240.csv` | `python experiments/pinn/verify_data.py` reconstructs five method medians from 240 paired rows | One full extracted configuration matched archive; remaining grid not validated. Distinct protocol from the PINN error tables |
| Appendix Figures `fig:cnn_grouping`, `fig:cnn_grouping_direction` | CNN runners; `figures/data/cnn_runs.csv` | `python figures/plot_from_csv.py` generates `fig_cnn_d4_grouping_3seed.pdf`, `fig_cnn_grouping_direction.pdf`; [one full random-label PLN[4] cell](CNN_FULL_VALIDATION_20261010.md) was checked separately | Per-seed numeric grid and plotting source included; raw grid logs and a fresh full-grid rerun remain absent |
| Appendix Figure `fig:vgg_results` | `experiments/vgg/model.py`, `train_cifar10.py`, `make_grid_plan.py`, `summarize_grid.py`, `plot_grid.py` | The paper grid can be planned, newly trained, audited and drawn, but no verified historical per-run CSV is included | **Historical figure not reproducible from this candidate**. The new summary/plot path requires all 945 complete runs and an explicit final/best epoch metric; no full grid has been run here |
| Appendix Figures `fig:vit_random6k_grid`, `fig:vit_random20k_grid` | ViT runner, fixed-label preparation, `figures/data/vit_runs.csv` | `python figures/plot_from_csv.py` generates the corresponding `fig_vit_random{6k,20k}_two_modes_3seed.pdf` files; one [random-20k PLN[4] full run](VIT_FULL_VALIDATION_20261010.md) was checked separately | Per-seed numeric grid and plotting source included; no fresh full-grid rerun |
| Appendix Figures `fig:vit_clean_grid`, `fig:vit_grouping_direction_20k` | ViT runner; `figures/data/vit_runs.csv` | The same plotting command generates `fig_vit_clean50k_two_modes_3seed.pdf` and `fig_vit_grouping_direction_20k.pdf` | Per-seed numeric grid and plotting source included; fresh full rerun pending |

The seven generated schematic/CNN/ViT PDFs and manuscript PDFs with matching
basenames were compared separately: their extractable text token sequences
matched exactly, but their SHA-256 digests differed. This supports label/text
correspondence only, not pixel identity or a claim that the source plots are
the precise manuscript files. Generated PDFs are intentionally ignored by Git;
the scripts and compact numerical inputs are the proposed release artifacts.

For protocol-specific limitations, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The author's release decision and remaining verification limits are in the
[release checklist](RELEASE_CHECKLIST.md).
