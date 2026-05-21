# Adaptive Bead-Pull Diagram Assets

This directory contains tracked, reproducible PDF slides for `spider-bead-pull`. The deck is split into:

- technical/reference slides for repo structure and artifact navigation
- physics/presentation slides for measurement geometry, reconstruction physics, and configuration tradeoffs

The diagrams are generated from current repo state, not hand-edited exports. The build step reads:

- notebook headings and orchestration stages from `adaptive_bead_pull_variations.ipynb`
- defaults and workflow helpers from `src/adaptive_bead_pull.py`
- current artifact schemas and counts from `data/adaptive_bead_pull_variations/`
- current figure layout from `figs/`

## Files

Technical / reference:

- `01_repo_purpose_and_inputs.pdf`
- `02_notebook_execution_flow.pdf`
- `03_single_configuration_reconstruction.pdf`
- `04_configuration_space.pdf`
- `05_figs_folder_guide.pdf`
- `06_data_folder_guide.pdf`

Physics / presentation:

- `07_configuration_geometry_and_sampling.pdf`
- `08_gap_and_z_coverage.pdf`
- `09_bead_pull_physics_chain.pdf`
- `10_mode_model_and_fit_parameters.pdf`
- `11_complex_ET_reconstruction.pdf`
- `12_configuration_tradeoffs_for_physics.pdf`

## Regenerate

Run:

```bash
python assets/build_diagrams.py
```

The renderer validates the current repo layout before writing PDFs. If notebook stages, artifact schemas, expected counts, or representative configuration selection drift, it exits with an error so the slides can be updated intentionally.

All outputs are rendered as 16:9 slide-sized PDFs (`960 x 540 pt`) so they can be dropped into presentation software without the old panoramic or portrait stretching.

## Editable Sources

- `sources/*.dot`: Graphviz templates for structural diagrams
- `sources/theme.json`: shared visual palette and typography
- `sources/diagram_text.json`: shared titles and explanatory captions
- `sources/physics_text.json`: physics slide titles, captions, and formulas
- `sources/folder_labels.json`: figure and data artifact descriptions used in the guides
