[![arXiv](https://img.shields.io/badge/arXiv-2605.18667-b31b1b.svg)](https://arxiv.org/abs/2605.18667)
[![License](https://img.shields.io/github/license/vdplasthijs/better_together.svg)](LICENSE)

# Better Together

Code and data to reproduce all results from the paper [Better Together: Evaluating the Complementarity of Earth Embedding Models](https://arxiv.org/abs/2605.18667) from Thijs L van der Plas, Jacob JW Bakermans, Vishal Nedungadi, Gabrielė Tijūnaitytė, Marc Rußwurm, Ioannis N Athanasiadis. 

## Instructions:
- All figures and analyses are created in the notebooks (which use the modules). See below for details. 
- Data is included in the `data/` folder. **The data is provided with a CC-BY-SA license.**
- There are 2 zip files that should be unzipped in `data/geoclip/`
- The python packages are listed in the `.yml` files (one with and one without builds).
- Some functions might require a GEE API key, but this can otherwise be disabled for most functionalities.

## Guide to notebooks (that generate figures/tables):
- Explain embeddings: contains most analyses.
- Explore downstream tasks: contains 1 supp fig.
- Sample locations dataset: contains supp figs regarding data sampling.
- Spatial relationship task: contains analyses regarding spatial scale.
- Task agnostic overlap: contains CCA/CKA analysis.

## Citation:
```
@misc{vanderplas2026bettertogetherevaluatingcomplementarity,
      title={Better Together: Evaluating the Complementarity of Earth Embedding Models}, 
      author={Thijs L van der Plas and Jacob JW Bakermans and Vishal Nedungadi and Gabrielė Tijūnaitytė and Marc Rußwurm and Ioannis N Athanasiadis},
      year={2026},
      eprint={2605.18667},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.18667}, 
}
```
