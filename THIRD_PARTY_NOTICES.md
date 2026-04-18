# Third-Party Notices

This repository contains code that was adapted from public upstream projects.
The notes below are intended to preserve provenance and make the current licensing
position easier to understand.

## Repository License

This repository is distributed under the GNU General Public License, version 3.
See [LICENSE](LICENSE).

## Upstream Projects

### gevaertlab/sequoia-pub

- Upstream repository: <https://github.com/gevaertlab/sequoia-pub>
- Upstream repository license: MIT
- Relevant current files include training, data-loading, and backbone code derived or adapted from that repository, including the retained ViS/SEQUOIA-style path and related training glue in this trimmed subset.

The upstream `sequoia-pub` repository ships an MIT `LICENSE` at the repository root.
That upstream attribution should be preserved when redistributing adapted code.

### owkin/HE2RNA_code

- Upstream repository: <https://github.com/owkin/HE2RNA_code>
- Upstream repository license: GNU GPL v3 or later
- The current `he2rna` baseline path in this repository is adapted through `gevaertlab/sequoia-pub/src/he2rna.py`.

The public `sequoia-pub/src/he2rna.py` file retains the Owkin copyright notice and GNU GPL notice.
Accordingly, the `he2rna` path in this repository should be treated as GPL-governed code.

## Practical Mapping

- `train/backbone/he2rna_backbone.py`: HE2RNA-style aggregation path adapted through `sequoia-pub/src/he2rna.py`
- `train/backbone/tformer_lin.py`: ViS / SEQUOIA-style transformer path adapted from `sequoia-pub`
- `train/read_data.py`, `train/training.py`, `train/utils.py`, `train/main.py`, `run_train.sh`: training and data-handling code adapted from the same project family and trimmed for this repository

## Notes

- This notice is not legal advice.
- If you need a version without the retained `he2rna` path, remove the `he2rna` backbone and its related entry points before redistributing under a different licensing strategy.
