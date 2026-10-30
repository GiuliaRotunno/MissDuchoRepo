# MissDucho

MissDucho is an extension of [Ducho 2.0](https://github.com/sisinflab/Ducho), a framework for the extraction of multimodal features for recommendation. On top of Ducho's feature extraction, MissDucho adds **imputation of missing modalities**: given item embeddings where some visual/textual modalities are missing, it reconstructs them with a range of imputation methods, so that the imputed embeddings can be used in downstream multimodal recommender systems.

## Table of contents

- [What MissDucho adds](#what-missducho-adds)
- [Imputation methods](#imputation-methods)
- [Installation](#installation)
- [Usage](#usage)
- [Demos](#demos)
- [Data](#data)
- [Third-party code](#third-party-code)
- [Credits and citations](#credits-and-citations)

## What MissDucho adds

- Imputation of missing modalities, starting from a dataset with missing modalities or from a complete dataset with a user-defined percentage of missing modalities.
- An imputation report that includes reconstruction errors.
- Export of imputed embeddings, ready to be used in recommender systems (e.g., Elliot, MMRec).

## Imputation methods

| Family | Methods (folder `ducho/imputation`) |
|---|---|
| Simple baselines | `zeros`, `random`, `mean` |
| Graph-based | `neighmean`, `featprop`, `perspagerank`, `heat` |
| Autoencoders | `autoenc`, `varautoenc`, `graphautoenc`, `vargraphautoenc` |
| Deep learning-based | `vigan`, `smil`, `mmin`, `dicmor` |

The implementations in this repository are my own. The original methods are described in the following works:

- **Graph-based imputation** (`neighmean`, `featprop`, `heat`, `perspagerank`): D. Malitesta, E. Rossi, C. Pomo, T. Di Noia, F. D. Malliaros. *Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation*. IEEE TKDE, 2026 (arXiv:2602.17354).
- **ViGAN**: C. Shang, A. Palmer, J. Sun, K. Chen, J. Lu, J. Bi. *VIGAN: Missing View Imputation with Generative Adversarial Networks*. IEEE BigData 2017.
- **SMIL**: M. Ma, J. Ren, L. Zhao, S. Tulyakov, C. Wu, X. Peng. *SMIL: Multimodal Learning with Severely Missing Modality*. AAAI 2021.
- **MMIN**: J. Zhao, R. Li, Q. Jin. *Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities*. ACL/IJCNLP 2021.
- **DiCMoR**: Y. Wang, Z. Cui, Y. Li. *Distribution-Consistent Modal Recovering for Incomplete Multimodal Learning*. ICCV 2023.

## Installation

```sh
git clone https://github.com/GiuliaRotunno/MissDuchoRepo.git
cd MissDuchoRepo
pip install -r requirements.txt
```

Other options: `requirements_colab.txt` (Google Colab) and `requirements_docker.txt` / `Dockerfile` / `docker-compose.yml` (Docker).

## Usage

```sh
python3 run.py --config=<path_to_config_file> [--additional_argument=value ...]
```

Arguments can override entries of the configuration file using the form `key1.key2.key3`. For details on configuration files, see the [Ducho documentation](https://ducho.readthedocs.io/en/latest/).

Hyperparameter search scripts: `grid_search.py` and `grid_search_crossmodal.py`.

## Demos

- `demos/demo_elliot`: imputation experiments starting from a dataset with missing modalities. The imputed embeddings can then be used in a recommender system (e.g., Elliot) to compute recommendation performance.
- `demos/demo_paper`: imputation experiments starting from a complete dataset, with a user-defined percentage of missing modalities. It produces an imputation report that includes reconstruction errors.


## Data

The following files are **not** included in the repository because of their size:

- `local/`: contains the data and embeddings used by the demos in `demos/`. It is not included in the repository because of its size. 
TODO: decide how to share it.
- `MMRec/MMRec/data/`: one folder per dataset (`Baby`, `Beauty`, `Office`, `Sports`, and the variants `Baby50`, `Baby90`, `Beauty50`, `Beauty90`), each containing the processed files and the embeddings generated for the experiments. They are not included in the repository because of their size. 
TODO: decide how to share them.



## Third-party code

This repository includes modified copies of two external projects, used to run recommendation experiments with the imputed embeddings:

- `Graph-Missing-Modalities`: from [sisinflab/Graph-Missing-Modalities](https://github.com/sisinflab/Graph-Missing-Modalities). Modifications: TODO (briefly describe).
- `MMRec`: from [danielemalitesta/Graph-Missing-Modalities-TKDE](https://github.com/danielemalitesta/Graph-Missing-Modalities-TKDE). Modifications: TODO (briefly describe).

Please refer to the license of each original project. TODO: check the `LICENSE` files inside the two folders.

## Credits and citations

MissDucho is built on top of Ducho, developed by the sisinflab group (Politecnico di Bari). If you use this code, please cite the original Ducho papers:

```bibtex
@inproceedings{DBLP:conf/www/AttimonelliDMPGD24,
  author    = {Matteo Attimonelli and Danilo Danese and Daniele Malitesta and Claudio Pomo and Giuseppe Gassi and Tommaso Di Noia},
  title     = {Ducho 2.0: Towards a More Up-to-Date Unified Framework for the Extraction of Multimodal Features in Recommendation},
  booktitle = {{WWW}},
  doi       = {10.1145/3589335.3651440},
  publisher = {{ACM}},
  year      = {2024}
}

@inproceedings{DBLP:conf/mm/MalitestaGPN23,
  author    = {Daniele Malitesta and Giuseppe Gassi and Claudio Pomo and Tommaso Di Noia},
  title     = {Ducho: {A} Unified Framework for the Extraction of Multimodal Features in Recommendation},
  booktitle = {{ACM} Multimedia},
  pages     = {9668--9671},
  doi       = {10.1145/3581783.3613458},
  publisher = {{ACM}},
  year      = {2023}
}
```