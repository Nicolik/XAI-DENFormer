#!/usr/bin/env bash
set -euo pipefail

# Generic local installation helper for the public export.
# For reproducible conda/mamba installation, prefer:
#   mamba env create -f environment.yml
#   conda activate denformer

ENV_NAME="${DENFORMER_ENV_NAME:-denformer}"

mamba create -y -n "$ENV_NAME" python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mamba install -y -c pytorch -c nvidia pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4
mamba install -y -c conda-forge biopython pandas matplotlib scikit-learn numpy h5py tqdm tensorboard country_converter seaborn opencv umap-learn scipy statsmodels openpyxl
mamba install -y -c bioconda cd-hit mafft
python -m pip install transformers performer-pytorch
