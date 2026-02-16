#!/bin/bash
# Download all datasets for HIDE experiments.
# Most datasets are downloaded on-demand via HuggingFace.
# This script pre-downloads CIFAR-100 via torchvision.

set -e

echo "Downloading CIFAR-100..."
python -c "import torchvision; torchvision.datasets.CIFAR100(root='data/cifar100', download=True)"

echo "Pre-caching bAbI..."
python -c "from datasets import load_dataset; load_dataset('Muennighoff/babi', 'en-10k')"

echo "Pre-caching TempLAMA..."
python -c "from datasets import load_dataset; load_dataset('Yova/templama')"

echo "All datasets ready."
