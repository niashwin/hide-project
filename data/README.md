# Data Sources

All datasets are publicly available. No proprietary or synthetic data were used.

## Included

- `drm_word_lists.json` — 24 DRM word lists (Roediger & McDermott, 1995). Public domain.

## Downloaded at Runtime

| Dataset | Source | License |
|---------|--------|---------|
| bAbI QA en-10k | HuggingFace: `Muennighoff/babi` | BSD |
| TempLAMA | HuggingFace: `Yova/templama` | MIT |
| CIFAR-100 | `torchvision.datasets.CIFAR100` | BSD |
| COCO Captions 2017 | HuggingFace: `jxie/coco_captions` | CC BY 4.0 |
| Flickr30k | HuggingFace: `lmms-lab/flickr30k` | Research |
| Wikipedia 20220301.en | HuggingFace: `wikipedia` (streaming) | CC BY-SA |

## Download Script

```bash
bash download_data.sh
```
