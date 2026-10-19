# Text-to-Video Retrieval with CLIP

Extending CLIP to video retrieval with a lightweight adapter, trained on MSR-VTT. The CLIP encoders stay frozen; only a small bottleneck MLP is trained.

**Result:** R@1 of **42.45%** on the MSR-VTT validation set, up from a **38.63%** zero-shot baseline, with negligible inference overhead. R@1 of **33.40%** on the 1k-A test set.

---

## The problem

Text-to-video retrieval means finding the right video from a database given a natural-language query. CLIP aligns images and text well, but it has no notion of time: it sees individual frames, not motion. The question this project asks is how far you can get by adding the smallest possible amount of machinery on top of a frozen CLIP, rather than training a heavy temporal model.

## Approach

```
video → uniform frame sampling (8 frames) → frozen CLIP ViT-B/32 image encoder
      → residual adapter (512 → 64 → 512, GELU) → mean pooling → L2 norm
                                                                      ↓
text query → frozen CLIP text encoder ─────────────────────────→ cosine similarity
```

The adapter applies a small residual correction to each frame embedding: `v' = v + F(v)`, where `F` is a two-layer bottleneck MLP. Frames are then mean-pooled into one video embedding. Training uses a symmetric InfoNCE contrastive loss with in-batch negatives. Only the adapter is updated — every CLIP parameter stays frozen.

## What the ablations showed

Each design choice was tested on the validation set before being fixed.

**Multiple frames matter a lot.** A single center frame gives R@1 of 26.76%. Eight frames with mean pooling gives 37.83%. Adding per-frame L2 normalization before aggregation gives 38.63%.

**Eight frames is enough.** Going from 4 to 8 frames improves R@1 from 36.22% to 38.63%. Going to 12 or 16 does not improve it further — performance saturates.

**Uniform sampling beats the clever alternatives.** Random (33.80%), Gaussian (34.81%), and motion-based sampling (34.00%) all underperform plain uniform sampling (38.63%). Motion-based sampling seemed like it should help and didn't.

**Mean pooling beats max pooling and attention.** Max pooling collapses to 24.14%. A parameter-free similarity-based self-attention pooling reaches 37.83%, slightly below mean pooling, at higher cost.

**Adapting features beats weighting them.** A learned frame-weighting head improves R@1 only marginally (39.03%). The adapter reaches 42.45% and drops median rank from 3 to 2 — adapting the representations before aggregation is more effective than learning how to combine them.

| Method | R@1 | R@5 | R@10 | MedR |
|---|---|---|---|---|
| Zero-shot, mean pooling | 38.63 | 62.98 | 72.03 | 3 |
| Learned frame weighting | 39.03 | 63.58 | 73.24 | 3 |
| **Adapter + mean pooling** | **42.45** | **69.21** | **78.27** | **2** |

*Validation set. Final configuration evaluated once on the 1k-A test set: R@1 33.40, R@5 59.80, R@10 69.70.*

## Where this sits against the literature

| Method | Year | R@1 (test) |
|---|---|---|
| X-CLIP (ViT-B/16) | 2022 | 49.3 |
| CLIP4Clip (sequential Transformer) | 2021 | 44.5 |
| CLIP4Clip (mean pooling) | 2021 | 42.1 |
| VideoCLIP | 2021 | 30.9 |
| **This work** | — | **33.40** |

The gap to CLIP4Clip and X-CLIP is real, and it comes from what those models add: sequential transformers, cross-modal attention, multi-grained contrastive objectives. This project deliberately doesn't add them. The point is how much of the performance is reachable with a frozen backbone, one small adapter, and no temporal modeling at all.

## Limitations

Evaluated on MSR-VTT only. Compute constraints limited hyperparameter search and ruled out deeper adapter configurations. No explicit temporal modeling means motion and action ordering are not captured — mean pooling discards frame order entirely, which is the most likely source of the remaining gap.

## Details

- **Backbone:** CLIP ViT-B/32, frozen, 512-dimensional embeddings
- **Dataset:** MSR-VTT, ~6,000 train / 497 validation / 1k-A test split
- **Adapter:** bottleneck dimension 64, learning rate 1e-4
- **Loss:** symmetric InfoNCE, in-batch negatives
- **Metrics:** R@1/5/10, median rank, mean rank, retrieval latency

Full write-up with derivations and complete ablation tables: [`Computer_Vision_Text-to-Video_Retrieval_System.pdf`](./Computer_Vision_Text-to-Video%20Retrieval%20Sy....pdf)

## Authors

Sarah Hassouneh and Sondos Qasarwa — Department of Electrical & Computer Engineering, Birzeit University, 2026. Instructor: Mr. Aziz Qaroush.
