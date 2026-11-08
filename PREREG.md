# Pre-registration: few-pair encoder stitching on RSICD

Written and committed before any embedding, fitting or evaluation was run. Any later deviation is listed under "Deviations" in RESULTS.md.

## Question
Can one orthogonal rotation, fitted on 100–1,000 image–caption pairs, connect a frozen vision-only encoder to a frozen text encoder well enough to beat off-the-shelf CLIP-style models on remote-sensing text↔image retrieval?

## Data
- RSICD (HF `arampacha/rsicd`, official splits). Train: 8,734 images. Valid: 1,094 (not used). Test: 1,093 images × 5 captions = 5,465 captions.
- The test split is used only for the final evaluation. No choices are made on it.
- Fit pairs: N ∈ {100, 300, 1000} train images, sampled with seeds {0, 1, 2}. Each pair is (image, one of its 5 captions chosen at random).
- Unpaired pool (label-free; used only for centring/PCA): all 8,734 train images and all 43,670 train captions. No pairing information is used. Assumption: users have unpaired corpora of each modality.

## Method (`summation` rotation)
1. L2-normalise native embeddings.
2. For each modality, centre with the pool mean and project with pool PCA to k dims, then L2-normalise again.
3. Orthogonal Procrustes on the N pairs: R = argmin ||X R − Y||, with R orthogonal (closed form, SVD).
4. Retrieval uses cosine similarity in the shared k-dim space.
- k is chosen automatically from {16, 32, 64, 128, 256} (k ≤ 0.8 N) by 5-fold cross-validation on the N fit pairs only. The selection score is (t2i R@1 + t2i R@5) / 2.
- Zero-pair variant: Gromov–Wasserstein coupling (POT) as the initialisation, then Wasserstein-Procrustes refinement. It uses 1,000 images from train[0:4367] and 1,000 captions from images in train[4367:]. The scenes are disjoint, so there is no hidden bijection. k = 64, fixed.

## Encoders
- Primary stitched pair: image `facebook/dinov2-base` (CLS/pooler), text `sentence-transformers/all-mpnet-base-v2`.
- Secondary stitched pair: `facebook/dinov2-small` + `sentence-transformers/all-MiniLM-L6-v2`.
- Known contamination: the training data for mpnet and MiniLM includes COCO captions. RSICD is not known to be included.

## Baselines (zero-shot, native embeddings)
- CLIP ViT-B/32 (OpenAI).
- CLIP ViT-L/14 (OpenAI).
- SigLIP base-patch16-224 (`google/siglip-base-patch16-224`). This is the off-the-shelf multimodal embedding. Paid APIs (Voyage, Cohere, Titan) are not run: there is no key.
- In-domain reference: RemoteCLIP ViT-B/32. It was trained on data that includes the RSICD train split, so it is a ceiling, not a zero-shot baseline.
- Exploratory (not a criterion): CLIP ViT-B/32 with the same rotation fitted on top (N = 1000).

## Metrics (test split)
- t2i R@{1, 5, 10}: each of the 5,465 captions queries 1,093 images.
- i2t R@{1, 5, 10}: each image queries 5,465 captions; a hit is any of its own 5 captions in the top k.
- mR = mean of the six recalls (the RSICD convention).
- Mutual-10NN alignment between the native image embeddings and the first-caption embeddings over the 1,093 test pairs. This is a diagnostic of the encoder pair.
- Chance level: t2i R@1 = 0.0009, R@10 = 0.0091.
- Stitched results: mean ± std over 3 seeds. Fit wall time on CPU is reported.

## Success criteria (primary pair, mR)
- P1 (primary): Stitch N=1000 beats CLIP ViT-B/32. Pass requires mean − std > baseline.
- P2: Stitch N=1000 beats SigLIP-base (same rule).
- P3: Stitch N=100 beats CLIP ViT-B/32 (same rule).
- P4 (stretch): Stitch N=1000 ≥ RemoteCLIP-B/32 − 5 mR points.
- P5 (zero-pair): t2i R@10 ≥ 5× chance (≥ 0.046).

## Verdict rule (fixed now)
- P1 fails → the tool has no value in this domain.
- P1 passes, P2 fails → it beats old CLIP only; not worth a product.
- P1 and P2 pass → worth a pilot in domains that have no domain-specific CLIP.
