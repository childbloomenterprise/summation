# Pre-registration 2: stitching on robot camera frames (DROID)

Committed before any DROID frame was decoded or embedded. Same method, encoders, metrics and verdict rule as `PREREG.md`, except as listed below.

## Question
Can one rotation, fitted on ≤1,000 robot frames paired with their instructions, find the right episode from a typed instruction better than CLIP-style models? This is the "search robot fleet footage by text" use case.

## Data
- Source: `lerobot/droid_1.0.1` (HF), episodes in data files 0–2 (about the first 3,270 episodes).
- One image per episode: the middle frame of the `exterior_1_left` camera (180×320).
- Texts: up to 3 independent human instructions per episode (`language_instruction`, `_2`, `_3`). Empty ones are dropped, and so are episodes with no instruction or the exact text "No action".
- Split by contiguous block, to reduce same-session leakage:
  - Fit pool: annotated episodes in data files 0–1.
  - Test: annotated episodes in data file 2.
- Fit pairs: N ∈ {100, 300, 1000} episodes from the fit pool, seeds {0, 1, 2}. Each pair uses one random instruction of its episode.
- Unpaired pool for centring/PCA: all fit-pool frames and all fit-pool instructions.
- If fewer than 1,000 annotated fit-pool episodes exist, the largest N becomes "all of them". This is reported as a deviation.

## Encoders and baselines
- Stitched pairs: DINOv2-base + all-mpnet-base-v2 (primary) and DINOv2-small + all-MiniLM-L6-v2.
- Zero-shot baselines: CLIP ViT-B/32 and ViT-L/14 (OpenAI), and SigLIP-B/16 (webli).
- No in-domain CLIP reference exists for this domain.
- Zero-pair: same procedure as in PREREG.md. It uses frames from the first half of the fit pool and instructions from the second half.

## Metrics
- t2i and i2t R@{1, 5, 10}, plus mR. Each instruction queries all test frames; the correct target is its own episode's frame.
- Mutual-10NN between native frame embeddings and first-instruction embeddings, over test episodes.
- Chance level: t2i R@1 = 1 / (number of test episodes).
- Expected difficulty: many episodes share a scene or an instruction ("Put the marker in the cup"). That caps R@1 for every method, so the comparison across methods is what matters.

## Success criteria (primary pair, mR; pass requires mean − std > baseline)
- R1: Stitch at the largest N beats CLIP ViT-B/32.
- R2: Stitch at the largest N beats SigLIP-B/16.
- R3: Stitch N=100 beats CLIP ViT-B/32.
- R5: zero-pair t2i R@10 ≥ 5× chance.

## Verdict rule
- R1 fails → no value for robot-footage search.
- R1 passes, R2 fails → it beats old CLIP only.
- R1 and R2 pass → worth a pilot with a robot-data team.
