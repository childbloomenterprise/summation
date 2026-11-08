# summation

Connect a frozen encoder for one modality (e.g. a vision-only DINOv2) to a frozen encoder for another (e.g. a sentence-transformer). It fits **one orthogonal rotation** on ~100–1,000 pairs, or on no pairs at all, and runs on CPU in seconds.

Basis: Balestriero & LeCun, *Proving and Improving the Platonic Representation Hypothesis with JEPAs* (arXiv:2610.39302). Prior art for the same mechanics: Latent Space Translation (Maiorca et al., NeurIPS 2023), ASIF (Norelli et al., NeurIPS 2023), GeoWarP (arXiv:2610.09411).

## Install
```bash
pip install -e ".[encoders,unpaired]"      # core needs only numpy + scipy
```

## CLI
```bash
# 1. embed with any frozen encoders (hf-image:<id> | st:<id> | clip:<arch>:<pretrained>)
summation embed --encoder hf-image:facebook/dinov2-base --images pairs_images.txt --out x.npy
summation embed --encoder st:sentence-transformers/all-mpnet-base-v2 --texts pairs_texts.txt --out y.npy
#    optional unpaired corpora (improve centring/PCA; no labels needed)
summation embed --encoder hf-image:facebook/dinov2-base --images all_images/ --out x_pool.npy
summation embed --encoder st:sentence-transformers/all-mpnet-base-v2 --texts all_texts.txt --out y_pool.npy

# 2. fit one rotation (rows of x.npy and y.npy are pairs; k chosen by 5-fold CV on the pairs)
summation fit --x x.npy --y y.npy --x-pool x_pool.npy --y-pool y_pool.npy --out map.npz
#    or with no pairs at all
summation fit --unpaired --x x_pool.npy --y y_pool.npy --out map_unpaired.npz

# 3. search an image index by text (query encoder is read from map.npz)
summation search --map map.npz --index x_pool.npy --query "a stadium surrounded by parking lots" -k 5

# 4. evaluate on your own held-out pairs
summation eval --map map.npz --x test_x.npy --y test_y.npy
```

## Python
```python
from summation import Stitch
s = Stitch().fit(img_pairs, txt_pairs, x_pool=img_corpus, y_pool=txt_corpus)
scores = s.transform_y(txt_queries) @ s.transform_x(img_index).T
```

## Demo and results
See `PREREG.md` (committed before any run) and `RESULTS.md`. Reproduce:
```bash
pip install -e ".[encoders,unpaired,demo]"
python demo/rsicd.py extract && python demo/rsicd.py evaluate
```

## Results

See [RESULTS.md](RESULTS.md). Short version: with 1,000 pairs, stitched DINOv2+MPNet beats CLIP and SigLIP on robot camera frames (DROID, mR 14.4 vs 8.0). It loses to SigLIP on satellite images (RSICD, 18.2 vs 22.0). It loses to CLIP at 100 pairs in both. Pair-free mode does not work on either.
