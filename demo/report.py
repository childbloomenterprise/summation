"""Render a results JSON as Markdown tables: python demo/report.py demo/results.json"""
import json
import sys

import numpy as np

KEYS = ["t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_R@1", "i2t_R@5", "i2t_R@10", "mR"]


def row(name, runs, extra=""):
    runs = runs if isinstance(runs, list) else [runs]
    cells = []
    for k in KEYS:
        v = np.array([r[k] for r in runs]) * 100
        cells.append(f"{v.mean():.1f}" + (f" ±{v.std():.1f}" if len(v) > 1 else ""))
    return f"| {name} | " + " | ".join(cells) + f" | {extra} |"


def main(path):
    r = json.load(open(path))
    print(f"Test: {r['n_test_images']} images, {r['n_test_texts']} texts. Chance t2i R@1 = {100 / r['n_test_images']:.2f}%.\n")
    print("| method | " + " | ".join(KEYS) + " | mutual-10NN / fit |")
    print("|" + "---|" * (len(KEYS) + 2))
    for n, m in r["baselines"].items():
        print(row(n, m, f"{m['mutual_10nn']:.3f}"))
    for label, d in r["stitched"].items():
        for n, runs in d.items():
            if n.isdigit():
                fit = np.mean([x["fit_seconds"] for x in runs])
                print(row(f"stitch {label} N={n}", runs, f"native {d['mutual_10nn_native']:.3f} / {fit:.2f}s"))
    for label, runs in r["zero_pair"].items():
        print(row(f"zero-pair {label}", runs, f"{np.mean([x['fit_seconds'] for x in runs]):.0f}s"))
    for label, d in r["exploratory"].items():
        for n, runs in d.items():
            if n.isdigit():
                print(row(f"exploratory {label} N={n}", runs))


if __name__ == "__main__":
    main(sys.argv[1])
