"""Run the A–F objective ablation (DESIGN.md §8.3) and print a comparison table.

    python scripts/run_ablation.py                      # all variants, base config
    python scripts/run_ablation.py --variants A D E --set train.steps=500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from homefm.train.pretrain import ROOT, load_config, train

VARIANTS = {
    "A": "A_contrastive.yaml",
    "B": "B_masked_recon.yaml",
    "C": "C_latent_mask.yaml",
    "D": "D_next_event.yaml",
    "E": "E_next_event_latent.yaml",
    "F": "F_full.yaml",
}
COLUMNS = ["probe_weighted_f1", "count_exact", "count_mae", "anomaly_auroc_rarity", "anomaly_auroc_surprise",
           "train_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    rows = {}
    for v in args.variants:
        cfg = load_config(str(ROOT / "configs" / "objectives" / VARIANTS[v]), args.set)
        rows[v] = train(cfg)

    print("\n" + " | ".join(["variant"] + COLUMNS))
    for v, r in rows.items():
        print(" | ".join([v] + [f"{r[c]:.3f}" if c in r else "-" for c in COLUMNS]))
    out = ROOT / "runs" / "ablation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
