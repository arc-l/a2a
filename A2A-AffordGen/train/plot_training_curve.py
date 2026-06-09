#!/usr/bin/env python3
"""
Plot train/val loss and learning rate curves from a Swift logging.jsonl.

Usage:
    python plot_training_curve.py \
        data/affordance/segagent_ckpt_0511/v13-20260511-130834/logging.jsonl

Output: loss_curve.png (and optionally token_acc_curve.png) saved next to the jsonl.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_logs(jsonl_path: str):
    train, val = [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            step_str = entry.get("global_step/max_steps", "")
            step = int(step_str.split("/")[0]) if "/" in str(step_str) else None
            epoch = entry.get("epoch")

            if "eval_loss" in entry:
                val.append({
                    "step": step,
                    "epoch": epoch,
                    "eval_loss": entry["eval_loss"],
                    "eval_token_acc": entry.get("eval_token_acc"),
                })
            elif "loss" in entry:
                train.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": entry["loss"],
                    "lr": entry.get("learning_rate"),
                    "token_acc": entry.get("token_acc"),
                    "grad_norm": entry.get("grad_norm"),
                })
    return train, val


def make_curves(train, val, out_dir: Path):
    t_steps  = [e["step"]  for e in train if e["step"] is not None]
    t_loss   = [e["loss"]  for e in train if e["step"] is not None]
    t_acc    = [e["token_acc"] for e in train if e["step"] is not None and e["token_acc"] is not None]
    t_lr     = [e["lr"]    for e in train if e["step"] is not None and e["lr"] is not None]
    t_gnorm  = [e["grad_norm"] for e in train if e["step"] is not None and e["grad_norm"] is not None]

    v_steps  = [e["step"]      for e in val if e["step"] is not None]
    v_loss   = [e["eval_loss"] for e in val if e["step"] is not None]
    v_acc    = [e["eval_token_acc"] for e in val if e["step"] is not None and e["eval_token_acc"] is not None]

    # ── Figure 1: Train Loss ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_steps, t_loss, color="#4a90d9", linewidth=0.9, alpha=0.75, label="train loss")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title("Train Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    out_train = out_dir / "train_loss_curve.png"
    fig.savefig(out_train, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_train}")

    # ── Figure 2: Val Loss ────────────────────────────────────────────────────
    if v_steps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Validation Metrics", fontsize=12)

        ax = axes[0]
        ax.plot(v_steps, v_loss, color="#e05c30", linewidth=2.0, marker="o", markersize=6)
        best_idx = v_loss.index(min(v_loss))
        ax.scatter([v_steps[best_idx]], [v_loss[best_idx]], s=120, c="#e05c30",
                   edgecolors="black", linewidths=1.2, zorder=5,
                   label=f"best={v_loss[best_idx]:.4f} @step {v_steps[best_idx]}")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.set_title("Val Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        ax = axes[1]
        if v_acc:
            ax.plot(v_steps[:len(v_acc)], v_acc, color="#e05c30", linewidth=2.0,
                    marker="o", markersize=6)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Token Accuracy")
        ax.set_title("Val Token Accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.35)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        plt.tight_layout()
        out_val = out_dir / "val_loss_curve.png"
        fig.savefig(out_val, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out_val}")

    # ── Figure 3: Train Token Acc ─────────────────────────────────────────────
    if t_acc:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_steps[:len(t_acc)], t_acc, color="#4a90d9", linewidth=0.9,
                alpha=0.75, label="train token acc")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Token Accuracy")
        ax.set_title("Train Token Accuracy")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        plt.tight_layout()
        out_tacc = out_dir / "train_token_acc_curve.png"
        fig.savefig(out_tacc, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out_tacc}")

    out1 = out_dir / "train_loss_curve.png"  # for backward compat reference

    # ── Figure 2: LR + Grad Norm ───────────────────────────────────────────────
    if t_lr or t_gnorm:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle("LR & Grad Norm", fontsize=13)

        ax = axes[0]
        if t_lr:
            ax.plot(t_steps[:len(t_lr)], t_lr, color="#6a0dad", linewidth=1.0)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.35)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        ax = axes[1]
        if t_gnorm:
            ax.plot(t_steps[:len(t_gnorm)], t_gnorm, color="#2a9d8f", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Grad Norm")
        ax.set_title("Gradient Norm")
        ax.grid(True, alpha=0.35)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        plt.tight_layout()
        out2 = out_dir / "lr_gradnorm_curve.png"
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {out2}")


def print_summary(train, val):
    if train:
        last = train[-1]
        print(f"  Train steps logged : {len(train)}")
        print(f"  Last step          : {last['step']}  loss={last['loss']:.4f}  "
              f"epoch={last['epoch']:.3f}")
    if val:
        best = min(val, key=lambda e: e["eval_loss"])
        print(f"  Val checkpoints    : {len(val)}")
        print(f"  Best val loss      : {best['eval_loss']:.4f}  @step {best['step']}  "
              f"epoch={best['epoch']:.3f}")
        if best["eval_token_acc"] is not None:
            print(f"  Best val token acc : {best['eval_token_acc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="path to logging.jsonl")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_dir = jsonl_path.parent

    print(f"Reading: {jsonl_path}")
    train, val = load_logs(str(jsonl_path))
    print(f"Parsed {len(train)} train entries, {len(val)} val entries")
    print_summary(train, val)

    make_curves(train, val, out_dir)


if __name__ == "__main__":
    main()
