"""XGBoost trainer for vessel-routing behaviour cloning.

Reads trajectory JSON files, steps through them using ``EnvGraph`` /
``FullEnvGraph`` to build feature–action pairs, then trains an XGBoost
multi-class classifier (4 cardinal actions: S, N, E, W).

Usage (from repo root)::

    python -m shipping_route_predictor.src.train --grid fine --model_type simple --n_estimators 500
    python -m shipping_route_predictor.src.train --grid coarse --model_type full --n_estimators 500

The trainer relies on the ``EnvGraph.build_input_features()`` /
``FullEnvGraph.build_input_features()`` pipeline for feature extraction,
so feature layout is always in sync with the environment.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import TrainingCallback
import wandb
from tqdm import tqdm
# import debugpy

# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

from shipping_route_predictor.config import (
    GridResolution,
    ModelSpec,
    ModelType,
    TrainingConfig,
    save_config,
)
from shipping_route_predictor.data import AISDataset, EnvDataset
from shipping_route_predictor.env import EnvGraph, build_env

log = logging.getLogger("xgboost.trainer")


# ======================================================================
# XGBoost callback
# ======================================================================

class _ProgressCallback(TrainingCallback):
    """XGBoost after-iteration callback with tqdm + optional WandB."""

    def __init__(
        self,
        n_estimators: int,
        out_dir: Optional[Path] = None,
        checkpoint_interval: int = 50,
        wandb_enabled: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.out_dir = out_dir
        self.checkpoint_interval = checkpoint_interval
        self.wandb_enabled = wandb_enabled
        self.best_val_loss = float("inf")
        self.best_round = 0
        self.pbar = tqdm(total=n_estimators, desc="XGBoost", unit="round")
        self.history: List[Dict[str, Any]] = []

    # XGBoost TrainingCallback interface
    def before_training(self, model):
        return model

    def before_iteration(self, model, epoch, evals_log):
        return False

    def after_iteration(self, model, epoch, evals_log):
        rnd = epoch + 1
        train_loss = val_loss = train_acc = val_acc = None
        metrics: Dict[str, float] = {}

        for ds_name, metric_dict in evals_log.items():
            for m_name, vals in metric_dict.items():
                if not vals:
                    continue
                v = vals[-1]
                metrics[f"{ds_name}/{m_name}"] = v
                if ds_name == "validation_0" and m_name == "mlogloss":
                    train_loss = v
                if ds_name == "validation_0" and m_name == "merror":
                    train_acc = 1.0 - v
                if ds_name == "validation_1" and m_name == "mlogloss":
                    val_loss = v
                    if v < self.best_val_loss:
                        self.best_val_loss = v
                        self.best_round = rnd
                if ds_name == "validation_1" and m_name == "merror":
                    val_acc = 1.0 - v

        if self.pbar:
            self.pbar.update(1)
            pf: Dict[str, str] = {}
            if train_loss is not None:
                pf["t_loss"] = f"{train_loss:.4f}"
            if train_acc is not None:
                pf["t_acc"] = f"{train_acc:.3f}"
            if val_loss is not None:
                pf["v_loss"] = f"{val_loss:.4f}"
            if val_acc is not None:
                pf["v_acc"] = f"{val_acc:.3f}"
            if self.best_round:
                pf["best"] = f"{self.best_val_loss:.4f}@{self.best_round}"
            self.pbar.set_postfix(pf)

        if self.wandb_enabled:
            wandb.log({"round": rnd, **metrics}, step=rnd)

        if (
            self.out_dir
            and self.checkpoint_interval > 0
            and rnd % self.checkpoint_interval == 0
        ):
            ckpt = self.out_dir / f"xgboost_model_r{rnd:04d}.json"
            model.save_model(str(ckpt))
            log.info("Checkpoint saved: %s", ckpt)

        # ---- record metrics ----
        self.history.append({
            "round": rnd,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        return False  # don't stop

    # XGBoost TrainingCallback interface
    def after_training(self, model):
        if self.pbar:
            self.pbar.close()
        return model


# ======================================================================
# Trainer
# ======================================================================

class XGBoostTrainer:
    """Train an XGBoost multi-class (4-action) classifier from AIS trajectories.

    Typical workflow::

        cfg = TrainingConfig(gridResolution=GridResolution.FINE, model=ModelSpec(name="my_xgb", model_type=ModelType.SIMPLE))
        trainer = XGBoostTrainer(cfg)
        trainer.run()
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger("xgboost.trainer")
        self.out_dir = Path(cfg.models_dir) / f"xgboost_{cfg.grid}_{cfg.model.model_type}_{cfg.env.grid_height}x{cfg.env.grid_width}_d{cfg.max_depth}_n{cfg.n_estimators}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env: EnvGraph = build_env(
            EnvDataset(cfg.env, cfg.model.model_type, cfg.weather),
        )
        self.dataset = AISDataset(self.env, cfg.env)
        self.model: Optional[Any] = None
        self._progress_cb: Optional[_ProgressCallback] = None
        self.wandb_enabled = (
            cfg.wandb_mode != "disabled"
        )

    def run(self, checkpoint_interval: int = 50) -> Dict[str, Any]:
        """Full pipeline: load data → train → save."""
        cfg = self.cfg
        env_cfg = cfg.env

        if self.wandb_enabled:
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                mode=cfg.wandb_mode,
                name=self.out_dir.name,
                dir=str(self.out_dir),
                config={
                    "grid": cfg.grid,
                    "model_type": cfg.model.model_type,
                    "n_estimators": cfg.n_estimators,
                    "max_depth": cfg.max_depth,
                    "learning_rate": cfg.learning_rate,
                    "subsample": cfg.subsample,
                    "train_limit": cfg.train_limit,
                    "val_limit": cfg.val_limit,
                },
            )

        save_config(cfg, self.out_dir, filename="config_training.json")
        X_train, y_train = self.dataset.get_ais_dataset(
            "train", limit=cfg.train_limit, desc="train",
        )
        X_val, y_val = self.dataset.get_ais_dataset(
            "val", limit=cfg.val_limit, desc="val",
        )

        # Train
        results = self.train(X_train, y_train, X_val, y_val, checkpoint_interval)

        # Save everything
        model_path = self.save_model()
        self._save_results(results, model_path)

        # WandB artifact + finish
        if self.wandb_enabled:
            art = wandb.Artifact(
                f"xgboost-{cfg.grid}-{cfg.model.model_type}", type="model",
            )
            art.add_file(str(model_path))
            wandb.log_artifact(art)
            wandb.finish()

        self.log.info("=" * 60)
        self.log.info("TRAINING COMPLETE")
        self.log.info("  Output: %s", self.out_dir)
        self.log.info("  Train accuracy: %.4f", results["train_accuracy"])
        if "val_accuracy" in results:
            self.log.info("  Val accuracy:   %.4f", results["val_accuracy"])
        self.log.info("=" * 60)

        return results

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        checkpoint_interval: int = 50,
    ) -> Dict[str, Any]:
        """Train the XGBoost classifier and return result metrics."""
        cfg = self.cfg
        self.log.info("Training XGBoost — n_estimators=%d, max_depth=%d, lr=%.4f",
                       cfg.n_estimators, cfg.max_depth, cfg.learning_rate)

        cb = _ProgressCallback(
            n_estimators=cfg.n_estimators,
            out_dir=self.out_dir,
            checkpoint_interval=checkpoint_interval,
            wandb_enabled=self.wandb_enabled,
        )
        self._progress_cb = cb

        self.model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=4,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            gamma=cfg.gamma,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            eval_metric=cfg.eval_metric,
            early_stopping_rounds=cfg.early_stopping_rounds if X_val is not None else None,
            verbosity=0,
            callbacks=[cb],
            device=cfg.device,
        )

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Metrics
        train_pred = self.model.predict(X_train)
        train_acc = float((train_pred == y_train).mean())
        results: Dict[str, Any] = {
            "train_accuracy": train_acc,
            "train_samples": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "best_iteration": int(
                self.model.best_iteration
                if hasattr(self.model, "best_iteration") and self.model.best_iteration is not None
                else cfg.n_estimators
            ),
        }
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            results["val_accuracy"] = float((val_pred == y_val).mean())
            results["val_samples"] = int(X_val.shape[0])

        self.log.info("Training complete: train_acc=%.4f", results["train_accuracy"])
        if "val_accuracy" in results:
            self.log.info("  val_acc=%.4f", results["val_accuracy"])

        if self.wandb_enabled:
            wandb.log({f"final/{k}": v for k, v in results.items()})

        return results

    def save_model(self, path: Optional[Path] = None) -> Path:
        """Save trained model to JSON (and binary .ubj)."""
        if self.model is None:
            raise ValueError("No model to save — call train() first.")
        path = path or self.out_dir / "xgboost_model_final.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        self.log.info("Model saved: %s", path)

        # Also save compact binary
        if str(path).endswith(".json"):
            ubj = Path(str(path).replace(".json", ".ubj"))
            self.model.save_model(str(ubj))
            self.log.info("Model saved (binary): %s", ubj)
        return path

    def load_model(self, path: Path) -> None:
        """Load a previously-trained model."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        self.log.info("Model loaded: %s", path)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return ``{feature_name: importance}``."""
        if self.model is None:
            raise ValueError("No model available.")
        imp = self.model.feature_importances_
        names = self.env.feature_names
        if len(imp) != len(names):
            names = [f"feature_{i}" for i in range(len(imp))]
        return dict(zip(names, imp.tolist()))
    
    def _save_results(
        self, results: Dict[str, Any], model_path: Path,
    ) -> None:
        """Persist all training artefacts to *self.out_dir*."""
        # Feature importance
        importance = self.get_feature_importance()
        imp_path = self.out_dir / "feature_importance.json"
        with open(imp_path, "w") as f:
            json.dump(importance, f, indent=2)
        self.log.info("Feature importance saved: %s", imp_path)
        top10 = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
        self.log.info("Top-10 features:")
        for name, score in top10:
            self.log.info("  %s: %.4f", name, score)

        if self._progress_cb is not None and self._progress_cb.history:
            history = self._progress_cb.history
            pd.DataFrame(history).to_csv(
                self.out_dir / "iteration_metrics.csv", index=False,
            )
            self.log.info("Iteration metrics saved: %s",
                          self.out_dir / "iteration_metrics.csv")
            self._save_training_curves(history)

        results_path = self.out_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    def _save_training_curves(self, history: List[Dict[str, Any]]) -> None:
        """Generate side-by-side loss & accuracy plots from iteration history."""
        rounds = [r["round"] for r in history]
        train_loss = [r["train_loss"] for r in history]
        val_loss = [r["val_loss"] for r in history]
        train_acc = [r["train_acc"] for r in history]
        val_acc = [r["val_acc"] for r in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(rounds, train_loss, label="Train loss")
        if val_loss[0] is not None:
            ax1.plot(rounds, val_loss, label="Val loss")
        ax1.set_xlabel("Boosting round")
        ax1.set_ylabel("Log-loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(rounds, train_acc, label="Train accuracy")
        if val_acc[0] is not None:
            ax2.plot(rounds, val_acc, label="Val accuracy")
        ax2.set_xlabel("Boosting round")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = self.out_dir / "training_curves.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        self.log.info("Training curves saved: %s", plot_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost vessel-routing model")
    p.add_argument("--grid", type=str, default="fine",
                   choices=["fine", "coarse"])
    p.add_argument("--model_type", type=str, default="simple",
                   choices=["simple", "full"])
    p.add_argument("--wandb_mode", type=str, default="disabled",
                   choices=["disabled", "online", "offline", "dryrun"])
    return p.parse_args()

def main() -> None:
    start = datetime.now()

    def _sighandler(signum, _frame):
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(f"\nTRAINING TERMINATED: {name} (duration {datetime.now() - start})", file=sys.stderr)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _sighandler)
    signal.signal(signal.SIGINT, _sighandler)

    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )

    grid = GridResolution(args.grid)
    model_type = ModelType(args.model_type)

    cfg = TrainingConfig(
        grid=grid,
        model=ModelSpec(
            name=f"xgb_{grid}_{model_type}",
            model_type=model_type,
        ),
        wandb_mode=args.wandb_mode,
    )

    trainer = XGBoostTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
