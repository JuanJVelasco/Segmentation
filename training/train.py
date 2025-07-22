import argparse
import datetime as dt
from pathlib import Path
import tensorflow as tf
from ..config import cfg
from ..data.loader import load_dataset, build_tf_dataset
from ..models.unet import build_unet
from .callbacks import build_callbacks

def parse_args():
    p = argparse.ArgumentParser(description="Train U‑Net on SEG.")
    p.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    return p.parse_args()

def main():
    args = parse_args()
    run_name = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    print(f"[train] Run: {run_name}")

    X_train, X_val, y_train, y_val = load_dataset()
    ds_train = build_tf_dataset(X_train, y_train, training=True)
    ds_val   = build_tf_dataset(X_val,  y_val,  training=False)

    model = build_unet()
    model.summary()

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        callbacks=build_callbacks(run_name),
        verbose=2
    )

    out_model = Path(cfg.OUTPUT_DIR) / run_name / "model_final.h5"
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_model)
    print(f"[train] Model saved to {out_model}")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()
