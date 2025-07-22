import tensorflow as tf
from pathlib import Path
from ..config import cfg

def build_callbacks(run_name: str):
    log_dir = Path(cfg.OUTPUT_DIR) / run_name / "tensorboard"
    ckpt_dir = Path(cfg.OUTPUT_DIR) / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "epoch{epoch:02d}-valLoss{val_loss:.4f}.h5"),
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4)
    ]
