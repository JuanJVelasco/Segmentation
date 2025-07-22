from pathlib import Path
import os

class _Cfg:
    _base_dir = Path(os.environ.get("_SEG_DATA_DIR", "../Data/SEG")).resolve()

    # Data
    IMAGES_DIR: Path = _base_dir / "images"
    MASKS_DIR: Path  = _base_dir / "masks"

    # Training hyper‑parameters
    IMG_SIZE  = (256, 256)
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 50))
    LR         = float(os.environ.get("LR", 1e-3))

    # Reproducibility
    SEED = 42

    # Derived / runtime
    OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", "outputs")).resolve()

    def as_dict(self):
        return {k: v for k, v in self.__class__.__dict__.items() if k.isupper()}

cfg = _Cfg()
