import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
from ..config import cfg

def _read(img_path: Path, is_mask: bool):
    img = imread(str(img_path))[:, :, :3] if not is_mask else imread(str(img_path))
    if is_mask:
        img = rgb2gray(img)
        img = np.expand_dims(img, axis=-1)
    img = resize(img, cfg.IMG_SIZE, mode="constant", preserve_range=True)
    return img.astype(np.float32)

def load_dataset(seed: int = cfg.SEED):
    image_paths = sorted(list(Path(cfg.IMAGES_DIR).glob("*.jpg")))  
    mask_paths  = sorted(list(Path(cfg.MASKS_DIR).glob("*.jpg")))
    assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

    X = np.zeros((len(image_paths), *cfg.IMG_SIZE, 3), dtype=np.float32)
    Y = np.zeros((len(mask_paths),  *cfg.IMG_SIZE, 1), dtype=np.float32)

    print("[data] Resizing images and masks →", cfg.IMG_SIZE)
    for i, (im_p, m_p) in tqdm(enumerate(zip(image_paths, mask_paths)), total=len(image_paths)):
        X[i] = _read(im_p, False) / 255.0   
        Y[i] = _read(m_p, True)  / 255.0     

    return train_test_split(X, Y, test_size=0.1, random_state=seed)

def build_tf_dataset(X, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=cfg.SEED)
    ds = ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
