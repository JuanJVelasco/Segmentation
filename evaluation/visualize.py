import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ..config import cfg
from ..data.loader import load_dataset
from ..models.unet import build_unet

def plot_training_curves(history):
    tr_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    tr_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(tr_acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    ax[0].plot(epochs, tr_loss, label="train")
    ax[0].plot(epochs, val_loss, label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(epochs, tr_acc, label="train")
    ax[1].plot(epochs, val_acc, label="val")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

def qualitative_demo(model=None):
    X_train, X_val, y_train, y_val = load_dataset()
    if model is None:
        model = build_unet()
        model.load_weights(sorted(Path(cfg.OUTPUT_DIR).glob("**/*.h5"))[-1])
    idx = random.randint(0, len(X_val)-1)
    img, gt = X_val[idx], y_val[idx]
    pred = model.predict(np.expand_dims(img,0))[0]

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[1].imshow(gt.squeeze(), cmap="gray")
    ax[1].set_title("Ground truth")
    ax[2].imshow(pred.squeeze(), cmap="gray")
    ax[2].set_title("Prediction")
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    qualitative_demo()
