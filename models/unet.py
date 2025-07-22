import tensorflow as tf
from ..config import cfg
from ..data.preprocessing import build_augmentation

def conv_block(x, filters, dropout_rate):
    x = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Conv2D(filters, 3, activation="relu", padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def encoder_block(x, filters, dropout_rate):
    c = conv_block(x, filters, dropout_rate)
    p = tf.keras.layers.MaxPool2D()(c)
    return c, p

def decoder_block(x, skip, filters, dropout_rate):
    u = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    concat = tf.keras.layers.Concatenate()([u, skip])
    c = conv_block(concat, filters, dropout_rate)
    return c

def build_unet(img_size=cfg.IMG_SIZE + (3,), base_filters=16, depth=4, dropout_factor=0.1):
    inputs = tf.keras.Input(img_size)
    aug    = build_augmentation()(inputs)

    # Encoder
    skips = []
    x = aug
    for d in range(depth):
        f = base_filters * 2 ** d
        c, x = encoder_block(x, f, dropout_factor)
        skips.append(c)

    # Bottleneck
    x = conv_block(x, base_filters * 2 ** depth, dropout_factor * 2)

    # Decoder
    for d in reversed(range(depth)):
        f = base_filters * 2 ** d
        x = decoder_block(x, skips[d], f, dropout_factor)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(x)
    model   = tf.keras.Model(inputs, outputs, name="UNet_modular")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)]
    )
    return model
