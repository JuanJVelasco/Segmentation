import tensorflow as tf

def build_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation_pipeline"
    )
