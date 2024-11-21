import tensorflow as tf

# Dice Loss
@tf.function
def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3])
    return tf.keras.backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

@tf.function
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)