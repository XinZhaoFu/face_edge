import tensorflow as tf


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0)
        y_true = tf.cast(y_true, tf.float32)

        loss = -y_true * tf.math.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        loss = tf.reduce_sum(loss, axis=1)
        return loss

    return focal_loss_fixed


def dice_loss(smooth=1.0):
    def dice_loss_fixed(y_true, y_pred):
        y_true_sum = tf.reduce_sum(y_true)
        y_pred_sum = tf.reduce_sum(y_pred)
        intersection = tf.reduce_sum(y_true*y_pred)

        loss = 1 - (2.0 * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
        return loss

    return dice_loss_fixed
