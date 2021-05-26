import tensorflow as tf


def dice_loss(smooth=1.0):
    """
    适用于二分类问题的dice loss
    dice_loss = 1 - (2 * reduce_sum(X * Y) + smooth) / (reduce_sum(X) + reduce_sum(Y) + smooth)

    :param smooth:
    :return:
    """
    def dice_loss_fixed(y_true, y_pred):
        """
        模型最后的激活函数需要为sigmoid

        :param y_true:
        :param y_pred:
        :return:
        """
        y_true_sum = tf.reduce_sum(y_true)
        y_pred_sum = tf.reduce_sum(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)

        loss = 1 - (2.0 * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
        return loss

    return dice_loss_fixed


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    :param gamma:
    :param alpha:
    :return:
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        模型最后的激活函数需要为sigmoid

        :param y_true:
        :param y_pred:
        :return:
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred) + \
              tf.keras.backend.epsilon()
        loss = - alpha_t * tf.math.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        return tf.reduce_sum(loss)

    return binary_focal_loss_fixed
