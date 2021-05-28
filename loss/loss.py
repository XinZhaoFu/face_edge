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


def mix_dice_focal_loss(focal_alpha=0.0001, smooth=1.0, gamma=2.0, alpha=0.25):
    """
    哦 明明已经有现成的dice loss 和focal loss的代码 想直接调用 试了试没成功 那就干脆command c加command v了

    :param focal_alpha:
    :param smooth:
    :param gamma:
    :param alpha:
    :return:
    """
    def mix_dice_focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred) + epsilon

        focal_loss = - alpha_t * tf.math.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)
        focal_loss = tf.reduce_sum(focal_loss)

        y_true_sum = tf.reduce_sum(y_true)
        y_pred_sum = tf.reduce_sum(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)

        dice_loss = 1 - (2.0 * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)

        return 0.5 * focal_loss * focal_alpha + 0.5 * dice_loss
    return mix_dice_focal_loss_fixed


def binary_crossentropy_weight():
    """
    bce的api文档过时了 sample_weight参数已经无效了

    :return:
    """
    def binary_crossentropy_weight_fixed(y_true, y_pred):
        sample_weight = [0.0145, 0.9854]
        bce = tf.keras.losses.BinaryCrossentropy()
        loss = bce(y_true, y_pred, sample_weight)
        return tf.reduce_sum(loss)
    return binary_crossentropy_weight_fixed
