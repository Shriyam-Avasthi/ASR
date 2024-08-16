import tensorflow as tf
import tensorflow.keras.metrics as tfkm
from tensorflow.keras import backend as K


MASKED_VALUE = 0


def get_1d_mask(y_true, masked_value):
    """Get 1D mask by comparing y_true with masked_value.
    By using `reduce_any`, it masks any item that has more than one y_true that is equal to `MASKED_VALUE`.
    """
    mask_2d = K.not_equal(y_true, masked_value)  # (batch, n_class)
    return K.cast_to_floatx(tf.math.reduce_any(mask_2d, axis=1))  # (batch, )


# losses
def masked_binary_crossentropy(y_true, y_pred):
    mask = K.cast_to_floatx(K.not_equal(y_true, MASKED_VALUE))

    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_categorical_crossentropy(y_true, y_pred):
    mask = K.cast_to_floatx(K.not_equal(y_true, MASKED_VALUE))

    return K.categorical_crossnetropy(y_true * mask, y_pred * mask)


# metrics - when there are parent classes
class MaskedRecall(tfkm.Recall):
    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = get_1d_mask(y_true, MASKED_VALUE)

        return super().update_state(
            tf.boolean_mask(y_true, y_mask),
            tf.boolean_mask(y_pred, mask),
            sample_weight,
        )


class MaskedPrecision(tfkm.Precision):
    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = get_1d_mask(y_true, MASKED_VALUE)

        return super().update_state(
            tf.boolean_mask(y_true, y_mask),
            tf.boolean_mask(y_pred, mask),
            sample_weight,
        )


class MaskedAUC(tfkm.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = get_1d_mask(y_true, MASKED_VALUE)

        return super().update_state(
            tf.boolean_mask(y_true, y_mask),
            tf.boolean_mask(y_pred, mask),
            sample_weight,
        )


# Customized metric
class MaskedCategoricalAccuracy(tfkm.Metric):
    def __init__(self, name="masked_categorical_accuracy", **kwargs):
        super(MaskedCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.n_corrects = self.add_weight(name="n_corrects", initializer="zeros")
        self.n_items = self.add_weight(name="n_items", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Note: this implementation ignores sample_weight
        mask = get_1d_mask(y_true, MASKED_VALUE)  # (batch, n_class)

        y_true = tf.boolean_mask(y_true, mask)  # (n_items, n_class)
        y_pred = tf.boolean_mask(y_pred, mask)

        n_item = K.int_shape(y_true)[0]
        if n_item in (0, None):
            return

        if_correct = K.equal(K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1))
        self.n_items.assign_add(K.cast_to_float(n_item))
        self.n_corrects.assign_add(K.sum(K.cast_to_floatx(if_correct)))

    def result(self):
        if self.n_items == 0.0:
            return 0.0

        return self.n_corrects / self.n_items

    def reset_states(self):
        self.n_corrects.assign(0.0)
        self.n_items.assign(0.0)