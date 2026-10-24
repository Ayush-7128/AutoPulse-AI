"""Grad-CAM Spectral Band Explainer for 2D Mel-Spectrogram Fault Localization."""
import numpy as np
import tensorflow as tf

def compute_spectral_gradcam(model, mel_input, imu_input, conv_layer_name="last_conv"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model({"mel_input": mel_input, "imu_input": imu_input})
        top_class = tf.argmax(predictions[0])
        loss = predictions[:, top_class]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap
