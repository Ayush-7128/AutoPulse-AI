"""Quantizes trained Keras Dual-Branch CNN into <150KB Edge TFLite format."""
import tensorflow as tf

def export_quantized_edge_model(keras_path="saved_models/autopulse_best.keras", out_path="saved_models/autopulse_edge.tflite"):
    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    return out_path
