import gradio as gr
import tensorflow as tf

classes = ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Mouth Ulcer", "Tooth Discoloration"]
model = tf.keras.models.load_model("final_model.keras")

def predict(img):
    if img is None:
        return None
    img = tf.image.resize(tf.cast(img, tf.float32), (224, 224))
    probs = model.predict(tf.expand_dims(img, 0), verbose=0)[0]
    return {classes[i]: float(probs[i]) for i in range(len(classes))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Oral Diseases Image Classification",
    description="Educational project. Not a medical diagnosis.",
).launch()