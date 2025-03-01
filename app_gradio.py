#Gradio est parfait pour créer une interface simple permettant aux utilisateurs de télécharger une image et 
#de voir les prédictions d'un modèle de classification.



import gradio as gr
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Charger un modèle pré-entraîné (ResNet50)
model = ResNet50(weights="imagenet")

def classify_image(image):
    image = image.resize((224, 224))  # Redimensionner l'image
    image = np.array(image)
    image = preprocess_input(image)  # Prétraiter l'image
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension de batch
    predictions = model.predict(image)
    return decode_predictions(predictions, top=3)[0]  # Retourner les 3 meilleures prédictions

# Créer une interface Gradio
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Classification d'Images avec ResNet50",
    description="Téléchargez une image pour voir les prédictions du modèle."
)

interface.launch()