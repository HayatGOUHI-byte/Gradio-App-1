import gradio as gr
from transformers import pipeline

# Charger un modèle de génération de texte (GPT-2)
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt):
    output = generator(prompt, max_length=50, num_return_sequences=1)
    return output[0]["generated_text"]

# Créer une interface Gradio
interface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Entrez votre texte ici..."),
    outputs=gr.Textbox(lines=5, label="Texte Généré"),
    title="Générateur de Texte avec GPT-2",
    description="Entrez un texte et voyez ce que GPT-2 génère !"
)

interface.launch()