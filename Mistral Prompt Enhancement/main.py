import gradio as gr
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import spacy

#loading environment variables - hugging face api key
load_dotenv(find_dotenv())

#spaCy model for nlp processing
nlp = spacy.load("en_core_web_sm")

def nlp_processing(text):
    doc = nlp(text) 
    
    filtered_tokens = [token.text for token in doc if not token.is_stop]

    return " ".join(filtered_tokens)


def enhanced_prompt(user_prompt):
    try:
        processed_prompt = nlp_processing(user_prompt)

        prompt_enhancer = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
        
        final_prompt = prompt_enhancer("give a short but descriptive enough enhanced prompt for the following prompt for image generation:" + processed_prompt)
        
        return final_prompt[0]['generated_text']
    
    except Exception as e:
        return e


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Generator")
    
    # Input for prompt enhancement
    text_input = gr.Textbox(label="Enter prompt for image generation")
    text_button = gr.Button("Generate Enhanced Prompt")
    text_output = gr.Textbox(label="Enhanced prompt")

    # Define interaction
    text_button.click(enhanced_prompt, inputs=text_input, outputs=text_output)

# Launch the app
demo.launch()