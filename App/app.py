import gradio as gr
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import spacy
import torch
from diffusers import StableDiffusionPipeline

# Load environment variables - hugging face api key
load_dotenv(find_dotenv())

# spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

# NLP processing function
def nlp_processing(text):
    doc = nlp(text) 
    
    filtered_tokens = [token.text for token in doc if not token.is_stop]

    return " ".join(filtered_tokens)

# Configuration class
class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9

# Load the stable diffusion model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your token here'
)
image_gen_model = image_gen_model.to(CFG.device)




# Function to enhance the prompt using Mistral model
def enhanced_prompt(user_prompt):
    try:
        processed_prompt = nlp_processing(user_prompt)
        prompt_enhancer = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
        final_prompt = prompt_enhancer("give a short but descriptive enough enhanced prompt for the following prompt for image generation:" + processed_prompt)
        return final_prompt[0]['generated_text']
    except Exception as e:
        return str(e)

# Function to generate an image from the enhanced prompt
def generate_image(prompt):
    try:
        image = image_gen_model(
            prompt, num_inference_steps=CFG.image_gen_steps,
            generator=CFG.generator,
            guidance_scale=CFG.image_gen_guidance_scale
        ).images[0]
        image = image.resize(CFG.image_gen_size)
        return image
    except Exception as e:
        return str(e)

# Combined function to enhance prompt and generate image
def enhance_and_generate_image(user_prompt):
    enhanced = enhanced_prompt(user_prompt)
    if isinstance(enhanced, str):
        return enhanced, None
    image = generate_image(enhanced)
    return enhanced, image

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Image Generator")

    # Input for prompt enhancement and image generation
    text_input = gr.Textbox(label="Enter prompt for image generation")
    text_button = gr.Button("Generate Enhanced Prompt and Image")
    text_output = gr.Textbox(label="Enhanced prompt")
    image_output = gr.Image(label="Generated Image")

    # Define interaction
    text_button.click(enhance_and_generate_image, inputs=text_input, outputs=[text_output, image_output])

# Launch the app
demo.launch()
