import os
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae

# Set paths for your trained checkpoint
checkpoint_dir = "/home/jovyan/workspace/bagel-training/h200-ckpt-0001200"
base_model_path = "/dev/shm/models/BAGEL-7B-MoT"

# Direct path to the safetensors file
checkpoint_file = "model_bf16.safetensors"
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

print(f"Checkpoint directory: {checkpoint_dir}")
print(f"Checkpoint file: {checkpoint_file}")
print(f"Full checkpoint path: {checkpoint_path}")
print(f"File exists: {os.path.exists(checkpoint_path)}")

print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"GPU memory per device:")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

# LLM config preparing (use base model configs)
llm_config = Qwen2Config.from_json_file(os.path.join(base_model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing (use base model configs)
vit_config = SiglipVisionConfig.from_json_file(os.path.join(base_model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading (use base model VAE)
vae_model, vae_config = load_ae(local_path=os.path.join(base_model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

# Create model with empty weights - IMPORTANT: Use float32 initially to match checkpoint
with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing (use base model tokenizer)
tokenizer = Qwen2Tokenizer.from_pretrained(base_model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 512, 14)

# Device mapping for 8x80GB GPUs - use bf16 directly
max_mem_per_gpu = "80GiB"

print("Setting up device mapping...")
device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    dtype=torch.bfloat16,  # Use bf16 for device mapping
)

print("Device map:", device_map)

# Handle same-device modules
same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    if first_device is not None:
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

print("Final device map:", device_map)

# Load checkpoint directly in bf16
print(f"Loading checkpoint directly in bfloat16: {checkpoint_path}")
print("Loading model from safetensors file...")

# Load model directly in bf16
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=checkpoint_path,
    device_map=device_map,
    offload_buffers=False,
    dtype=torch.bfloat16,   # Load directly as bf16
    force_hooks=True,
)

model = model.eval()

print('Model loaded directly in bfloat16!')
print(f"Model dtype: {next(model.parameters()).dtype}")
print("Model loading completed successfully!")

# Check memory usage
print("GPU memory usage after loading:")
for i in range(torch.cuda.device_count()):
    if torch.cuda.memory_allocated(i) > 0:
        allocated = torch.cuda.memory_allocated(i) / 1e9
        cached = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

# Rest of inference code
from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

inference_hyper=dict(
    do_sample=True,
    text_temperature=0.7,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)

INTERLEAVED_SYSTEM_PROMPT = '''You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. Think step by step and use visual aids to enhance your problem-solving. Provide your final conclusion clearly in the format of "Final Answer: <answer here>"'''


# prompt = '''A multi‑piece box jigsaw is missing part(s). Identify which option fills the hole(s).'''
# image = Image.open('/home/jovyan/workspace/bagel-training/eval/sample_12400/raw_files/images/problem_image_1.jpg')

# prompt = '''What is the best move for Black to play?\n\nA: Qd6\nB: Kf8\nC: Nf4\nD: Nxe4'''
# image = Image.open('/home/jovyan/workspace/bagel-training/eval/chess/game_2040/raw_files/images/problem_image_1.png')
# pdf_filename = "chess.pdf"

# prompt = '''Apply the following sequence of transformations to the blue shape: scale by 2×, then translate 1 left, then translate 1 down and 2 right, then translate 2 down and 1 right, then rotate 90° clockwise. Choose the option that shows the resulting shape.'''
# image = Image.open('/home/jovyan/workspace/bagel-training/new_eval/compose_8847/images/problem_image_1.jpg')
# pdf_filename = "compose_8847.pdf"

# prompt = '''Which of the figures shown bellow cannot be cut out of the figure illustrated nearby?'''
# image = Image.open('/home/jovyan/workspace/bagel-training/image.jpg')
# pdf_filename = "math.pdf"

prompt = '''Subtract all green metallic cylinders. Subtract all cyan blocks. How many objects are left?'''
image = Image.open('/home/jovyan/workspace/bagel-training/new_eval/image.png')
pdf_filename = "clevr.pdf"

# print(prompt)
# print('-'*50)

reasoning_text = []
reasoning_images = []
current_input = [prompt, image]
think = False

# Loop until no more vision_start tokens
iteration = 0
while True:    
    # Get understanding output
    print(f"iteration: {iteration}")
    output = inferencer.interleave_inference(current_input, understanding_output=True, system_prompt=INTERLEAVED_SYSTEM_PROMPT, think=think, **inference_hyper)

    should_stop = ('<|vision_start|>' not in output[0]) or ('Final Answer' in output[0])

    if should_stop:
        # print(f"should_stop: {output[0]}")
        if output[0].strip():
            extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
            reasoning_text.append(extracted_text)
            print(f"{extracted_text}")
            current_input = current_input + [extracted_text]
        break
    
    # Extract reasoning text
    # print(f"raw output: {output[0]}")
    extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
    reasoning_text.append(extracted_text)
    print(f"{extracted_text}")
    
    # Generate image based on current reasoning
    current_input_with_reasoning = current_input + [extracted_text]
    if not think:
        output = inferencer.interleave_inference(current_input_with_reasoning, system_prompt=INTERLEAVED_SYSTEM_PROMPT, think=think, **inference_hyper)
        image_output = output[0]
        
    else: 
        output = inferencer.interleave_inference(current_input_with_reasoning, system_prompt=INTERLEAVED_SYSTEM_PROMPT, think=think, **inference_hyper)

        thinking_text = output[0]
        print(f"image generation thinking_text: {thinking_text}")
        extracted_text = thinking_text.split('<|im_end|>')[0].split('<|im_start|>')[1]
        # reasoning_text.append(extracted_text)
        # current_input_with_reasoning = current_input + [extracted_text]
        image_output = output[1]

    # Save and collect the generated image
    reasoning_images.append(image_output)
    image_filename = f'reasoning_image_{iteration + 1}.png'
    image_output.save(image_filename)
    print(f"Image saved at '{image_filename}'")

    
    # Update input for next iteration
    current_input = current_input_with_reasoning + [image_output]
    
    iteration += 1
    print('-'*50)

# Create PDF with all reasoning text and images
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io

# Create PDF
doc = SimpleDocTemplate(pdf_filename, pagesize=A4)
styles = getSampleStyleSheet()
story = []

# Add title
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=16,
    spaceAfter=30,
    alignment=1  # Center alignment
)
story.append(Paragraph("Example", title_style))
story.append(Spacer(1, 20))

# Add original question
story.append(Paragraph("Original Question:", styles['Heading2']))
story.append(Paragraph(prompt.replace('\n', '<br/>'), styles['Normal']))
story.append(Spacer(1, 20))

# Add original image
story.append(Paragraph("Problem Image:", styles['Heading2']))
# Convert PIL image to reportlab format
img_buffer = io.BytesIO()
image.save(img_buffer, format='PNG')
img_buffer.seek(0)
img = RLImage(img_buffer, width=4*inch, height=4*inch)
story.append(img)
story.append(Spacer(1, 20))

# Add reasoning steps
for i, (text, img) in enumerate(zip(reasoning_text, reasoning_images)):
    story.append(Paragraph(f"Reasoning Step {i + 1}:", styles['Heading2']))
    story.append(Paragraph(text.replace('\n', '<br/>'), styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Add generated image
    story.append(Paragraph(f"Generated Image {i + 1}:", styles['Heading3']))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    rl_img = RLImage(img_buffer, width=4*inch, height=4*inch)
    story.append(rl_img)
    story.append(Spacer(1, 20))

# Add any final reasoning text that didn't generate an image
if len(reasoning_text) > len(reasoning_images):
    for i in range(len(reasoning_images), len(reasoning_text)):
        story.append(Paragraph(f"Final Reasoning {i + 1}:", styles['Heading2']))
        story.append(Paragraph(reasoning_text[i].replace('\n', '<br/>'), styles['Normal']))
        story.append(Spacer(1, 20))
# Build PDF
doc.build(story)
print(f"PDF saved as '{pdf_filename}'")
print(f"Total reasoning steps: {len(reasoning_text)}")
print(f"Total generated images: {len(reasoning_images)}")


