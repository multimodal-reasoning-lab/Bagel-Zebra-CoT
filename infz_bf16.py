import os
import json
from datetime import datetime
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
checkpoint_dir = "path/to/your/HF_HOME/models/Bagel-Zebra-CoT"
checkpoint_file = "model_bf16.safetensors"
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)


print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"GPU memory per device:")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

# LLM config preparing (use base model configs)
llm_config = Qwen2Config.from_json_file(os.path.join(checkpoint_dir, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing (use base model configs)
vit_config = SiglipVisionConfig.from_json_file(os.path.join(checkpoint_dir, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading (use base model VAE)
vae_model, vae_config = load_ae(local_path=os.path.join(checkpoint_dir, "ae.safetensors"))

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
tokenizer = Qwen2Tokenizer.from_pretrained(checkpoint_dir)
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
    text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)

INTERLEAVED_SYSTEM_PROMPT = '''You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. Think step by step and use visual aids to enhance your problem-solving. Provide your final conclusion clearly in the format of "Final Answer: <answer here>"'''

prompt = '''Subtract all cylinders. Add 1 red sphere. How many objects are left?'''
image = Image.open('test_images/image.png')

print(prompt)
print('-'*50)

# Create output folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"reasoning_output_{timestamp}"
images_folder = os.path.join(output_folder, "images")
os.makedirs(images_folder, exist_ok=True)

# Save the original problem images if they exist
problem_image_paths = []
if image is not None:
    if isinstance(image, list):
        # Handle multiple images
        for i, img in enumerate(image):
            problem_image_path = os.path.join(images_folder, f"problem_image_{i+1}.png")
            relative_path = os.path.join("images", f"problem_image_{i+1}.png")
            img.save(problem_image_path)
            problem_image_paths.append(relative_path)
            print(f"Problem image {i+1} saved at '{problem_image_path}'")
    else:
        # Handle single image
        problem_image_path = os.path.join(images_folder, "problem_image.png")
        relative_path = os.path.join("images", "problem_image.png")
        image.save(problem_image_path)
        problem_image_paths.append(relative_path)
        print(f"Problem image saved at '{problem_image_path}'")

reasoning_text = []
reasoning_images = []
image_paths = []  # Store relative paths to images

# Create input with multiple images properly flattened
if image is not None:
    if isinstance(image, list):
        current_input = [prompt] + image  # Flatten the list of images
    else:
        current_input = [prompt, image]
else:
    current_input = [prompt]

# Loop until no more vision_start tokens
iteration = 0
while True:    
    # Get understanding output
    print(f"iteration: {iteration}")
    output = inferencer.interleave_inference(current_input, understanding_output=True, system_prompt=INTERLEAVED_SYSTEM_PROMPT, **inference_hyper)

    # Check for stopping conditions
    has_final_answer = 'Final Answer:' in output[0] or '<answer>' in output[0]
    
    # Stop if we have a final answer OR if there's no vision token (no more images to generate)
    # should_stop = has_final_answer or not has_vision_token
    should_stop = has_final_answer


    if should_stop:
        if output[0].strip():
            extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
            reasoning_text.append(extracted_text)
            print(f"{extracted_text}")
            current_input = current_input + [extracted_text]
        break
    
    extracted_text = output[0].split('<|im_end|>')[0].split('<|im_start|>')[1]
    reasoning_text.append(extracted_text)
    print(f"{extracted_text}")
    
    # Generate image based on current reasoning
    current_input_with_reasoning = current_input + [extracted_text]
    output = inferencer.interleave_inference(current_input_with_reasoning, system_prompt=INTERLEAVED_SYSTEM_PROMPT, **inference_hyper)
    image_output = output[0]

    # Save and collect the generated image
    reasoning_images.append(image_output)
    image_filename = f'reasoning_image_{iteration + 1}.png'
    image_path = os.path.join(images_folder, image_filename)
    relative_image_path = os.path.join("images", image_filename)  # Relative path for JSON
    
    image_output.save(image_path)
    image_paths.append(relative_image_path)
    print(f"Image saved at '{image_path}'")

    # Update input for next iteration
    current_input = current_input_with_reasoning + [image_output]
    
    iteration += 1
    print('-'*50)

# Save reasoning data to JSON
reasoning_data = {
    "timestamp": timestamp,
    "prompt": prompt,
    "system_prompt": INTERLEAVED_SYSTEM_PROMPT,
    "problem_image_paths": problem_image_paths if problem_image_paths else None,
    "response": [
        {
            "step": i + 1,
            "text": text,
            "image_path": image_paths[i] if i < len(image_paths) else None
        }
        for i, text in enumerate(reasoning_text)
    ],
    "total_steps": len(reasoning_text),
    "total_images": len(image_paths)
}

# Save JSON file
json_path = os.path.join(output_folder, "reasoning_data.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(reasoning_data, f, indent=2, ensure_ascii=False)

print(f"\nReasoning complete!")
print(f"Output folder: {output_folder}")
print(f"JSON metadata: {json_path}")
print(f"Generated {len(image_paths)} images and {len(reasoning_text)} text steps")

