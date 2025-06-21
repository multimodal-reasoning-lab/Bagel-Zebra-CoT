# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any, Tuple
import json
import os

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache


VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
        # Add vision start token to the tokenizer if not already present
        self.vision_start_token = "<|vision_start|>"
        
        # Get token ID for vision start token
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids(self.vision_start_token)
        
        # If token doesn't exist, you may need to add it to tokenizer
        if self.vision_start_token_id == self.tokenizer.unk_token_id:
            print(f"Warning: {self.vision_start_token} not found in tokenizer vocabulary")
            
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50, 
        timestep_shift=3.0
    ):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image
        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text_with_vision_detection(
        self, 
        gen_context, 
        max_length: int = 500, 
        do_sample: bool = True, 
        temperature: float = 1.0,
        stop_at_vision_token: bool = True
    ) -> Tuple[str, bool, List[int]]:
        """
        Generate text and detect if vision_start_token is generated.
        Returns: (generated_text, vision_token_detected, generated_token_ids)
        """
        gen_context_copy = deepcopy(gen_context)
        past_key_values = gen_context_copy['past_key_values']
        kv_lens = gen_context_copy['kv_lens']
        ropes = gen_context_copy['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        
        # Modified to track individual tokens during generation
        generated_tokens = []
        vision_token_detected = False
        
        # Generate token by token to detect vision_start_token
        for _ in range(max_length):
            # Generate one token at a time
            next_token = self.model.generate_text(
                past_key_values=past_key_values,
                max_length=1,  # Generate only one token
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=self.new_token_ids['eos_token_id'],
                **generation_input,
            )
            
            token_id = next_token[0, 0].item()
            generated_tokens.append(token_id)
            
            # Check if we hit the vision start token
            if token_id == self.vision_start_token_id:
                vision_token_detected = True
                if stop_at_vision_token:
                    break
                    
            # Check if we hit the end token
            if token_id == self.new_token_ids['eos_token_id']:
                break
                
            # Update generation input for next token
            generation_input['input_ids'] = next_token
            
        # Decode the generated tokens
        output = self.tokenizer.decode(generated_tokens)
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1] if '<|im_start|>' in output else output
        
        return output, vision_token_detected, generated_tokens

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        """Original gen_text method for backward compatibility"""
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    @torch.no_grad()
    def seamless_interleaved_generation(
        self,
        initial_inputs: Union[str, List[Union[str, Image.Image]]],
        max_length: int = 2000,
        max_images: int = 5,
        do_sample: bool = True,
        text_temperature: float = 0.7,
        image_shape: Tuple[int, int] = (1024, 1024),
        cfg_text_scale: float = 3.0,
        cfg_img_scale: float = 1.5,
        cfg_interval: List[float] = [0.4, 1.0],
        timestep_shift: float = 3.0,
        num_timesteps: int = 50,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
    ) -> List[Union[str, Image.Image]]:
        """
        Seamlessly generate interleaved text and images based on model predictions.
        When the model generates <|vision_start|> token, it switches to image generation.
        
        Args:
            initial_inputs: Can be a string prompt or a list of text/image inputs
        """
        output_list = []
        generated_images = 0
        
        # Initialize contexts
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Process initial inputs
            if isinstance(initial_inputs, str):
                initial_inputs = [initial_inputs]
            
            # Process all initial inputs (text and/or images)
            for input_item in initial_inputs:
                if isinstance(input_item, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_item, gen_context)
                    cfg_img_context = self.update_context_text(input_item, cfg_img_context)
                elif isinstance(input_item, Image.Image):
                    input_item = self.vae_transform.resize_transform(pil_img2rgb(input_item))
                    gen_context = self.update_context_image(input_item, gen_context, vae=True, vit=True)
                    image_shape = input_item.size[::-1]  # Update image shape based on input
                    cfg_text_context = deepcopy(gen_context)
            
            remaining_length = max_length
            
            while remaining_length > 0 and generated_images < max_images:
                # Generate text until we hit a vision token or reach max length
                text_output, vision_detected, token_ids = self.gen_text_with_vision_detection(
                    gen_context,
                    max_length=remaining_length,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    stop_at_vision_token=True
                )
                
                # Add generated text to output (excluding vision token if present)
                if text_output and text_output != self.vision_start_token:
                    clean_text = text_output.replace(self.vision_start_token, "").strip()
                    if clean_text:
                        output_list.append(clean_text)
                
                # Update context with generated text
                if text_output:
                    gen_context = self.update_context_text(text_output, gen_context)
                    cfg_img_context = self.update_context_text(text_output, cfg_img_context)
                
                remaining_length -= len(token_ids)
                
                # If vision token was detected, generate an image
                if vision_detected and generated_images < max_images:
                    # Save current context for CFG
                    cfg_text_context = deepcopy(gen_context)
                    
                    # Generate image
                    image = self.gen_image(
                        image_shape,
                        gen_context,
                        cfg_text_precontext=cfg_text_context,
                        cfg_img_precontext=cfg_img_context,
                        cfg_text_scale=cfg_text_scale,
                        cfg_img_scale=cfg_img_scale,
                        cfg_interval=cfg_interval,
                        timestep_shift=timestep_shift,
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    
                    output_list.append(image)
                    generated_images += 1
                    
                    # Update context with generated image for future generation
                    gen_context = self.update_context_image(image, gen_context, vae=True, vit=False)
                    cfg_text_context = deepcopy(gen_context)
                
                # If no vision token detected and we've generated text, we're done
                if not vision_detected:
                    break
                    
        return output_list
    
    def load_json_inputs(self, json_path: str, image_base_dir: Optional[str] = None) -> List[Union[str, Image.Image]]:
        """
        Load inputs from a JSON file that can contain text and image paths.
        
        Expected JSON format:
        [
            {"type": "text", "content": "Hello, this is some text"},
            {"type": "image", "path": "path/to/image.jpg"},
            {"type": "text", "content": "More text here"},
            ...
        ]
        
        Or simpler format:
        {
            "inputs": [
                "text string",
                {"image": "path/to/image.jpg"},
                "more text",
                ...
            ]
        }
        
        Args:
            json_path: Path to JSON file
            image_base_dir: Optional base directory for relative image paths
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        inputs = []
        
        # Handle different JSON formats
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'inputs' in data:
            items = data['inputs']
        else:
            raise ValueError("JSON must be a list or dict with 'inputs' key")
        
        for item in items:
            if isinstance(item, str):
                # Direct string input
                inputs.append(item)
            elif isinstance(item, dict):
                if 'type' in item:
                    # Format: {"type": "text/image", "content/path": "..."}
                    if item['type'] == 'text':
                        inputs.append(item.get('content', ''))
                    elif item['type'] == 'image':
                        image_path = item.get('path', item.get('content', ''))
                        if image_base_dir and not os.path.isabs(image_path):
                            image_path = os.path.join(image_base_dir, image_path)
                        try:
                            image = Image.open(image_path).convert('RGB')
                            inputs.append(image)
                        except Exception as e:
                            print(f"Error loading image {image_path}: {e}")
                elif 'image' in item:
                    # Format: {"image": "path"}
                    image_path = item['image']
                    if image_base_dir and not os.path.isabs(image_path):
                        image_path = os.path.join(image_base_dir, image_path)
                    try:
                        image = Image.open(image_path).convert('RGB')
                        inputs.append(image)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                elif 'text' in item:
                    # Format: {"text": "content"}
                    inputs.append(item['text'])
        
        return inputs
    
    def process_json_inputs(
        self,
        json_path: str,
        image_base_dir: Optional[str] = None,
        seamless_mode: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process inputs from a JSON file using either seamless or traditional mode.
        
        Args:
            json_path: Path to JSON file
            image_base_dir: Optional base directory for relative image paths
            seamless_mode: Whether to use seamless generation
            **kwargs: Additional generation parameters
        """
        # Load inputs from JSON
        inputs = self.load_json_inputs(json_path, image_base_dir)
        
        if seamless_mode:
            # Use seamless generation
            outputs = self.seamless_interleaved_generation(
                initial_inputs=inputs,
                **kwargs
            )
            
            # Organize outputs
            output_dict = {'images': [], 'texts': []}
            for output in outputs:
                if isinstance(output, Image.Image):
                    output_dict['images'].append(output)
                elif isinstance(output, str):
                    output_dict['texts'].append(output)
                    
            # For backward compatibility
            if output_dict['images']:
                output_dict['image'] = output_dict['images'][0]
            if output_dict['texts']:
                output_dict['text'] = ' '.join(output_dict['texts'])
                
            return output_dict
        else:
            # Use traditional interleave inference
            outputs = self.interleave_inference(inputs, **kwargs)
            
            # Organize outputs to match expected format
            output_dict = {'images': [], 'texts': []}
            for output in outputs:
                if isinstance(output, Image.Image):
                    output_dict['images'].append(output)
                elif isinstance(output, str):
                    output_dict['texts'].append(output)
                    
            # For backward compatibility
            if output_dict['images']:
                output_dict['image'] = output_dict['images'][0]
            if output_dict['texts']:
                output_dict['text'] = ' '.join(output_dict['texts'])
                
            return output_dict

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
    ) -> List[Union[str, Image.Image]]:
        """Original interleave_inference method for backward compatibility"""
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None,
        json_path: Optional[str] = None,
        image_base_dir: Optional[str] = None,
        seamless_mode: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point with seamless mode option and JSON input support.
        
        Args:
            image: Optional input image
            text: Optional input text
            json_path: Optional path to JSON file with inputs
            image_base_dir: Base directory for relative image paths in JSON
            seamless_mode: If True, uses seamless generation with vision tokens
            **kwargs: Additional arguments passed to generation methods
        """
        # Handle JSON input
        if json_path:
            return self.process_json_inputs(
                json_path, 
                image_base_dir, 
                seamless_mode=seamless_mode, 
                **kwargs
            )
        
        output_dict = {'images': [], 'texts': []}

        if seamless_mode and (text or image):
            # Build initial inputs list
            initial_inputs = []
            if image:
                initial_inputs.append(image)
            if text:
                initial_inputs.append(text)
                
            # Use seamless interleaved generation
            outputs = self.seamless_interleaved_generation(
                initial_inputs=initial_inputs,
                **kwargs
            )
            
            # Organize outputs
            for output in outputs:
                if isinstance(output, Image.Image):
                    output_dict['images'].append(output)
                elif isinstance(output, str):
                    output_dict['texts'].append(output)
                    
            # For backward compatibility, also set single image/text
            if output_dict['images']:
                output_dict['image'] = output_dict['images'][0]
            if output_dict['texts']:
                output_dict['text'] = ' '.join(output_dict['texts'])
                
        else:
            # Use original logic
            if image is None and text is None:
                print('Please provide at least one input: either an image or text.')
                return {'image': None, 'text': None}

            input_list = []
            if image is not None:
                input_list.append(image)
            if text is not None:
                input_list.append(text)

            output_list = self.interleave_inference(input_list, **kwargs)

            for i in output_list:
                if isinstance(i, Image.Image):
                    output_dict['image'] = i
                    output_dict['images'] = [i]
                elif isinstance(i, str):
                    output_dict['text'] = i
                    output_dict['texts'] = [i]
                    
        return output_dict

