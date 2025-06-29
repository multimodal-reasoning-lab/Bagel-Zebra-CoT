import json
import os
import re
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset
from ..data_utils import pil_img2rgb
from ..distributed_iterable_dataset import DistributedIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class ThinkTraceJSONLIterableDataset(InterleavedBaseIterableDataset, DistributedIterableDataset):
    def __init__(
        self, 
        dataset_name, 
        transform, 
        tokenizer, 
        vit_transform,
        jsonl_path_list, 
        data_dir_list, 
        num_used_data,
        local_rank=0, 
        world_size=1, 
        num_workers=8, 
        data_status=None,
        shuffle_lines=True, 
        shuffle_seed=0,
        image_prefix_dir=None,
    ):
        """
        Dataset for think-trace style JSONL files with interleaved text and images.
        
        Args:
            dataset_name: Name of the dataset
            transform: Transform for VAE images  
            tokenizer: Text tokenizer
            vit_transform: Transform for VIT images
            jsonl_path_list: List of JSONL file paths
            data_dir_list: List of base directories (should match jsonl_path_list)
            num_used_data: List of number of samples to use from each JSONL. If a value is None or non-positive, all data from that JSONL will be used.
            image_prefix_dir: Absolute path to prepend to relative image paths
            Other args: Standard distributed dataset args
        """
        DistributedIterableDataset.__init__(self, dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform  
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.image_prefix_dir = image_prefix_dir or ""
        
        self.start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        self.end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.im_start = tokenizer.convert_tokens_to_ids('<|im_start|>')
        
        self.data_paths = self.get_data_paths(
            jsonl_path_list,
            num_used_data, 
            shuffle_lines,
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(self, jsonl_path_list, num_used_data, shuffle_lines, shuffle_seed):
        data_paths = []
        if not isinstance(num_used_data, list):
            num_used_data = [num_used_data] * len(jsonl_path_list)

        for jsonl_path, num_data_point in zip(jsonl_path_list, num_used_data):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            
            # Convert 'None' string to None type
            if num_data_point == 'None':
                num_data_point = None

            if num_data_point is not None and int(num_data_point) > 0:
                raw_data = raw_data[:int(num_data_point)]

            data_paths.extend(raw_data)
        return data_paths

    def extract_image_references(self, text):
        """Extract image references from text like <image_start>[problem_image_1]<image_end>"""
        pattern = r'<image_start>\[([^\]]+)\]<image_end>'
        matches = re.findall(pattern, text)
        return matches

    def replace_image_references(self, text):
        """Replace image references with placeholder tokens for processing"""
        pattern = r'<image_start>\[([^\]]+)\]<image_end>'
        # Replace with a special placeholder that we'll process later
        return re.sub(pattern, '<IMAGE_PLACEHOLDER>', text)

    def remove_thought_patterns(self, text):
        """Remove THOUGHT x: patterns from text"""
        # Remove patterns like "THOUGHT 1:", "THOUGHT 2:", etc.
        pattern = r'THOUGHT\s*\d+:\s*'
        return re.sub(pattern, '', text)

    def load_image_safely(self, data_item, image_key):
        """Load image with null checking and path resolution"""
        if image_key not in data_item or data_item[image_key] is None:
            return None
        
        image_path = data_item[image_key]
        full_path = os.path.join(self.image_prefix_dir, image_path)
        
        try:
            return pil_img2rgb(Image.open(full_path))
        except Exception as e:
            print(f"Failed to load image {full_path}: {e}")
            return None

    def parse_row(self, json_line):
        """Parse a single JSON line into the required format"""
        try:
            data_item = json.loads(json_line.strip())
        except:
            traceback.print_exc()
            return {}

        # Extract the main fields
        prompt = "You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. Think step by step and generate visual aids to enhance your problem-solving. You should first think about the reasoning and planning process in the mind before generating visual aids. Wrap your text reasoning with <think></think> tokens, and wrap your final conclusion with <answer></answer> tokens. Provide your final conclusion clearly in the format of '<answer>Final Answer: <answer here></answer>'"
        question = data_item.get('Question', '')
        question = f'Question: {question}'
        reasoning_trace = data_item.get('Text Reasoning Trace', '')
        reasoning_trace = f'{reasoning_trace}'
        final_answer = data_item.get('Final Answer', '')
        final_answer = f'<answer>Final Answer: {final_answer}</answer>'

        if not question or not reasoning_trace or not final_answer:
            return {}

        # Build the sequence
        data = self._init_data()

        # 0. Add prompt
        data = self._add_text(data, prompt, need_loss=False, enable_cfg=True)

        # 1. Add question (with image parsing)
        question_image_refs = self.extract_image_references(question)
        if question_image_refs:
            clean_question = self.replace_image_references(question)
            question_text_parts = clean_question.split('<IMAGE_PLACEHOLDER>')
            
            if len(question_text_parts) != len(question_image_refs) + 1:
                print(f"Mismatch in question: text parts {len(question_text_parts)}, images {len(question_image_refs)}")
                return {}

            question_images = []
            for image_ref in question_image_refs:
                image = self.load_image_safely(data_item, image_ref)
                if image is None:
                    print(f"Skipping sample due to missing image in question: {image_ref}")
                    return {}
                question_images.append(image)


            for i, text_part in enumerate(question_text_parts):
                if text_part.strip():
                    # Question text has no loss, so no need for vision start prediction
                    data = self._add_text(data, text_part.strip(), need_loss=False, enable_cfg=True)
                if i < len(question_images):
                    data = self._add_image(
                        data, question_images[i], 
                        need_loss=False, # No loss for question images
                        need_vae=False,   # VAE conditioning
                        need_vit=True,   # VIT understanding
                        enable_cfg=True,
                    )
        else:
            # Original behavior if no images in question
            data = self._add_text(data, question, need_loss=False, enable_cfg=True)
        
        # 2. Interleave text parts and images from reasoning trace
        image_refs = self.extract_image_references(reasoning_trace)
        
        loaded_images = []
        for image_ref in image_refs:
            image = self.load_image_safely(data_item, image_ref)
            if image is not None:
                loaded_images.append(image)
            else:
                # If image fails to load, skip this sample
                print(f"Skipping sample due to missing image: {image_ref}")
                return {}

        # Clean reasoning trace by removing image references for text processing
        clean_reasoning_trace = self.replace_image_references(reasoning_trace)
        
        # Remove THOUGHT patterns from the reasoning trace
        clean_reasoning_trace = self.remove_thought_patterns(clean_reasoning_trace)
        
        # Append final answer to the reasoning trace
        # clean_reasoning_trace += f"\n\nFinal Answer: {final_answer}"
        
        # Split reasoning trace by image placeholders to interleave text and images
        text_parts = clean_reasoning_trace.split('<IMAGE_PLACEHOLDER>')
        
        if len(text_parts) != len(loaded_images) + 1:
            print(f"Mismatch between text parts ({len(text_parts)}) and images ({len(loaded_images)})")
            return {}

        # 4. Interleave text parts and images from reasoning trace
        for i, text_part in enumerate(text_parts):
            # Add text part if not empty
            if text_part.strip():
                # Wrap reasoning text with <think></think> tokens
                wrapped_text = f"<think>{text_part.strip()}</think>"
                
                # Determine what the im_end token should predict
                if i < len(loaded_images):
                    # If this text part is followed by an image, predict vision_start
                    next_token_label = self.start_of_image
                elif i == len(text_parts) - 1:
                    # If this is the last text part, predict im_start for final answer
                    next_token_label = self.im_start
                else:
                    next_token_label = None
                    
                data = self._add_text(data, wrapped_text, need_loss=True, enable_cfg=True, next_token_label=next_token_label)
            
            # Add image if available
            if i < len(loaded_images):
                # Add image with both VAE and VIT processing for full capability
                data = self._add_image(
                    data,
                    loaded_images[i], 
                    need_loss=True,  # VAE generation loss
                    need_vae=True,   # VAE conditioning
                    need_vit=True,   # VIT understanding
                    enable_cfg=True,
                )

        # 5. Add final answer
        data = self._add_text(data, final_answer, need_loss=True, enable_cfg=True)

        return data

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, json_line in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data = self.parse_row(json_line)
                    if len(data) == 0:
                        continue

                    # Check if we have any loss 
                    has_loss = any(item['loss'] for item in data['sequence_plan'])
                    if not has_loss:
                        print('No loss defined, skipped.')
                        continue

                    data['data_indexes'] = {
                        "data_indexes": row_idx,
                        "worker_id": worker_id, 
                        "dataset_name": self.dataset_name,
                    }
                    yield data

                except Exception as e:
                    print(f"Error processing row {row_idx}: {e}")
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
