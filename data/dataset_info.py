# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'think_trace': ThinkTraceJSONLIterableDataset,
}


DATASET_INFO = {
    'think_trace': {
        'think_trace_dataset': {
            'data_dir': '/dev/shm/data/Zebra-CoT/zebra-cot-images',
            'jsonl_path': '/dev/shm/data/Zebra-CoT/zebra_cot.jsonl',
            'image_prefix_dir': '/dev/shm/data/Zebra-CoT',  # Base path for relative image paths
            'num_total_samples': 1000,
        },
    },
}