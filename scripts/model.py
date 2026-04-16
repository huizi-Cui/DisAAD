# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
import time
import os
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

# predefined models - using HuggingFace model names
model_fullnames = {
    'llama3.1-8b':'meta-llama/Llama-3.1-8B-Instruct',
    'llama2-70b': 'meta-llama/Llama-2-70b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3.2-3b':'meta-llama/Llama-3.2-3B-Instruct',
    'GPT-2':'gpt2',
    'llama3.2-1b':'meta-llama/Llama-3.2-1B-Instruct',
    'llama2-7b':'meta-llama/Llama-2-7b-chat-hf',
    'qwen3-32b':'Qwen/Qwen3-32B',
}

float16_models = ['llama3.1-8b', 'llama2-70b', 'llama3.2-3b', 'llama3.2-3b', 'GPT-2', 'llama3.2-1b', 'llama2-7b', 'qwen3-32b']

def get_model_fullname(model_name):
    return model_fullnames[model_name] if model_name in model_fullnames else model_name

def load_black_model(model_name):
    """Load blackbox model"""
    print(f"\n\033[1;34m=== Loading model: {model_name} ===\033[0m")
    model_path = get_model_fullname(model_name)
    
    bnb_config = BitsAndBytesConfig(
        # load_in_8bit=True,
        load_in_4bit=True,          
        bnb_4bit_compute_dtype=torch.float16,  
        bnb_4bit_quant_type="nf4",  
    )

    if '70b' in model_path:
        print("I'm loading 70b model")
        model = LlamaForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path, 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return model

def load_model(model_name, device, cache_dir, device_map=None, max_memory=None, 
             low_cpu_mem_usage=True, use_quantization=False):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname} from local...')

    model_kwargs: dict = {
        "local_files_only": True,
        "trust_remote_code": True,
    }

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["gate_proj", "down_proj", "up_proj"]
        )
        model_kwargs["quantization_config"] = bnb_config

    if model_name in ["llama2-7b", "llama3.1-8b", "llama2-13b"]:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif model_name in float16_models:
        model_kwargs["torch_dtype"] = torch.float16

    if device_map:
        model_kwargs["device_map"] = device_map
    if max_memory:
        model_kwargs["max_memory"] = max_memory
        model_kwargs["offload_folder"] = os.path.join(cache_dir, "offload")
        model_kwargs["offload_state_dict"] = True
    if low_cpu_mem_usage:
        model_kwargs["low_cpu_mem_usage"] = True

    if any(name in model_name for name in ["llama", "gpt-neo"]):
        model_kwargs["use_safetensors"] = True

    if device_map == "auto" and not max_memory:
        model_kwargs["max_memory"] = {i: "23GiB" for i in range(torch.cuda.device_count())}
        
    model = AutoModelForCausalLM.from_pretrained(
        model_fullname,
        **model_kwargs,
    )
    return model

def load_tokenizer(model_name, for_dataset, cache_dir):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading tokenizer {model_fullname} from local...')
    
    optional_tok_kwargs: dict = {
        "local_files_only": True,
        "trust_remote_code": True
    }

    if for_dataset in ['pubmed', 'wild_sft'] or 'llama3' in model_name.lower():
        optional_tok_kwargs.update({
            'padding_side': 'left',
            'truncation_side': 'left'
        })
    else:
        optional_tok_kwargs['padding_side'] = 'right'

    if "gpt-neo" in model_name.lower():
        optional_tok_kwargs['fast'] = False

    base_tokenizer = AutoTokenizer.from_pretrained(
        model_fullname,
        **optional_tok_kwargs
    )

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        if '13b' in model_fullname.lower():
            base_tokenizer.pad_token_id = 0

    return base_tokenizer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama3.2-1b")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    args = parser.parse_args()

    load_tokenizer(args.model_name, 'pubmed', args.cache_dir)
    load_model(args.model_name, 'cuda', args.cache_dir, device_map="auto")