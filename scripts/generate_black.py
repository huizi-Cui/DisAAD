import os
import argparse
import nltk
import openai
import re
import json
import hashlib
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import transformers
from tqdm import tqdm
import textwrap
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import datasets
from datasets import load_dataset
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from utils import merge_jsonl_files
from metrics import topk, get_eu
from utils import seed_everything, color_print
from utils import save2jsonl, save2json_SE, load_jsonl_file
from utils import get_one_pass_metric, get_muti_pass_metric
from utils import process_decoded_str
from utils import calculate_and_save_metrics
from utils import calculate_and_save_metrics_llm
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel

nltk.download = lambda *a, **kw: True

metrics = [
    ("prob", None),
    ("entropy", None),
    ("au", 5),
    ("eu", 5),
    ("au_2", 2),
    ("eu_2", 2)
]

# Base directory for local model files and adapters
# Can be overridden via MODEL_BASE_DIR environment variable
_MODEL_BASE_DIR = os.environ.get('MODEL_BASE_DIR', './models')

HF_NAMES = {
    'gpt4': {
        "black_box": "gpt-4",
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "black_box_prompt_type": "gpt4",
        "proxy_prompt_type": "none"
    },
    'gpt4_llama3.2_3b': {
        "black_box": "gpt-4",
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "black_box_prompt_type": "gpt4",
        "proxy_prompt_type": "llama3",
        "api_key": "",
        "api_base": "https://api.openai.com/v1"
    },
    'sft_gpt4_llama3.2_3b': {
        "black_box": "gpt-4",
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "lora_adapter": f"{_MODEL_BASE_DIR}/sft_models/gpt4_llama3.2_3b",
        "black_box_prompt_type": "gpt4",
        "proxy_prompt_type": "llama3",
    }
}

def get_prompt(prompt_type: str, question: str) -> str:
    if prompt_type == "llama2":
        return f"Answer the question concisely. Q: {question} A:"
    elif prompt_type == "gpt4":
        return f"Answer the question concisely. Q: {question} A:"
    else:
        return textwrap.dedent(f"""\
            <|start_header_id|>system<|end_header_id|>

            Environment: ipython
            Tools: none

            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>

            Answer the question concisely. Q: {question} A:<|eot_id|>
        """)

def load_model(model_path: str, model_type: str):
    print(f"\n\033[1;34m=== Loading model: {model_type} ===\033[0m")
    print(f"Model path: {model_path}")

    if '70b' in model_path:
        print("Loading 70b model")
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

def load_sft_model(model_config: dict):
    print(f"\n\033[1;34m=== Loading SFT model ===\033[0m")
    print(f"Base model: {model_config['base_model']}")
    print(f"LoRA adapter: {model_config['lora_adapter']}")

    if not os.path.exists(model_config["lora_adapter"]):
        raise FileNotFoundError(f"LoRA adapter not found: {model_config['lora_adapter']}")

    base_model = LlamaForCausalLM.from_pretrained(
        model_config["base_model"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    try:
        model = PeftModel.from_pretrained(
            base_model,
            model_config["lora_adapter"],
            torch_dtype=torch.float16
        )
        print(f"\033[1;32mLoRA adapter loaded\033[0m")
    except Exception as e:
        print(f"\033[1;31mLoRA loading failed: {str(e)}\033[0m")
        raise

    print("Merging LoRA weights...")
    try:
        model = model.merge_and_unload()
        print(f"\033[1;32mLoRA weights merged\033[0m")
        return model
    except Exception as e:
        print(f"\033[1;31mLoRA merge failed: {str(e)}\033[0m")
        raise

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--black_model', type=str, default='llama2_70b')
    parser.add_argument('--model_name', type=str, default='llama3.2_3b')
    parser.add_argument('--model_type', type=str, default='base')
    parser.add_argument('--dataset_name', type=str, default='tqa')
    parser.add_argument('--gene', type=int, default=0)
    parser.add_argument('--num_gene', type=int, default=1)
    parser.add_argument('--generate_gt', type=int, default=0)
    parser.add_argument('--thres_gt', type=float, default=0.5)
    parser.add_argument('--generate_gt_llm', type=int, default=0)
    parser.add_argument("--mode", type=str, default='one_pass', choices=['one_pass', 'muti_pass'])
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--gpuid", type=str, default="0,1")
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument('--api_key', type=str, default="", help="OpenAI API key")
    parser.add_argument('--api_base', type=str, default="https://api.openai.com/v1")
    parser.add_argument('--api_model_name', type=str, default="gpt-4-0613")
    parser.add_argument('--top_p', type=float, default=0.96)
    args = parser.parse_args()

    print(args.model_name, args.mode, args.temp, args.gpuid)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
    max_memory = {}
    for i in range(num_gpus):
        total_mem = torch.cuda.get_device_properties(i).total_memory
        max_memory[i] = f"{int(total_mem * 0.9 / 1024**3)}GiB"

    if args.mode == 'one_pass':
        args.num_gene = 1
    elif args.mode == 'muti_pass':
        args.num_gene = 10

    model_config = HF_NAMES[args.model_name]

    black_box_prompt_type = model_config["black_box_prompt_type"]
    proxy_prompt_type = model_config["proxy_prompt_type"]

    generation_config_one = {
        "num_beams": 1,
        "num_return_sequences": 1,
        "do_sample": False,
        "max_new_tokens": 128,
        "output_scores": False,
        "return_dict_in_generate": True,
    }

    generation_config_muti = {
        "num_beams": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "max_new_tokens": 128,
        "output_scores": False,
        "return_dict_in_generate": True,
        "temperature": args.temp,
        "top_p": 1.0
    }

    temper = generation_config_muti["temperature"]
    new_folder_one = f"./results/exp_{args.black_model}/gene_{args.model_name}/gene_{args.dataset_name}/one_eval"
    new_folder_muti = f"./results/exp_{args.black_model}/gene_{args.model_name}/gene_{args.dataset_name}/muti_eval"

    os.makedirs(new_folder_one, exist_ok=True)
    os.makedirs(new_folder_muti, exist_ok=True)

    save_one_pass = f"{new_folder_one}/{args.model_name}_one_pass_gene_{temper}_{args.num_samples}.jsonl"
    save_muti_pass = f"{new_folder_muti}/{args.model_name}_muti_pass_gene_{temper}_{args.num_samples}.jsonl"

    save_eval_bleurt = f"{new_folder_one}/{args.model_name}_eval_bleurt_{temper}_{args.num_samples}.jsonl"
    save_eval_llm = f"{new_folder_one}/{args.model_name}_eval_llm_{temper}_{args.num_samples}.jsonl"
    save_merge_one_pass = f"{new_folder_one}/{args.model_name}_one_pass_gene_merge_{temper}_{args.num_samples}_llm{args.generate_gt_llm}.jsonl"
    save_merge_muti_pass = f"{new_folder_muti}/{args.model_name}_muti_pass_gene_merge_{temper}_{args.num_samples}_llm{args.generate_gt_llm}.jsonl"

    black_box_answer_file = f"./results/exp_{args.black_model}/gene_{args.black_model}/gene_{args.dataset_name}/{args.black_model}_answers_{args.dataset_name}_{args.num_samples}.jsonl"

    # TQA dataset
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    if args.num_samples:
        if hasattr(dataset, 'select'):
            dataset = dataset.select(range(args.num_samples))
        else:
            dataset = dataset[:args.num_samples] if len(dataset) > args.num_samples else dataset

    end_index_diy = min(args.num_samples, len(dataset))
    total_samples = end_index_diy

    if args.gene:
        if args.mode == "one_pass" and os.path.exists(save_one_pass):
            with open(save_one_pass, 'w') as f:
                pass

        is_gpt4_black_box = model_config["black_box"] == "gpt-4"
        black_box_model = None
        black_box_tokenizer = None
        black_box_answers = []

        if os.path.exists(black_box_answer_file):
            print(f"\033[1;33m=== Loading saved black box answers: {black_box_answer_file} ===\033[0m")
            with open(black_box_answer_file, "r", encoding="utf-8") as f:
                for line in f:
                    black_box_answers.append(json.loads(line.strip())["answer"])
        else:
            print(f"\033[1;33m=== Generating black box answers: {black_box_answer_file} ===\033[0m")
            os.makedirs(os.path.dirname(black_box_answer_file), exist_ok=True)
            if is_gpt4_black_box:
                with open(black_box_answer_file, "w", encoding="utf-8") as f:
                    for i in tqdm(range(0, total_samples), desc="Generating GPT-4 answers"):
                        question = dataset[i]['question']
                        try:
                            client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)
                            completion = client.chat.completions.create(
                                model=args.api_model_name,
                                messages=[{"role": "user", "content": get_prompt(black_box_prompt_type, question)}],
                                max_tokens=200,
                                temperature=args.temp,
                                top_p=args.top_p
                            )
                            clean_decoded = completion.choices[0].message.content.strip()
                            black_box_answers.append(clean_decoded)
                            f.write(json.dumps({"question_id": i, "question": question, "answer": clean_decoded}, ensure_ascii=False) + "\n")
                        except Exception:
                            black_box_answers.append("")
                            f.write(json.dumps({"question_id": i, "question": question, "answer": ""}, ensure_ascii=False) + "\n")
            else:
                black_box_model = load_model(model_config["black_box"], "black_box")
                black_box_tokenizer = AutoTokenizer.from_pretrained(model_config["black_box"], trust_remote_code=True)
                with open(black_box_answer_file, "w", encoding="utf-8") as f:
                    for i in tqdm(range(0, total_samples), desc="Generating local model answers"):
                        question = dataset[i]['question']
                        try:
                            prompt_ids = black_box_tokenizer(get_prompt(black_box_prompt_type, question), return_tensors='pt').input_ids.to(black_box_model.device)
                            output = black_box_model.generate(prompt_ids, **generation_config_one)
                            clean_decoded, _ = process_decoded_str(
                                black_box_tokenizer.decode(output.sequences[0, prompt_ids.shape[-1]:], skip_special_tokens=True),
                                ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >"],
                                black_box_tokenizer
                            )
                            black_box_answers.append(clean_decoded)
                            f.write(json.dumps({"question_id": i, "question": question, "answer": clean_decoded}, ensure_ascii=False) + "\n")
                        except Exception:
                            black_box_answers.append("Error occurred")
                            f.write(json.dumps({"question_id": i, "question": question, "answer": "Error occurred"}, ensure_ascii=False) + "\n")

        if not is_gpt4_black_box and black_box_model is None:
            black_box_model = load_model(model_config["black_box"], "black_box", max_memory)
            black_box_tokenizer = AutoTokenizer.from_pretrained(model_config["black_box"], trust_remote_code=True)

        if args.model_type == 'black':
            proxy_model = black_box_model if not is_gpt4_black_box else None
            proxy_tokenizer = black_box_tokenizer if not is_gpt4_black_box else AutoTokenizer.from_pretrained(model_config["base_model"], trust_remote_code=True)
        elif args.model_type == 'sft':
            model_config_copy = model_config.copy()
            if "lora_adapter" in model_config_copy:
                model_config_copy["lora_adapter"] = model_config_copy["lora_adapter"].format(dataset_name=args.dataset_name)
            proxy_model = load_sft_model(model_config_copy, max_memory)
            proxy_tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"], trust_remote_code=True)
        elif args.model_type == 'base':
            proxy_model = load_model(model_config["base_model"], "proxy", max_memory)
            proxy_tokenizer = AutoTokenizer.from_pretrained(model_config["base_model"], trust_remote_code=True)

        print(f"\n\033[1;34m=== Generating answers ({total_samples} questions) ===\033[0m")
        for i in tqdm(range(0, total_samples), desc="Processing questions"):
            try:
                question = dataset[i]['question']
                if args.mode == 'one_pass':
                    generated_text = black_box_answers[i]
                    if is_gpt4_black_box:
                        clean_decoded = generated_text
                        for phrase in ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >"]:
                            if phrase in clean_decoded:
                                clean_decoded = clean_decoded.split(phrase)[0].strip()
                    else:
                        clean_decoded, clean_generated_tokens_length = process_decoded_str(
                            generated_text,
                            ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >"],
                            black_box_tokenizer
                        )

                    full_text = get_prompt(proxy_prompt_type, question) + generated_text
                    prefix_ids = proxy_tokenizer(get_prompt(proxy_prompt_type, question), return_tensors='pt').input_ids

                    full_input_ids = proxy_tokenizer(full_text, return_tensors='pt').input_ids.to(proxy_model.device)
                    prefix_length = prefix_ids.shape[1]
                    labels = full_input_ids.clone()
                    labels[:, :prefix_length] = -100
                    with torch.no_grad():
                        outputs = proxy_model(input_ids=full_input_ids, labels=labels)

                    logits = outputs.logits
                    start_index = prefix_length - 1
                    end_index_raw = min(start_index + (full_input_ids.shape[1] - prefix_length), logits.shape[1])
                    gen_logits = logits[0, start_index:end_index_raw, :]
                    logits_list = [gen_logits[j].unsqueeze(0) for j in range(gen_logits.shape[0])]

                    if is_gpt4_black_box:
                        clean_generated_tokens_length = len(proxy_tokenizer.encode(clean_decoded, add_special_tokens=False))
                    metric_dict, logit_dict = get_one_pass_metric(logits_list, clean_generated_tokens_length, metrics, get_eu, topk)
                    save2jsonl(clean_decoded, metric_dict, logit_dict, save_one_pass, i)

                elif args.mode == 'muti_pass':
                    metric_dict = {}
                    for gen_iter in range(args.num_gene):
                        if is_gpt4_black_box:
                            try:
                                client = openai.OpenAI(api_key=model_config.get("api_key", args.api_key), base_url=model_config.get("api_base", args.api_base))
                                completion = client.chat.completions.create(
                                    model=model_config.get("api_model_name", args.api_model_name),
                                    messages=[{"role": "user", "content": get_prompt(black_box_prompt_type, question)}],
                                    max_tokens=200,
                                    temperature=args.temp,
                                    top_p=args.top_p,
                                    n=args.num_gene
                                )
                                generated_text = completion.choices[0].message.content
                                clean_decoded = generated_text
                                for ph in ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >", "<think>", "</think>"]:
                                    if ph in clean_decoded:
                                        clean_decoded = clean_decoded.split(ph)[0].strip()
                                clean_decoded = clean_decoded.replace("\n\n", "\n").strip()
                            except Exception:
                                continue
                        else:
                            with torch.no_grad():
                                try:
                                    p_ids = black_box_tokenizer(get_prompt(black_box_prompt_type, question), return_tensors='pt').input_ids.to(black_box_model.device)
                                    out = black_box_model.generate(p_ids, **generation_config_muti)
                                    generated_text = black_box_tokenizer.decode(out.sequences[0, p_ids.shape[-1]:], skip_special_tokens=True)
                                    clean_decoded, _ = process_decoded_str(
                                        generated_text,
                                        ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >", "<think>", "</think>"],
                                        black_box_tokenizer
                                    )
                                except RuntimeError:
                                    continue

                        full_text = get_prompt(proxy_prompt_type, question) + generated_text
                        prefix_ids = proxy_tokenizer(get_prompt(proxy_prompt_type, question), return_tensors='pt').input_ids

                        full_input_ids = proxy_tokenizer(full_text, return_tensors='pt').input_ids.to(proxy_model.device)
                        prefix_length = prefix_ids.shape[1]
                        with torch.no_grad():
                            outputs = proxy_model(input_ids=full_input_ids, labels=full_input_ids.clone())
                        gen_logits = outputs.logits[0, prefix_length-1:min(prefix_length-1 + (full_input_ids.shape[1]-prefix_length), outputs.logits.shape[1]), :]
                        logits_list = [gen_logits[j].unsqueeze(0) for j in range(gen_logits.shape[0])]
                        if is_gpt4_black_box:
                            clean_generated_tokens_length = len(proxy_tokenizer.encode(clean_decoded, add_special_tokens=False))
                        else:
                            _, clean_generated_tokens_length = process_decoded_str(
                                generated_text,
                                ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >", "<think>", "</think>"],
                                black_box_tokenizer
                            )
                        metric_dict = get_muti_pass_metric(
                            logits_list,
                            full_input_ids[0, prefix_length:],
                            clean_generated_tokens_length,
                            question,
                            clean_decoded,
                            gen_iter,
                            metric_dict
                        )
                    save2json_SE(metric_dict, save_muti_pass, i)
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"\033[1;31mError processing question {i+1}: {str(e)}\033[0m")

    elif args.generate_gt:
        if os.path.exists(save_eval_bleurt):
            with open(save_eval_bleurt, 'w') as f:
                pass
        if os.path.exists(save_merge_one_pass):
            with open(save_merge_one_pass, 'w') as f:
                pass

        model = BleurtForSequenceClassification.from_pretrained('BLEURT-20').cuda()
        tokenizer = BleurtTokenizer.from_pretrained('BLEURT-20')

        one_pass_data = load_jsonl_file(save_one_pass)
        actual_length = min(len(one_pass_data), end_index_diy)

        for i in range(actual_length):
            question_id = one_pass_data[i].get("question_id", [])
            predictions = np.array([one_pass_data[i].get("answer", [])], dtype=object)
            all_answers = [dataset[i]['best_answer']] + dataset[i]['correct_answers']
            calculate_and_save_metrics(question_id, predictions, all_answers, save_eval_bleurt, model, tokenizer)

        merge_jsonl_files(save_one_pass, save_eval_bleurt, save_merge_one_pass)

    elif args.generate_gt_llm:
        if os.path.exists(save_eval_llm):
            with open(save_eval_llm, 'w') as f:
                pass
        if os.path.exists(save_merge_one_pass):
            with open(save_merge_one_pass, 'w') as f:
                pass

        Judge_model = 'mistralai/Mistral-7B-Instruct-v0.1'
        judge_model = AutoModelForCausalLM.from_pretrained(
            Judge_model,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(Judge_model, trust_remote_code=True)

        one_pass_data = load_jsonl_file(save_one_pass)
        actual_length = min(len(one_pass_data), end_index_diy)

        for i in range(actual_length):
            try:
                question = dataset[i]['question']
                all_answers = [dataset[i]['best_answer']] + dataset[i]['correct_answers']
                predictions = np.array([one_pass_data[i].get("answer", [])], dtype=object)
                calculate_and_save_metrics_llm(i, question, predictions, all_answers, save_eval_llm, judge_model, judge_tokenizer)
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

        merge_jsonl_files(save_one_pass, save_eval_llm, save_merge_one_pass)

if __name__ == '__main__':
    seed_everything(42)
    main()
