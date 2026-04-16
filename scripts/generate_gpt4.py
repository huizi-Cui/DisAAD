import os
import argparse
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
import evaluate
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer, BleurtConfig
from utils import merge_jsonl_files
from metrics import topk, get_eu
from utils import seed_everything, color_print
from utils import save2jsonl, convert_float32_to_float, save2json_SE, load_jsonl_file
from utils import get_one_pass_metric, get_muti_pass_metric
from utils import process_decoded_str
from utils import calculate_and_save_metrics
from utils import calculate_and_save_metrics_llm
from transformers.utils.quantization_config import BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

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
    },
    'llama2_13b': {
        "black_box": "meta-llama/Llama-2-13b-chat-hf",
        "base_model": "meta-llama/Llama-2-13b-chat-hf",
        "black_box_prompt_type": "llama2",
        "proxy_prompt_type": "llama2"
    },
    'llama3.2_3b': {
        "black_box": "meta-llama/Llama-2-70b-chat-hf",
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "black_box_prompt_type": "llama2",
        "proxy_prompt_type": "llama3"
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
    print(f"\n\033[1;34m=== Loading SFT proxy model ===\033[0m")
    print(f"Base model path: {model_config['base_model']}")
    print(f"LoRA adapter path: {model_config['lora_adapter']}")

    if not os.path.exists(model_config["lora_adapter"]):
        raise FileNotFoundError(f"LoRA adapter path not found: {model_config['lora_adapter']}")

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
        print(f"\033[1;32mLoRA adapter loaded successfully\033[0m")
    except Exception as e:
        print(f"\033[1;31mLoRA adapter load failed: {str(e)}\033[0m")
        raise

    print("Merging LoRA weights...")
    try:
        model = model.merge_and_unload()
        print(f"\033[1;32mLoRA weights merged successfully\033[0m")

        if hasattr(model, 'peft_config'):
            print("\033[1;31mWarning: PEFT config detected after merge, may not have unloaded properly\033[0m")
        else:
            print("\033[1;32mWeight merge verified\033[0m")

        return model
    except Exception as e:
        print(f"\033[1;31mLoRA weight merge failed: {str(e)}\033[0m")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--black_model', type=str, default='llama2_70b', help='llama2_70b, gpt4')
    parser.add_argument('--model_name', type=str, default='llama2_13b')
    parser.add_argument('--model_type', type=str, default='black', help='black, sft, base')
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
        "max_new_tokens": 64,
        "output_scores": False,
        "return_dict_in_generate": True,
    }

    generation_config_muti = {
        "num_beams": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "max_new_tokens": 64,
        "output_scores": False,
        "return_dict_in_generate": True,
        "temperature": args.temp,
        "top_p": 1.0
    }

    temper = generation_config_muti["temperature"]
    new_folder_one = f"exp_{args.black_model}/gene_{args.model_name}/gene_{args.dataset_name}/one_eval"
    new_folder_muti = f"exp_{args.black_model}/gene_{args.black_model}/gene_{args.dataset_name}/muti_eval"

    os.makedirs(new_folder_one, exist_ok=True)
    os.makedirs(new_folder_muti, exist_ok=True)

    save_one_pass = f"exp_{args.black_model}/gene_{args.model_name}/gene_{args.dataset_name}/{args.black_model}_answers_{args.dataset_name}_{args.num_samples}.jsonl"
    save_muti_pass = f"{new_folder_muti}/{args.black_model}_muti_pass_gene_{temper}_{args.num_samples}.jsonl"
    save_eval_bleurt = f"{new_folder_one}/{args.model_name}_eval_bleurt_{temper}_{args.num_samples}.jsonl"
    save_eval_llm = f"{new_folder_one}/{args.model_name}_eval_llm_{temper}_{args.num_samples}.jsonl"
    save_merge_one_pass = f"{new_folder_one}/{args.model_name}_one_pass_gene_merge_{temper}_{args.num_samples}_llm{args.generate_gt_llm}.jsonl"
    save_merge_muti_pass = f"{new_folder_muti}/{args.model_name}_muti_pass_gene_merge_{temper}_{args.num_samples}_llm{args.generate_gt_llm}.jsonl"

    gpt4_answer_file = f"exp_{args.black_model}/gene_{args.black_model}/gene_{args.dataset_name}/{args.black_model}_answers_{args.dataset_name}_{args.num_samples}.jsonl"

    if args.dataset_name == "tqa":
        dataset = load_dataset(
            "truthful_qa",
            "generation",
            split="validation"
        )
        if args.num_samples:
            if hasattr(dataset, 'select'):
                dataset = dataset.select(range(args.num_samples))
            else:
                dataset = dataset[:args.num_samples] if len(dataset) > args.num_samples else dataset
        end_index = args.num_samples

    elif args.dataset_name == 'bioasq':
        scratch_dir = os.getenv('SCRATCH_DIR', '.')
        path = "bioasq_training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question['exact_answer'], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question['exact_answer']
                    ]
                else:
                    exact_answers = [question['exact_answer']]

                dataset_dict["answers"].append({
                    "text": exact_answers,
                    "answer_start": [0] * len(question["exact_answer"])
                })
            else:
                dataset_dict["answers"].append({
                    "text": question["ideal_answer"],
                    "answer_start": [0]
                })
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset = dataset.train_test_split(test_size=0.8, seed=42)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']
        dataset = validation_dataset

    elif args.dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

        dataset = validation_dataset

    if args.dataset_name in ["bioasq", "svamp", "squad"]:
        unanswerable_indices = []
        val_answerable, val_unanswerable = split_dataset(dataset)
        del val_unanswerable
        if args.dataset_name in ["bioasq", "svamp", "squad", "nq", "trivia_qa"]:
            dataset = [dataset[i] for i in val_answerable]
            possible_indices = range(0, len(dataset))

            indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))

            dataset = [dataset[i] for i in indices]

    end_index_diy = min(args.num_samples, len(dataset))
    total_samples = end_index_diy

    if args.gene:
        if os.path.exists(save_one_pass):
            with open(save_one_pass, 'w') as f:
                pass
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

        is_gpt4_black_box = model_config["black_box"] == "gpt-4"

        if is_gpt4_black_box:
            print("\033[1;34m=== Using GPT-4 as blackbox model ===\033[0m")
            black_box_model = None
            black_box_tokenizer = None
            gpt4_answers = []
            if os.path.exists(gpt4_answer_file):
                with open(gpt4_answer_file, "r", encoding="utf-8") as f:
                    for line in f:
                        gpt4_answers.append(json.loads(line.strip())["answer"])
            else:
                os.makedirs(os.path.dirname(gpt4_answer_file), exist_ok=True)
                with open(gpt4_answer_file, "w", encoding="utf-8") as f:
                    for i in tqdm(range(0, total_samples), desc="Generating GPT-4 answers"):
                        question = dataset[i]['question']
                        black_box_prompt = get_prompt(black_box_prompt_type, question)
                        try:
                            client = openai.OpenAI(api_key=args.api_key, base_url=args.api_base)
                            completion = client.chat.completions.create(
                                model=args.api_model_name,
                                messages=[{"role": "user", "content": black_box_prompt}],
                                max_tokens=200,
                                temperature=args.temp,
                                top_p=args.top_p
                            )
                            generated_text = completion.choices[0].message.content
                            stop_phrases = ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >"]
                            clean_decoded = generated_text
                            for phrase in stop_phrases:
                                if phrase in clean_decoded:
                                    clean_decoded = clean_decoded.split(phrase)[0].strip()
                            gpt4_answers.append(clean_decoded)
                            f.write(json.dumps({"question": question, "answer": clean_decoded}, ensure_ascii=False) + "\n")
                        except Exception as e:
                            print(f"GPT-4 API error: {str(e)}")
                            gpt4_answers.append("")
                            f.write(json.dumps({"question": question, "answer": ""}, ensure_ascii=False) + "\n")
        else:
            black_box_model = load_model(model_config["black_box"], "Blackbox model")
            black_box_tokenizer = AutoTokenizer.from_pretrained(
                model_config["black_box"],
                trust_remote_code=True
            )

        if args.model_type == 'black':
            proxy_model = black_box_model
            proxy_tokenizer = black_box_tokenizer
            print("\033[1;33mUsing blackbox model as proxy\033[0m")
        elif args.model_type == 'sft':
            model_config_copy = model_config.copy()
            if "lora_adapter" in model_config_copy:
                model_config_copy["lora_adapter"] = model_config_copy["lora_adapter"].format(dataset_name=args.dataset_name)

            proxy_model = load_sft_model(model_config_copy)
            proxy_tokenizer = AutoTokenizer.from_pretrained(
                model_config["base_model"],
                trust_remote_code=True
            )
        elif args.model_type == 'base':
            proxy_model = load_model(model_config["base_model"], "Proxy model")
            proxy_tokenizer = AutoTokenizer.from_pretrained(
                model_config["base_model"],
                trust_remote_code=True
            )

        if not is_gpt4_black_box:
            newline_tokens = black_box_tokenizer.encode("\n", add_special_tokens=False)
            newline_token_id = newline_tokens[-1] if newline_tokens else black_box_tokenizer.eos_token_id
            period_token_id = [newline_token_id, black_box_tokenizer.eos_token_id]

            generation_config_one["eos_token_id"] = period_token_id
            generation_config_muti["eos_token_id"] = period_token_id

        print(f"\n\033[1;34m=== Starting answer generation ({total_samples} questions) ===\033[0m")

        begin_index = 0
        for i in tqdm(range(begin_index, total_samples), desc="Processing questions"):
            try:
                question_idx = i
                if hasattr(dataset, '__getitem__'):
                    question = dataset[question_idx]['question']
                else:
                    question = dataset[question_idx]['question']

                black_box_prompt = get_prompt(black_box_prompt_type, question)
                proxy_prompt = get_prompt(proxy_prompt_type, question)

                print(f"\n\033[1;35m=== Question {i+1}/{total_samples} ===\033[0m")
                print(f"Question: {question}")
                print(f"Blackbox prompt type: {black_box_prompt_type}")
                print(f"Proxy prompt type: {proxy_prompt_type}")

                if is_gpt4_black_box:
                    print(f"GPT-4 prompt: {black_box_prompt}")
                else:
                    encoded_input = black_box_tokenizer(black_box_prompt, return_tensors='pt')
                    input_ids = encoded_input.input_ids

                    min_token = torch.min(input_ids).item()
                    max_token = torch.max(input_ids).item()
                    print(f"Input token range: {min_token}-{max_token}, vocab size: {black_box_tokenizer.vocab_size}")

                    input_ids = torch.clamp(input_ids, 0, black_box_tokenizer.vocab_size - 1)
                    prompt = input_ids.to(black_box_model.device)

                if args.mode == 'one_pass':
                    if is_gpt4_black_box:
                        generated_text = gpt4_answers[i]
                        print(f"\033[1;32mRead GPT-4 answer: {generated_text}\033[0m")
                        stop_phrases = ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >"]
                        clean_decoded = generated_text
                        for phrase in stop_phrases:
                            if phrase in clean_decoded:
                                clean_decoded = clean_decoded.split(phrase)[0].strip()
                        print(f"\033[1;32mCleaned text: {clean_decoded}\033[0m")
                    else:
                        with torch.no_grad():
                            try:
                                print("\033[1;33mGenerating answer with blackbox model...\033[0m")
                                black_box_output = black_box_model.generate(prompt, **generation_config_one)
                            except RuntimeError as e:
                                print(f"Generation error: {str(e)}")
                                continue
                        input_length = prompt.shape[-1]
                        generated_tokens = black_box_output.sequences[0, input_length:]
                        generated_text = black_box_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        print(f"\033[1;32mBlackbox raw text: {generated_text}\033[0m")
                        stop_phrases = ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >"]
                        clean_decoded, clean_generated_tokens_length = process_decoded_str(generated_text, stop_phrases, black_box_tokenizer)
                        print(f"\033[1;32mCleaned text: {clean_decoded}\033[0m")

                    full_text = proxy_prompt + generated_text

                    full_input_ids = proxy_tokenizer(full_text, return_tensors='pt').input_ids.to(proxy_model.device)

                    prefix_ids = proxy_tokenizer(proxy_prompt, return_tensors='pt').input_ids
                    prefix_length = prefix_ids.shape[1]

                    labels = full_input_ids.clone()
                    labels[:, :prefix_length] = -100

                    print("\033[1;33mComputing logits with proxy model...\033[0m")
                    with torch.no_grad():
                        outputs = proxy_model(input_ids=full_input_ids, labels=labels)

                    logits = outputs.logits

                    raw_gen_len = full_input_ids.shape[1] - prefix_length
                    start_index = prefix_length - 1
                    end_index_raw = start_index + raw_gen_len

                    if end_index_raw > logits.shape[1]:
                        end_index_raw = logits.shape[1]

                    gen_logits = logits[0, start_index:end_index_raw, :]

                    logits_list = [gen_logits[i].unsqueeze(0) for i in range(gen_logits.shape[0])]

                    if is_gpt4_black_box:
                        clean_generated_tokens_length = len(proxy_tokenizer.encode(clean_decoded, add_special_tokens=False))

                    metric_dict, logit_dict = get_one_pass_metric(
                        logits_list,
                        clean_generated_tokens_length,
                        metrics,
                        get_eu,
                        topk
                    )

                    save2jsonl(clean_decoded, metric_dict, logit_dict, save_one_pass, i)

                elif args.mode == 'muti_pass':
                    metric_dict = {}
                    for gen_iter in range(args.num_gene):
                        if is_gpt4_black_box:
                            try:
                                print(f"\033[1;33mGenerating with GPT-4 API (sample {gen_iter+1}/{args.num_gene})...\033[0m")
                                client = openai.OpenAI(api_key=model_config.get("api_key", args.api_key),
                                                        base_url=model_config.get("api_base", args.api_base))

                                completion = client.chat.completions.create(
                                    model=model_config.get("api_model_name", args.api_model_name),
                                    messages=[{"role": "user", "content": black_box_prompt}],
                                    max_tokens=200,
                                    temperature=args.temp,
                                    top_p=args.top_p,
                                )

                                generated_text = completion.choices[0].message.content
                                print(f"\033[1;32mGPT-4 raw text (sample {gen_iter+1}): {generated_text}\033[0m")

                                stop_phrases = ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >"]
                                clean_decoded = generated_text
                                for phrase in stop_phrases:
                                    if phrase in clean_decoded:
                                        clean_decoded = clean_decoded.split(phrase)[0].strip()

                                print(f"\033[1;32mCleaned text (sample {gen_iter+1}): {clean_decoded}\033[0m")

                            except Exception as e:
                                print(f"GPT-4 API error: {str(e)}")
                                continue
                        else:
                            with torch.no_grad():
                                try:
                                    print(f"\033[1;33mGenerating with blackbox model (sample {gen_iter+1}/{args.num_gene})...\033[0m")
                                    black_box_output = black_box_model.generate(prompt, **generation_config_muti)
                                except RuntimeError as e:
                                    print(f"Generation error: {str(e)}")
                                    continue

                            input_length = prompt.shape[-1]
                            generated_tokens_raw = black_box_output.sequences[0, input_length:]
                            generated_text = black_box_tokenizer.decode(generated_tokens_raw, skip_special_tokens=True)

                            print(f"\033[1;32mBlackbox raw text (sample {gen_iter+1}): {generated_text}\033[0m")

                            stop_phrases = ["Answer the question concisely", "Q:", "\n\nQ:", "\nQ:", "\"Q:\"", "Home >"]
                            clean_decoded, clean_generated_tokens_length = process_decoded_str(generated_text, stop_phrases, black_box_tokenizer)

                            print(f"\033[1;32mCleaned text (sample {gen_iter+1}): {clean_decoded}\033[0m")

                        full_text = proxy_prompt + generated_text

                        full_input_ids = proxy_tokenizer(full_text, return_tensors='pt').input_ids.to(proxy_model.device)

                        prefix_ids = proxy_tokenizer(proxy_prompt, return_tensors='pt').input_ids
                        prefix_length = prefix_ids.shape[1]

                        labels = full_input_ids.clone()
                        labels[:, :prefix_length] = -100

                        print(f"\033[1;33mComputing logits with proxy model (sample {gen_iter+1})...\033[0m")
                        with torch.no_grad():
                            outputs = proxy_model(input_ids=full_input_ids, labels=labels)

                        logits = outputs.logits

                        raw_gen_len = full_input_ids.shape[1] - prefix_length
                        start_index = prefix_length - 1
                        end_index_raw = start_index + raw_gen_len

                        if end_index_raw > logits.shape[1]:
                            end_index_raw = logits.shape[1]

                        gen_logits = logits[0, start_index:end_index_raw, :]

                        gen_tokens = full_input_ids[0, prefix_length:]

                        logits_list = [gen_logits[i].unsqueeze(0) for i in range(gen_logits.shape[0])]

                        if is_gpt4_black_box:
                            clean_generated_tokens_length = len(proxy_tokenizer.encode(clean_decoded, add_special_tokens=False))

                        metric_dict = get_muti_pass_metric(
                            logits_list,
                            gen_tokens,
                            clean_generated_tokens_length,
                            question,
                            clean_decoded,
                            gen_iter,
                            metric_dict
                        )

                    save2json_SE(metric_dict, save_muti_pass, question_idx)

                torch.cuda.empty_cache()
            except Exception as e:
                print(f"\033[1;31mError processing question {i+1}/{total_samples}: {str(e)}\033[0m")


    elif args.generate_gt:
        if os.path.exists(save_eval_bleurt):
            with open(save_eval_bleurt, 'w') as f:
                pass

        if os.path.exists(save_merge_one_pass):
            with open(save_merge_one_pass, 'w') as f:
                pass

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
        model = BleurtForSequenceClassification.from_pretrained('BLEURT-20').cuda()
        tokenizer = BleurtTokenizer.from_pretrained('BLEURT-20')

        from evaluate import load
        rouge_evaluator = load("rouge")

        one_pass_data = load_jsonl_file(save_one_pass)
        print(f"Loaded one_pass_data length: {len(one_pass_data)}")
        print(f"end_index_diy: {end_index_diy}")

        actual_length = min(len(one_pass_data), end_index_diy)
        print(f"Actual data length: {actual_length}")

        for i in range(actual_length):
            question_id = one_pass_data[i].get("question_id", [])
            if args.dataset_name == 'tqa':
                best_answer = dataset[i]['best_answer']
                correct_answer = dataset[i]['correct_answers']
                all_answers = [best_answer] + correct_answer
                predictions = one_pass_data[i].get("answer", [])
                predictions = np.array([predictions], dtype=object)
                calculate_and_save_metrics(question_id, predictions, all_answers, save_eval_bleurt, model, tokenizer, rouge_evaluator)
            
            elif args.dataset_name == 'bioasq':
                all_answers = dataset[i]['answers']['text']
                predictions = one_pass_data[i].get("answer", [])
                predictions = np.array([predictions], dtype=object)
                calculate_and_save_metrics(question_id, predictions, all_answers, save_eval_bleurt, model, tokenizer, rouge_evaluator)

            elif args.dataset_name == 'trivia_qa':
                all_answers = dataset[i]['answers']['text']
                predictions = one_pass_data[i].get("answer", [])
                predictions = np.array([predictions], dtype=object)
                calculate_and_save_metrics(question_id, predictions, all_answers, save_eval_bleurt, model, tokenizer, rouge_evaluator)

        merge_jsonl_files(save_one_pass, save_eval_bleurt, save_merge_one_pass)

    elif args.generate_gt_llm:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
        Judge_model = 'mistralai/Mistral-7B-Instruct-v0.1'
        model = AutoModelForCausalLM.from_pretrained(Judge_model, low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                        device_map="auto", use_safetensors=False)
        tokenizer = AutoTokenizer.from_pretrained(Judge_model, trust_remote_code=True)

        from evaluate import load
        rouge_evaluator = load("rouge")

        length = len(dataset)
        length = end_index_diy
        one_pass_data = load_jsonl_file(save_one_pass)
        print(f"Loaded one_pass_data length: {len(one_pass_data)}")
        print(f"end_index_diy: {end_index_diy}")

        actual_length = min(len(one_pass_data), end_index_diy)
        print(f"Actual data length: {actual_length}")

        for i in range(actual_length):
            try:
                if args.dataset_name == 'tqa':
                    best_answer = dataset[i]['best_answer']
                    correct_answer = dataset[i]['correct_answers']
                    all_answers = [best_answer] + correct_answer
                    question = dataset[i]['question']
                elif args.dataset_name == 'triviaqa':
                    all_answers = dataset[i]['answer']['aliases']
                    question = dataset[i]['question']
                elif args.dataset_name == 'bioasq':
                    all_answers = dataset[i]['answers']['text']
                    question = dataset[i]['question']
                elif args.dataset_name == 'squad':
                    all_answers = dataset[i]['answers']['text']
                    question = dataset[i]['question']
                elif args.dataset_name == 'nq':
                    all_answers = dataset[i]['answer']
                    question = dataset[i]['question']
                elif args.dataset_name == 'trivia_qa':
                    all_answers = dataset[i]['answers']['text']
                    question = dataset[i]['question']
                elif args.dataset_name == 'svamp':
                    all_answers = dataset[i]['answer']
                    question = dataset[i]['question']
                    print(question)
                    print(all_answers)

                predictions = one_pass_data[i].get("answer", [])
                predictions = np.array([predictions], dtype=object)
                print(predictions)
                calculate_and_save_metrics_llm(i, question, predictions, all_answers, save_eval_llm, model, tokenizer, rouge_evaluator)
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue
        merge_jsonl_files(save_one_pass, save_eval_llm, save_merge_one_pass)

def split_dataset(dataset):
    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    assert set(answerable_indices) | set(unanswerable_indices) == set(range(len(dataset)))
    assert set(answerable_indices) - set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices

if __name__ == '__main__':
    seed_everything(42)
    main()
