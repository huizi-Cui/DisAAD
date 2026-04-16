import os
import torch
import json
import numpy as np
from evaluate import load
import traceback
import math

YELLOW = "\033[93m"
RESET = "\033[0m"

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def color_print(prefix: str, text: str) -> None:
    highlighted_text = f"{YELLOW}{prefix} {RESET}{text}"
    print(highlighted_text)

def convert_float32_to_float(data):
    if hasattr(data, 'dtype') and data.dtype in [np.float32, np.float64]:
        return float(data)
    elif isinstance(data, list):
        return [convert_float32_to_float(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_float32_to_float(v) for k, v in data.items()}
    else:
        return data
    
def load_jsonl_file(save_file_name):
    data = []
    with open(save_file_name, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save2jsonl(new_decoded, metric_dict, logit_dict, save_file_name, i):
    with open(save_file_name, 'a') as output_file:
        prob_list = metric_dict['prob']
        entropy_list = metric_dict['entropy']
        au_list = metric_dict['au']
        eu_list = metric_dict['eu']
        au_2_list = metric_dict['au_2']
        eu_2_list = metric_dict['eu_2']
        prob_list = convert_float32_to_float(prob_list)
        entropy_list = convert_float32_to_float(entropy_list)
        au_list = convert_float32_to_float(au_list)
        eu_list = convert_float32_to_float(eu_list)
        au_2_list = convert_float32_to_float(au_2_list)
        eu_2_list = convert_float32_to_float(eu_2_list)

        final_output = {
            'question_id': i,
            'answer': new_decoded,
            'prob': prob_list,
            'entropy': entropy_list,
            'au': au_list,
            'eu': eu_list,
            'au_2': au_2_list,
            'eu_2': eu_2_list,
            'Token_logit_dict': logit_dict
        }
        output_file.write(json.dumps(final_output, ensure_ascii=False) + '\n')

def save2json_SE(gen_data, save_file_name, question_id):
    """Save multi-round iteration results to a JSON file."""
    if os.path.exists(save_file_name):
        with open(save_file_name, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    question_entry = {
        'question_id': question_id,
        'generations': gen_data
    }
    existing_data[str(question_id)] = question_entry

    with open(save_file_name, 'w') as f:
        json.dump(existing_data, f, indent=2)

def calculate_and_save_metrics(idx, predictions, all_answers, save_file_name, model, tokenizer, rouge_evaluator=None):
    result = {"question_id": idx}
    
    if rouge_evaluator is None:
        rouge_evaluator = load("rouge")
    
    rouge_results = np.zeros((len(all_answers), len(predictions)))
    for anw in range(len(all_answers)):
        results = rouge_evaluator.compute(predictions=predictions, references=[all_answers[anw]] * len(predictions), use_aggregator=False)
        if results and 'rougeL' in results:
            rouge_results[anw] = results['rougeL']
        else:
            rouge_results[anw] = 0.0
    result["rouge"] = np.max(rouge_results, axis=0).tolist()
    all_answers_bleurt = [answer if answer.endswith('.') else answer + '.' for answer in all_answers]
    model.eval()

    predictions_str = predictions.tolist()[0]
    with torch.no_grad():
        inputs = tokenizer(
            all_answers_bleurt,
            [predictions_str] * len(all_answers_bleurt),
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        
        scores = model(**inputs).logits.flatten()
        scores = scores.cpu().tolist()
    
    result["bleurt"] = max(scores)
    
    try:
        align_scorer = AlignScoreEvaluator(
            ckpt_path="prithivida/AlignScore-large",
            batch_size=len(all_answers),
            target_is_claims=False
        )

        replicated_predictions = np.array([predictions_str] * len(all_answers))
        stats = {'greedy_texts': replicated_predictions, 'input_texts': []}

        align_scores_array = align_scorer(stats=stats, target_texts=all_answers)

        result["alignscore"] = float(np.max(align_scores_array))

    except ImportError:
        print("AlignScore not installed or found. Skipping AlignScore calculation.")
        result["alignscore"] = 0.0
    except Exception as e:
        print(f"Error during AlignScore calculation: {e}")
        traceback.print_exc()
        result["alignscore"] = 0.0
    
    with open(save_file_name, 'a') as f:
        f.write(json.dumps(result) + '\n')
        
def calculate_and_save_metrics_llm(idx, question, predictions, all_answers, save_file_name, model, tokenizer, rouge_evaluator=None):
    result = {"question_id": idx}
    
    if rouge_evaluator is None:
        rouge_evaluator = load("rouge")
    
    rouge_results = np.zeros((len(all_answers), len(predictions)))
    for anw in range(len(all_answers)):
        results = rouge_evaluator.compute(predictions=predictions, references=[all_answers[anw]] * len(predictions), use_aggregator=False)
        if results and 'rougeL' in results:
            rouge_results[anw] = results['rougeL']
        else:
            rouge_results[anw] = 0.0
    result["rouge"] = np.max(rouge_results, axis=0).tolist()

    model.eval()
    predictions = predictions.tolist()[0]
    
    messages = [
        {"role": "system", "content": "Your task is to determine if the provided answer is true or false based solely on the ground truth answers given to you in the format ['s answer 1', 'answer 2', ...]. DO NOT rely on your memory; only use the information provided after this instruction. Respond with 1 if the predicted answer is correct, which means semantically consistent with any of the ground truth answers, otherwise respond with 0. Respond with just 0 or 1, and DO NOT include anything else in your response. This is the only instruction you need to follow."},
        {
            "role": "user",
            "content": "Input: Who is elected as the vice president of india in 2017?\nGround Truth: ['Venkaiah Naidu', 'Muppavarapu Venkaiah Naidu']\nProvided Answer: M. Venkaiah Naidu"
        },
        {"role": "assistant", "content": "1"},
        {
            "role": "user", 
            "content": "Input: who sings you are a magnet and i am steel?\nGround Truth: ['Walter Egan']\nProvided Answer: The song 'You Are a Magnet and I Am Steel' is performed by the band The 1975."
        },
        {"role": "assistant", "content": "0"}
    ]

    current_prompt = {
        "role": "user",
        "content": f"Input: {question}\nGround Truth: {all_answers}\nProvided Answer: {predictions}"
    }
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages + [current_prompt],
        tokenize=False,
        add_generation_prompt=True
    )

    with torch.no_grad():
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        decoded_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        input_len = inputs.input_ids.shape[-1]
        judge_result = 0
        clean_decoded_output = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)
        last_response = decoded_output.split("assistant\n")[-1].strip()
        
        if "1" in last_response:
            judge_result = 1
        elif "0" in last_response:
            judge_result = 0
        else:
            print(f"Abnormal output: {decoded_output}")
            judge_result = 0

    result["llm_judge"] = judge_result

    with open(save_file_name, 'a') as f:
        f.write(json.dumps(result) + '\n')

def process_decoded_str(new_decoded, stop_phrases, tokenizer):
    """Process the decoded string and truncate by stop phrases"""
    min_pos = None
    
    if "<think>" in new_decoded:
        think_start = new_decoded.find("<think>")
        think_end = new_decoded.find("</think>")
        
        if think_end != -1:
            new_decoded = new_decoded[:think_start] + new_decoded[think_end + 8:]
    
    new_decoded = new_decoded.replace("<think>", "").replace("</think>", "")
    new_decoded = new_decoded.replace("\n\n", "\n").strip()
    
    for phrase in stop_phrases:
        pos = new_decoded.find(phrase)
        if pos != -1:
            if (min_pos is None) or (pos < min_pos):
                min_pos = pos

    min_pos = min_pos if min_pos is not None else len(new_decoded)
    clean_decoded = new_decoded[:min_pos] if min_pos is not None else new_decoded

    clean_tokens = tokenizer.encode(clean_decoded, add_special_tokens=False)
    return clean_decoded, len(clean_tokens)

def get_one_pass_metric(logits, clean_generated_tokens_length, metrics, get_eu, topk):
    """Process logits and calculate various metrics"""
    metric_dict = {}

    sequence_length = len(logits)
    for idx_l in range(min(clean_generated_tokens_length, sequence_length)):
        logit = logits[idx_l]
        logit = logit.to(torch.float32).cpu().numpy()
        for metric, k in metrics:
            if metric not in metric_dict:
                metric_dict[metric] = []
            eu = get_eu(metric, k)
            metric_dict[metric].append(eu(logit[0]))

    logit_dict = {}
    logit_start_idx = 0
    logit_end_idx = min(clean_generated_tokens_length, sequence_length)
    ii = 0

    for idx_ll in range(logit_start_idx, logit_end_idx):
        logit = logits[idx_ll]
        logit = logit.to(torch.float32).cpu().numpy()  
        
        top_k = 10
        top_values, top_indices = topk(logit[0], top_k)
        
        logit_dict[ii] = {'top_values': top_values, 'top_indices': top_indices}
        ii += 1

    for key in logit_dict:
        logit_dict[key]['top_values'] = logit_dict[key]['top_values'].tolist()
        logit_dict[key]['top_indices'] = logit_dict[key]['top_indices'].tolist()

    return metric_dict, logit_dict

def get_muti_pass_metric(logits, clean_generated_tokens, clean_generated_tokens_length, question, clean_decoded, gen_iter, metric_dict):
    """Calculate negative log likelihood and update results"""
    log_likelihood = 0.0
    sequence_length = len(logits)
    logit_end_idx = min(clean_generated_tokens_length, sequence_length)
    for step in range(logit_end_idx):
        cur_logits = logits[step]
        log_probs = torch.log_softmax(cur_logits, dim=-1)
        next_token_id = clean_generated_tokens[step]
        log_likelihood += log_probs[0, next_token_id].item()

    nll = -log_likelihood
    ln_nll = nll / clean_generated_tokens_length if clean_generated_tokens_length > 0 else 0.0

    metric_dict[gen_iter] = {
        "text": question + " " + clean_decoded,
        "nll": nll,
        "ln_nll": ln_nll,
        "ans": clean_decoded,
        "ques": question
    }
        
    return metric_dict

def merge_jsonl_files(file1, file2, output_file):
    """Merge two JSONL files by combining data based on question_id"""
    data1 = {}
    with open(file1, 'r') as f:
        for line in f:
            item = json.loads(line)
            data1[item['question_id']] = item

    data2 = {}
    with open(file2, 'r') as f:
        for line in f:
            item = json.loads(line)
            data2[item['question_id']] = item

    merged_data = {}
    for question_id in set(data1.keys()).union(data2.keys()):
        merged_item = {"question_id": question_id}
        if question_id in data1:
            merged_item.update(data1[question_id])
        if question_id in data2:
            merged_item.update(data2[question_id])
        merged_data[question_id] = merged_item

    with open(output_file, 'w') as f:
        for question_id in merged_data:
            f.write(json.dumps(merged_data[question_id]) + '\n')

    print(f"Merge complete. Result saved to {output_file}")

def merge_jsonl_files_se(file1, file2, output_file):
    """Merge two JSONL files with integer question_id"""
    data1 = {}
    with open(file1, 'r') as f:
        for line in f:
            item = json.loads(line)
            item["question_id"] = int(item["question_id"])
            data1[item['question_id']] = item

    data2 = {}
    with open(file2, 'r') as f:
        for line in f:
            item = json.loads(line)
            item["question_id"] = int(item["question_id"]) if isinstance(item["question_id"], str) else item["question_id"]
            data2[item['question_id']] = item

    merged_data = {}
    for question_id in set(data1.keys()).union(data2.keys()):
        merged_item = {"question_id": question_id}
        if question_id in data1:
            merged_item.update(data1[question_id])
        if question_id in data2:
            merged_item.update(data2[question_id])
        merged_data[question_id] = merged_item

    with open(output_file, 'w') as f:
        for question_id in sorted(merged_data.keys()):
            f.write(json.dumps(merged_data[question_id]) + '\n')

    print(f"Merge complete. Result saved to {output_file}")

def save_one_pass_entry(file_path, index, answer, metric_dict, logit_dict):
    """Save one_pass mode entry"""
    
    entry = {
        "question_id": index,
        "answer": answer,
        "prob": [metric_dict.get("prob", 0.0)],
        "entropy": [metric_dict.get("entropy", 0.0)],
        "au": [metric_dict.get("au", 0.0)],
        "eu": [metric_dict.get("eu", 0.0)],
        "au_2": [metric_dict.get("au_2", 0.0)],
        "eu_2": [metric_dict.get("eu_2", 0.0)],
        "Token_logit_dict": logit_dict
    }
    
    entry = convert_float32_to_float(entry)
    
    print(f"{'='*40}\n")
    with open(file_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def save_muti_pass_entry(file_path, entry):
    """Save multi-pass mode entry"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    def convert_floats(obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(x) for x in obj]
        else:
            return obj
            
    entry = convert_floats(entry)
    
    with open(file_path, 'a') as f:
        json_line = json.dumps(entry, ensure_ascii=False)
        f.write(json_line + '\n')