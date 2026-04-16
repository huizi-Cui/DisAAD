import numpy as np
import datasets
import torch
import random
import argparse
import os
import json
from datasets import Dataset, load_dataset, load_from_disk
from model import load_tokenizer, load_model, load_black_model
import logging
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def save_data(output_file, args, data):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    args_file = f"{output_file}.args.json"
    with open(args_file, "w") as fout:
        json.dump(args.__dict__, fout, indent=4)
        print(f"Args written into {args_file}")

    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")

    dataset = Dataset.from_dict(data)
    arrow_file = f"{output_file}.arrow"
    dataset.save_to_disk(arrow_file)
    print(f"Data saved in arrow format to {arrow_file}")


class DataBuilder:
    def __init__(self, args):
        self.args = args

        if args.what_to_do == "local":
            self.base_tokenizer = load_tokenizer(args.base_model_name, args.dataset, args.cache_dir)
            if 'llama' in args.base_model_name:
                self.base_model = load_black_model(args.base_model_name)
            else:
                self.base_model = load_model(args.base_model_name, args.device, args.cache_dir, device_map='auto',use_quantization=True)

            self.base_model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.similarity_model = None

    def _filter_outliers(self, responses):
        if len(responses) < 12:
            return responses

        if self.similarity_model is None:
            self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(args.device)

        embeddings = self.similarity_model.encode(responses, convert_to_tensor=True)

        similarity_matrix = cosine_similarity(embeddings.cpu().numpy())

        avg_similarities = []
        for i in range(len(responses)):
            other_similarities = [sim for j, sim in enumerate(similarity_matrix[i]) if j != i]
            avg_similarities.append(np.mean(other_similarities))

        mean_sim = np.mean(avg_similarities)
        std_sim = np.std(avg_similarities)

        threshold = mean_sim - 1.5 * std_sim
        filtered_responses = [
            resp for resp, avg_sim in zip(responses, avg_similarities)
            if avg_sim >= threshold
        ]

        if len(filtered_responses) < 10:
            sorted_indices = np.argsort(avg_similarities)[::-1]
            filtered_responses = [responses[i] for i in sorted_indices[:3]]

        return filtered_responses

    def generate_training_samples(self, other_dataset_name="tqa"):
        print("Loading WildChat dataset...")
        wildchat_path = os.path.join(self.args.cache_dir, "wildchat")
        wildchat_dataset = load_from_disk(wildchat_path)
        if isinstance(wildchat_dataset, dict):
            wildchat_dataset = wildchat_dataset["train"]

        def filter_by_english(sample):
            return (
                sample["language"] == 'English' and
                50 < len(sample["conversation"][0]['content'].split()) < 300
            )
        wildchat_dataset = wildchat_dataset.filter(filter_by_english)
        print(f"WildChat dataset loaded, samples: {len(wildchat_dataset)}")

        other_dataset = None
        other_map_fn = None
        print(f"Loading {other_dataset_name} dataset...")

        if other_dataset_name == "tqa":
            other_dataset = load_dataset("truthful_qa", "generation", split="validation")
            other_map_fn = lambda sample: {"prompt": sample["question"], "source": "tqa"}
        else:
            raise ValueError(f"Unsupported dataset: {other_dataset_name}.")

        print(f"{other_dataset_name} dataset loaded, samples: {len(other_dataset)}")

        wildchat_samples = min(len(wildchat_dataset), self.args.wildchat_samples)
        other_samples = min(len(other_dataset), self.args.other_samples)

        print(f"Planned sampling: WildChat {wildchat_samples}, {other_dataset_name} {other_samples}")

        wildchat_dataset = wildchat_dataset.shuffle(seed=42).select(range(wildchat_samples))
        if hasattr(other_dataset, 'shuffle'):
            other_dataset = other_dataset.shuffle(seed=42).select(range(other_samples))
        else:
            import random
            idxs = list(range(len(other_dataset)))
            random.seed(42)
            random.shuffle(idxs)
            other_dataset = [other_dataset[i] for i in idxs[:other_samples]]
            from datasets import Dataset
            other_dataset = Dataset.from_list(other_dataset)

        def map_wildchat(sample):
            return {
                "prompt": sample["conversation"][0]['content'],
                "source": "wildchat"
            }
        wildchat_mapped = wildchat_dataset.map(map_wildchat, remove_columns=wildchat_dataset.column_names)

        if other_map_fn:
            other_mapped = other_dataset.map(other_map_fn, remove_columns=other_dataset.column_names)
        else:
            other_mapped = other_dataset

        combined_dataset = datasets.concatenate_datasets([wildchat_mapped, other_mapped])
        combined_dataset = combined_dataset.shuffle(seed=42)

        print(f"Combined dataset size: {len(combined_dataset)}")

        output_dir = self.args.ori_dataset_path
        os.makedirs(output_dir, exist_ok=True)
        combined_dataset.save_to_disk(output_dir)

        stats = {
            "total_samples": len(combined_dataset),
            "wildchat_samples": wildchat_samples,
            f"{other_dataset_name}_samples": other_samples
        }

        print(f"Dataset saved to {output_dir}")
        print(json.dumps(stats, indent=2))

        return output_dir

    def _has_repetition(self, text, max_repeat=3):
        words = text.split()
        for i in range(len(words) - max_repeat):
            if words[i:i+max_repeat] == words[i+max_repeat:i+2*max_repeat]:
                return True
        return False

    def _calculate_perplexity(self, text):
        inputs = self.base_tokenizer(text, return_tensors="pt", return_token_type_ids=False)
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.base_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return torch.exp(loss).item()

    def generate_text_from_prompts(self, prompts, min_length=50, max_new_tokens=128, num_generations=15):
        self.base_model.eval()
        all_responses = []

        pbar = tqdm(total=len(prompts), desc="Generating responses", unit="prompt")

        for prompt_idx, prompt in enumerate(prompts):
            prompt_responses = []

            low_temp_kwargs = {
                'temperature': 0.01,
                'do_sample': True,
                'top_k': None,
                'top_p': None
            }

            try:
                low_temp_encoded = self.base_tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_token_type_ids=False
                ).to(self.args.device)

                low_temp_outputs = self.base_model.generate(
                    **low_temp_encoded,
                    min_length=min_length,
                    max_new_tokens=max_new_tokens,
                    **low_temp_kwargs,
                    pad_token_id=self.base_tokenizer.eos_token_id,
                    eos_token_id=self.base_tokenizer.eos_token_id,
                    num_return_sequences=1
                )

                low_temp_text = self.base_tokenizer.decode(
                    low_temp_outputs[0],
                    skip_special_tokens=True
                )

                if low_temp_text.startswith(prompt):
                    low_temp_response = low_temp_text[len(prompt):].strip()
                else:
                    low_temp_response = low_temp_text

                prompt_responses.append(low_temp_response)
            except Exception as e:
                print(f"Low temp generation error: {str(e)}")
                prompt_responses.append("")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            high_temp_kwargs = {}
            if self.args.do_top_p:
                high_temp_kwargs['top_p'] = self.args.top_p
            elif self.args.do_top_k:
                high_temp_kwargs['top_k'] = self.args.top_k
            elif self.args.do_temperature:
                high_temp_kwargs['temperature'] = self.args.temperature
            else:
                high_temp_kwargs['temperature'] = 0.7
                high_temp_kwargs['top_p'] = 0.9

            batch_size = min(2, num_generations - 1)
            for i in range(0, num_generations - 1, batch_size):
                current_batch_size = min(batch_size, num_generations - 1 - i)

                repeated_prompts = [prompt] * current_batch_size

                try:
                    high_temp_encoded = self.base_tokenizer(
                        repeated_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_token_type_ids=False
                    ).to(self.args.device)

                    high_temp_outputs = self.base_model.generate(
                        **high_temp_encoded,
                        min_length=min_length,
                        max_new_tokens=max_new_tokens,
                        **high_temp_kwargs,
                        pad_token_id=self.base_tokenizer.eos_token_id,
                        eos_token_id=self.base_tokenizer.eos_token_id,
                        num_return_sequences=1
                    )

                    high_temp_texts = self.base_tokenizer.batch_decode(
                        high_temp_outputs,
                        skip_special_tokens=True
                    )

                    for full_text in high_temp_texts:
                        if full_text.startswith(prompt):
                            response = full_text[len(prompt):].strip()
                        else:
                            response = full_text
                        prompt_responses.append(response)

                except Exception as e:
                    print(f"High temp generation error: {str(e)}")
                    prompt_responses.extend([""] * current_batch_size)

                del high_temp_encoded
                del high_temp_outputs
                if 'high_temp_texts' in locals():
                    del high_temp_texts

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            all_responses.append(prompt_responses)

            if prompt_idx % 10 == 0:
                pbar.write(f"\nProgress: {prompt_idx+1}/{len(prompts)}")
                pbar.write(f"Current prompt: {prompt[:100]}...")
                pbar.write(f"Low temp response: {prompt_responses[0][:100]}...")
                if len(prompt_responses) > 1:
                    pbar.write(f"High temp response example: {prompt_responses[1][:100]}...")
                pbar.write(f"Response count for this prompt: {len(prompt_responses)}")

            pbar.update(1)

        pbar.close()
        return all_responses

    def generate_finetune_dataset(self, ori_dataset_path, sft_dataset_path, batch_size=2, num_samples=None):
        try:
            dataset = load_from_disk(ori_dataset_path)
            if isinstance(dataset, dict):
                dataset = dataset["train"]
            print(f"Dataset loaded, samples: {len(dataset)}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return

        print("Dataset columns:", dataset.column_names)

        def validate_sample(sample):
            return "prompt" in sample and sample["prompt"].strip()

        dataset = dataset.filter(validate_sample)
        print(f"Filtered dataset samples: {len(dataset)}")

        if len(dataset) == 0:
            print("Dataset empty after filtering, printing first 5 samples:")
            for i in range(min(5, len(dataset))):
                print(f"Sample {i}: {dataset[i]}")
            return

        prompts = [sample["prompt"] for sample in dataset]
        print(f"Extracted prompts: {len(prompts)}")

        if num_samples is not None and num_samples < len(prompts):
            prompts = prompts[:num_samples]
            print(f"Limited to {num_samples} prompts for generation")

        all_responses = []
        total_prompts = len(prompts)

        pbar = tqdm(total=total_prompts, desc="Generating responses", unit="prompt")
        batch_size = 1
        for i in range(0, total_prompts, batch_size):
            batch_prompts = prompts[i:i+batch_size]
            try:
                batch_responses = self.generate_text_from_prompts(
                    batch_prompts,
                    num_generations=10,
                    max_new_tokens=128
                )
                all_responses.extend(batch_responses)
                pbar.update(len(batch_prompts))

                if (i // batch_size) % 10 == 0:
                    sample_prompt = batch_prompts[0]
                    sample_low_temp = batch_responses[0][0]
                    sample_high_temp = batch_responses[0][1] if len(batch_responses[0]) > 1 else ""

                    pbar.write(f"\nProcessed {i+batch_size}/{total_prompts} prompts")
                    pbar.write(f"Example prompt: {sample_prompt[:200]}...")
                    pbar.write(f"Low temp response: {sample_low_temp[:100]}...")
                    if sample_high_temp:
                        pbar.write(f"High temp response: {sample_high_temp[:100]}...")
                    pbar.write(f"Response count for this prompt: {len(batch_responses[0])}")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error generating responses for prompt {i}: {str(e)}")
                all_responses.append([""] * 10)
                pbar.update(len(batch_prompts))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        pbar.close()

        filtered_prompts = []
        filtered_responses = []
        for prompt, responses in zip(prompts, all_responses):
            if len(responses) == 0:
                continue
            valid_responses = [responses[0]] if responses[0] else []
            for resp in responses[1:]:
                if len(resp.split()) < 15:
                    continue
                if self._has_repetition(resp, max_repeat=3):
                    continue
                if self._calculate_perplexity(resp) > 100:
                    continue
                valid_responses.append(resp)
            if len(valid_responses) >= 10:
                if len(valid_responses) > 1:
                    high_temp_responses = self._filter_outliers(valid_responses[1:])
                    valid_responses = [valid_responses[0]] + high_temp_responses
                filtered_prompts.append(prompt)
                filtered_responses.append(valid_responses[:10])

        finetune_data = {
            "sft_prompt": filtered_prompts,
            "sft_text": filtered_responses
        }
        total_prompts = len(filtered_prompts)
        total_responses = sum(len(resp_list) for resp_list in filtered_responses)
        avg_responses = total_responses / total_prompts if total_prompts > 0 else 0
        print("\nFinal dataset statistics:")
        print(f"Retained prompts: {total_prompts}")
        print(f"Retained responses: {total_responses}")
        print(f"Average responses per prompt: {avg_responses:.2f}")
        print(f"Low temp response ratio: 100% (at least 1 per prompt)")
        print(f"High temp response ratio: {(total_responses - total_prompts) / total_responses * 100:.1f}%")
        print(f"\nOutput dataset path: {sft_dataset_path}")
        save_data(sft_dataset_path, self.args, finetune_data)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return sft_dataset_path


def load_data(input_file):
    data_file = f"{input_file}.raw_data.json"
    with open(data_file, "r") as fin:
        data = json.load(fin)
        print(f"Raw data loaded from {data_file}")
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="tqa")
    parser.add_argument('--base_model_name', type=str, default="llama2-13b")
    parser.add_argument('--api_model_name', type=str, default="gpt-4-0613")
    parser.add_argument('--api_base', type=str, default="https://api.openai.com/v1")
    parser.add_argument('--api_key', type=str, default="", help="OpenAI API key")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--do_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--ori_dataset_path', type=str, default="datasets/wild_tqa_100/ori_data")
    parser.add_argument('--sft_dataset_path', type=str, default="datasets/wild_tqa_100/llama3.2-3b")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--wildchat_samples', type=int, default=50)
    parser.add_argument('--other_samples', type=int, default=50)
    parser.add_argument('--other_dataset_name', type=str, default="tqa")
    parser.add_argument('--what_to_do', type=str, default="mix")
    parser.add_argument('--gpuids', type=str, default="6 7")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuids
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_builder = DataBuilder(args)

    if args.what_to_do == "mix":
        data_builder.generate_training_samples(args.other_dataset_name)
    elif args.what_to_do == "local":
        data_builder.generate_finetune_dataset(args.ori_dataset_path, args.sft_dataset_path, args.batch_size, args.num_samples)
    else:
        raise ValueError(f"Invalid what_to_do: {args.what_to_do}")
    torch.cuda.empty_cache()
