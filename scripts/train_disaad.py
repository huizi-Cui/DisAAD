import argparse
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    AutoModel
)
from peft import (
    LoraConfig,
    get_peft_model,
)
import torch
from datasets import Dataset
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import json
import datetime
import os
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import re
from diff_match_patch import diff_match_patch
from model import load_model, load_tokenizer

def filter_by_english_and_llama3_mistral(sample):
    return len(sample["conversation"][0]["content"]) < 480

def filter_by_english_and_version_3_5(sample):
    return sample["language"] == 'English' \
            and sample["model"] == "gpt-3.5-turbo" and len(sample["conversation"][0]["content"]) < 480

def filter_by_english_and_version(sample):
    return sample["language"] == 'English' and  sample["timestamp"].month <= 6 \
            and sample["model"] == "gpt-4" and len(sample["conversation"][0]["content"]) < 512

def filter_claude3(sample):
    return len(sample["conversation"][0]["content"]) < 1024

def filter_llama(sample):
    return len(sample["sft_prompt"]) < 4096

filter_dict = {
    'ChatGPT': filter_by_english_and_version_3_5,
    'GPT-4': filter_by_english_and_version,
    'Claude-3': filter_claude3,
    'llama3.1-8b': filter_llama,
    'llama2-7b': filter_llama
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_answer(prompt, text):
    matches = [m.start() for m in re.finditer(re.escape(prompt.strip()), text)]
    if matches:
        return text[matches[-1]+len(prompt):].lstrip()

    dmp = diff_match_patch()
    patches = dmp.patch_make(prompt, text)
    if len(patches) > 0:
        return dmp.patch_apply(patches, text)[0]

    return text

def build_sft_dataset(json_path, train_num_sample=None, val_num_sample=None):
    if not json_path:
        raise ValueError("json_path parameter is required")

    print(f"Loading pre-generated blackbox dataset: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    prompts = data_dict["sft_prompt"]
    responses = data_dict["sft_text"]

    samples = []
    for i in range(len(prompts)):
        if not responses[i] or len(responses[i]) == 0:
            continue
        sample = {
            "sft_prompt": prompts[i],
            "low_temp_answer": responses[i][0],
            "blackbox_answers": responses[i][1:]
        }
        samples.append(sample)

    total_samples = len(samples)
    val_start_index = total_samples - val_num_sample if val_num_sample else 0

    train_data = samples[:train_num_sample] if train_num_sample else samples
    val_data = samples[val_start_index:val_start_index+val_num_sample] if val_num_sample else []

    print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")

    train_dataset = Dataset.from_dict({
        'sft_prompt': [item['sft_prompt'] for item in train_data],
        'blackbox_answers': [item['blackbox_answers'] for item in train_data]
    })

    val_dataset = Dataset.from_dict({
        'sft_prompt': [item['sft_prompt'] for item in val_data],
        'blackbox_answer': [item['low_temp_answer'] for item in val_data]
    })

    return {
        "train": train_dataset,
        "val": val_dataset
    }


class SimilarityLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
        self.cos = nn.CosineSimilarity(dim=1)
        self.device = device

    def forward(self, generated, references_list):
        if not generated or not references_list:
            return torch.tensor(0.0, device=self.device)

        gen_emb = self.sim_model.encode(generated, convert_to_tensor=True)

        total_sim = 0.0
        valid_count = 0

        for i in range(len(generated)):
            gen_emb_i = gen_emb[i].unsqueeze(0)
            refs = references_list[i]

            if len(refs) == 0:
                continue

            ref_embs = self.sim_model.encode(refs, convert_to_tensor=True)
            expanded_gen = gen_emb_i.expand_as(ref_embs)
            sims = self.cos(expanded_gen, ref_embs)
            total_sim += torch.mean(sims)
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=self.device)

        avg_sim = total_sim / valid_count
        return 1 - avg_sim


class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.model_dir = os.path.join(log_dir, "saved_models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.train_log = os.path.join(log_dir, "training_log.jsonl")
        self.val_log = os.path.join(log_dir, "validation_log.jsonl")
        self.train_console = os.path.join(log_dir, "train_console.log")
        self.val_console = os.path.join(log_dir, "val_console.log")
        self.best_metrics_file = os.path.join(log_dir, "best_metrics.json")

        self.best_metrics = {
            "semantic_sim": 0.0,
            "pred_gap": float("inf"),
            "epoch": -1
        }

        self.plot_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def log_train(self, epoch, batch, metrics):
        converted_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.generic):
                converted_metrics[k] = v.item()
            elif isinstance(v, torch.Tensor):
                converted_metrics[k] = v.item()
            else:
                converted_metrics[k] = v

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": epoch,
            "batch": batch,
            **converted_metrics
        }
        with open(self.train_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_validation(self, epoch, metrics):
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": epoch,
            **metrics
        }
        log_str = f"[Epoch {epoch}] Validation - "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        with open(self.val_console, "a") as f:
            f.write(log_str + "\n")
        print(log_str)

    def update_best_metrics(self, metrics, epoch, model):
        current_gap = float(metrics["pred_gap"])
        semantic_sim = float(metrics["semantic_sim"])

        if (semantic_sim > self.best_metrics["semantic_sim"]) or \
           (semantic_sim == self.best_metrics["semantic_sim"] and \
            current_gap < self.best_metrics["pred_gap"]):

            self.best_metrics = {
                "semantic_sim": semantic_sim,
                "pred_gap": current_gap,
                "epoch": epoch
            }
            self.save_model(model, epoch, is_best=True)
            self.save_best_metrics()

    def save_best_metrics(self):
        def convert(o):
            if isinstance(o, np.generic):
                return o.item()
            return o

        with open(self.best_metrics_file, "w") as f:
            json.dump(self.best_metrics, f, default=convert)

    def log_samples(self, epoch, batch, prompts, gens, reals):
        entry = {
            "epoch": epoch,
            "batch": batch,
            "samples": [
                {
                    "prompt": p[:200] + "..." if len(p) > 200 else p,
                    "generated": g[:300] + "..." if len(g) > 300 else g,
                    "real": r[:300] + "..." if len(r) > 300 else r
                } for p, g, r in zip(prompts, gens, reals)
            ]
        }
        with open(self.sample_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_model(self, model, epoch, is_best=False):
        model_to_save = model.module if hasattr(model, 'module') else model
        save_path = os.path.join(self.model_dir, f"epoch_{epoch}")
        model_to_save.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")

        if is_best:
            best_path = os.path.join(self.model_dir, "best_model")
            model_to_save.save_pretrained(best_path)
            print(f"Best model updated to: {best_path}")


class TextDiscriminator(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained("gpt2")
        self.encoder = AutoModel.from_pretrained(
            "gpt2",
            torch_dtype=torch.bfloat16,
            config=config
        ).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(config.n_embd, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device, torch.bfloat16)

        for layer in self.encoder.h[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.encoder(**inputs)
        last_hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(last_hidden)


def print_model_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        def format_num(num):
            if num >= 1e9:
                return f"{num/1e9:.2f}B"
            elif num >= 1e6:
                return f"{num/1e6:.2f}M"
            elif num >= 1e3:
                return f"{num/1e3:.2f}K"
            return f"{num}"

        print("\n" + "="*50)
        print(f"Proxy Model Parameters: {args.scoring_model_name}")
        print("="*50)
        print(f"Total parameters: {format_num(total_params)} ({total_params})")
        print(f"Trainable parameters: {format_num(trainable_params)} ({trainable_params})")
        print(f"Non-trainable parameters: {format_num(non_trainable_params)} ({non_trainable_params})")
        print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")
        print("="*50 + "\n")


def train(args):
    set_seed(args.seed)

    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    max_memory_mapping = {i: "23GiB" for i in range(num_gpus)}
    max_memory_mapping['cpu'] = "32GiB"
    logger = TrainingLogger(os.path.join(args.output_path, args.scoring_model_name, "logs"))

    tokenizer = load_tokenizer(args.scoring_model_name, args.train_dataset_name, args.cache_dir)
    cutoff_len = 1024

    discriminator = TextDiscriminator(device=args.device)
    if torch.cuda.device_count() > 1:
        discriminator = nn.DataParallel(discriminator)
    d_optimizer = AdamW(discriminator.parameters(), lr=1e-5, weight_decay=0.01)

    model = load_model(
        model_name=args.scoring_model_name,
        device="cpu",
        cache_dir=args.cache_dir,
        device_map="auto",
        max_memory=max_memory_mapping,
        low_cpu_mem_usage=True,
        use_quantization=False
    )

    modules = {
        "llama3.1-8b":["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "llama3.2-1b":["q_proj", "v_proj","k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "llama3.2-3b":["q_proj", "v_proj","k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "llama2-7b":["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "qwen2.5-7b":["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "qwen2.5-3b":["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "mistral-7b":["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "Gpt-neo":["q_proj", "v_proj", "k_proj", "out_proj", "c_fc", " c_proj"]
    }

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=modules[args.scoring_model_name],
        fan_in_fan_out=False,
        lora_dropout=0.1,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print_model_parameters(model)

    model.prepare_inputs_for_generation = model.base_model.prepare_inputs_for_generation
    model = model.to(torch.bfloat16)

    g_optimizer = AdamW(model.parameters(), lr=1e-4)

    def manual_collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
            'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
            'labels': torch.stack([torch.tensor(x['labels']) for x in batch]),
            'sft_prompt': [x['sft_prompt'] for x in batch],
            'blackbox_answer': [x['blackbox_answer'] for x in batch]
        }

    def custom_train_loop():
        sim_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(args.device)
        sim_model.eval()
        sim_criterion = SimilarityLoss(device=args.device)

        sft_data = build_sft_dataset(
            json_path=args.train_dataset_path,
            train_num_sample=args.train_num_samples,
            val_num_sample=args.val_num_samples
        )

        train_dataset = sft_data["train"]
        val_dataset = sft_data["val"]

        print(f"Training samples: {len(train_dataset)} (each sample contains {len(train_dataset[0]['blackbox_answers'])} blackbox answers)")
        print(f"Validation samples: {len(val_dataset)}")

        def generate_llama3_input_and_tokenize(batch, is_training=True):
            prompts = []
            target_answers = []
            all_answers = []

            if is_training:
                for p, ans_list in zip(batch["sft_prompt"], batch["blackbox_answers"]):
                    if not ans_list:
                        continue
                    p_strip = p.strip()
                    for a in ans_list:
                        a_strip = a.strip()
                        prompts.append(p_strip)
                        target_answers.append(a_strip)
                        all_answers.append(ans_list)
            else:
                for p, a in zip(batch["sft_prompt"], batch["blackbox_answer"]):
                    p_strip = p.strip()
                    a_strip = a.strip()
                    prompts.append(p_strip)
                    target_answers.append(a_strip)
                    all_answers.append([a_strip])

            if not prompts:
                return {
                    "input_ids": torch.empty((0, cutoff_len), dtype=torch.long),
                    "attention_mask": torch.empty((0, cutoff_len), dtype=torch.long),
                    "labels": torch.empty((0, cutoff_len), dtype=torch.long),
                    "sft_prompt": [],
                    "target_answer": [],
                    "all_answers": [],
                }

            full_texts = [f"{p}{a}" for p, a in zip(prompts, target_answers)]

            inputs = tokenizer(
                full_texts,
                max_length=cutoff_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
                "sft_prompt": prompts,
                "target_answer": target_answers,
                "all_answers": all_answers,
            }

        train_tokenized = train_dataset.map(
            lambda batch: generate_llama3_input_and_tokenize(batch, is_training=True),
            batched=True,
            remove_columns=train_dataset.column_names if hasattr(train_dataset, "column_names") else list(train_dataset.keys()),
        )

        val_tokenized = val_dataset.map(
            lambda batch: generate_llama3_input_and_tokenize(batch, is_training=False),
            batched=True,
            remove_columns=val_dataset.column_names if hasattr(val_dataset, "column_names") else list(val_dataset.keys()),
        )

        def manual_collate_fn(batch):
            return {
                'input_ids': torch.stack([torch.tensor(x['input_ids']) for x in batch]),
                'attention_mask': torch.stack([torch.tensor(x['attention_mask']) for x in batch]),
                'labels': torch.stack([torch.tensor(x['labels']) for x in batch]),
                'sft_prompt': [x['sft_prompt'] for x in batch],
                'target_answer': [x['target_answer'] for x in batch],
                'all_answers': [x['all_answers'] for x in batch]
            }

        train_loader = torch.utils.data.DataLoader(
            train_tokenized,
            batch_size=2,
            collate_fn=manual_collate_fn,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_tokenized,
            batch_size=2,
            collate_fn=manual_collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        for epoch in range(args.epochs):
            model.train()
            discriminator.train()
            epoch_sim_scores = []
            epoch_all_sim_scores = []

            print(f"\n{'='*50}\nEpoch {epoch+1}/{args.epochs}\n{'='*50}")

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                torch.cuda.empty_cache()

                prompts = batch['sft_prompt']
                target_answers = batch['target_answer']
                all_answers = batch['all_answers']

                inputs = {
                    k: v.to(args.device)
                    for k, v in batch.items()
                    if k in ['input_ids', 'attention_mask', 'labels']
                }

                model.eval()
                gen_full_texts = generate_texts(
                    model,
                    tokenizer,
                    inputs["input_ids"],
                    inputs["attention_mask"]
                )
                model.train()

                gen_answers = [extract_answer(p, gen) for p, gen in zip(prompts, gen_full_texts)]

                d_losses = []
                for _ in range(2):
                    d_optimizer.zero_grad()

                    real_full_texts = [p + a for p, a in zip(prompts, target_answers)]
                    real_preds = discriminator(real_full_texts)
                    d_real_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))

                    fake_full_texts = [p + a for p, a in zip(prompts, gen_answers)]
                    fake_preds = discriminator(fake_full_texts)
                    d_fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))

                    d_loss = (d_real_loss + d_fake_loss) / 2
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    d_optimizer.step()
                    d_losses.append(d_loss.item())

                g_optimizer.zero_grad()

                outputs = model(**{k:v for k,v in inputs.items() if k != 'text'})
                task_loss = outputs.loss.mean()

                current_preds = discriminator(fake_full_texts)
                adv_loss = -torch.log(current_preds + 1e-8).mean()

                sim_loss = sim_criterion(gen_answers, all_answers)

                total_loss = 1 * task_loss + 1 * adv_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                g_optimizer.step()

                with torch.no_grad():
                    gen_emb = sim_model.encode(gen_answers, convert_to_tensor=True)
                    target_emb = sim_model.encode(target_answers, convert_to_tensor=True)
                    batch_sims = F.cosine_similarity(gen_emb, target_emb).cpu().numpy()
                    avg_sim = np.mean(batch_sims)

                all_sims = []
                for i in range(len(gen_answers)):
                    if not all_answers[i]:
                        continue

                    gen_emb_i = gen_emb[i].unsqueeze(0)
                    ref_embs = sim_model.encode(all_answers[i], convert_to_tensor=True)
                    expanded_gen = gen_emb_i.expand_as(ref_embs)
                    sims = F.cosine_similarity(expanded_gen, ref_embs)
                    all_sims.append(torch.mean(sims).item())

                all_avg_sim = np.mean(all_sims) if all_sims else 0.0
                epoch_sim_scores.append(avg_sim)
                epoch_all_sim_scores.append(all_avg_sim)

                if batch_idx % 2 == 0:
                    print(f"\n[Batch {batch_idx}]")
                    print(f"Prompt: {prompts[0][:100]}...")
                    print(f"Generated: {gen_answers[0][:400]}...")
                    print(f"Target: {target_answers[0][:400]}...")
                    print(f"Reference count: {len(all_answers[0])}")
                    print(f"Current similarity: {avg_sim:.4f}")
                    print(f"Average semantic similarity: {all_avg_sim:.4f}")

                    log_str = (
                        f"D Loss: {np.mean(d_losses):.4f} (Real: {real_preds.mean().item():.3f} | Fake: {fake_preds.mean().item():.3f})\n"
                        f"G Loss: {total_loss.item():.4f} [Task: {task_loss.item():.4f} | Adv: {adv_loss.item():.4f} | Sim: {sim_loss.item():.4f}]"
                    )
                    print(log_str)

                if batch_idx % 10 == 0:
                    metrics = {
                        "d_loss": np.mean(d_losses),
                        "d_real": real_preds.mean().item(),
                        "d_fake": fake_preds.mean().item(),
                        "g_total_loss": total_loss.item(),
                        "g_task_loss": task_loss.item(),
                        "g_adv_loss": adv_loss.item(),
                        "g_sim_loss": sim_loss.item(),
                        "semantic_sim": avg_sim,
                        "avg_semantic_sim": all_avg_sim
                    }
                    logger.log_train(epoch+1, batch_idx, metrics)

            print(f"\n{'='*30}\nEpoch {epoch+1} Average Statistics:")
            print(f"Target similarity: {np.mean(epoch_sim_scores):.4f}")
            print(f"All answers similarity: {np.mean(epoch_all_sim_scores):.4f}")
            print(f"{'='*30}")

            val_metrics = validate(model, discriminator, tokenizer, val_loader, epoch, sim_model)
            logger.log_validation(epoch+1, val_metrics)
            logger.save_model(model, epoch+1)
            logger.update_best_metrics(val_metrics, epoch+1, model)

    def validate(gen_model, discriminator, tokenizer, val_loader, epoch, sim_model):
        if hasattr(gen_model, 'parameters'):
            device = next(gen_model.parameters()).device
        else:
            device = torch.device("cuda")
        gen_model.eval()
        discriminator.eval()

        all_prompts = []
        all_gen_texts = []
        all_blackbox_texts = []

        print(f"\n{'='*50}\nEpoch {epoch+1} Validation\n{'='*50}")

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Generating validation texts"):
                prompts = batch['sft_prompt']
                blackbox_answers = batch['target_answer']

                inputs = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in ['input_ids', 'attention_mask', 'labels']
                }

                gen_full_texts = generate_texts(
                    gen_model,
                    tokenizer,
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    max_new_tokens=256
                )

                gen_answers = [extract_answer(p, gen)[:512] for p, gen in zip(prompts, gen_full_texts)]
                blackbox_answers = [a[:512] for a in blackbox_answers]

                all_prompts.extend(prompts)
                all_gen_texts.extend(gen_answers)
                all_blackbox_texts.extend(blackbox_answers)

                del inputs, gen_full_texts
                torch.cuda.empty_cache()

        batch_size = 2
        real_preds = []
        fake_preds = []

        for i in tqdm(range(0, len(all_blackbox_texts), batch_size), desc="Discriminating blackbox samples"):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_answers = all_blackbox_texts[i:i+batch_size]

            batch_texts = [
                f"{p}{a}"[:1024]
                for p, a in zip(batch_prompts, batch_answers)
            ]

            with torch.cuda.amp.autocast():
                batch_preds = discriminator(batch_texts)

            real_preds.append(batch_preds.mean().item())

            del batch_texts, batch_preds
            torch.cuda.empty_cache()

        for i in tqdm(range(0, len(all_gen_texts), batch_size), desc="Discriminating generated samples"):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_answers = all_gen_texts[i:i+batch_size]

            batch_texts = [
                f"{p}{a}"[:1024]
                for p, a in zip(batch_prompts, batch_answers)
            ]

            with torch.cuda.amp.autocast():
                batch_preds = discriminator(batch_texts)

            fake_preds.append(batch_preds.mean().item())

            del batch_texts, batch_preds
            torch.cuda.empty_cache()

        real_pred_mean = np.mean(real_preds)
        fake_pred_mean = np.mean(fake_preds)

        print("\nCalculating semantic similarity...")
        semantic_sims = []
        for i in range(0, len(all_gen_texts), 8):
            batch_gen = all_gen_texts[i:i+8]
            batch_blackbox = all_blackbox_texts[i:i+8]

            with torch.no_grad():
                gen_emb = sim_model.encode(batch_gen, convert_to_tensor=True)
                blackbox_emb = sim_model.encode(batch_blackbox, convert_to_tensor=True)
                batch_sims = F.cosine_similarity(gen_emb, blackbox_emb).cpu().numpy()

            semantic_sims.extend(batch_sims)
            del gen_emb, blackbox_emb

        plt.figure(figsize=(12, 7), dpi=120)

        mean_sim = np.mean(semantic_sims)
        median_sim = np.median(semantic_sims)
        std_sim = np.std(semantic_sims)
        max_sim = np.max(semantic_sims)
        min_sim = np.min(semantic_sims)

        plot_dir = os.path.join(logger.log_dir, "semantic_sim_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"epoch_{epoch+1}.png")

        n, bins, patches = plt.hist(
            semantic_sims,
            bins=20,
            range=(0, 1),
            color='#2c7fb8',
            edgecolor='white',
            linewidth=1.2,
            alpha=0.85,
            density=False
        )

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for count, x in zip(n, bin_centers):
            if count > 0:
                plt.text(
                    x, count + 0.5,
                    f'{int(count)}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='#2c7fb8',
                    fontweight='bold'
                )

        plt.axvline(mean_sim, color='#e41a1c', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.2f}')
        plt.axvline(median_sim, color='#4daf4a', linestyle='-.', linewidth=2, label=f'Median: {median_sim:.2f}')

        stats_text = f'''
        Statistics:
        Max: {max_sim:.2f}
        Min: {min_sim:.2f}
        Std: {std_sim:.2f}
        Samples: {len(semantic_sims)}
        '''
        plt.text(
            0.05, 0.6, stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.9,
                edgecolor='lightgray'
            )
        )
        plt.title(
            f'Semantic Similarity Distribution - Epoch {epoch+1}\n',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        plt.xlabel(
            'Cosine Similarity Score',
            fontsize=12,
            labelpad=10,
            fontweight='bold'
        )
        plt.ylabel(
            'Frequency',
            fontsize=12,
            labelpad=10,
            fontweight='bold'
        )
        plt.xticks(
            np.arange(0, 1.1, 0.1),
            fontsize=10,
            color='#555555'
        )
        plt.yticks(fontsize=10, color='#555555')
        plt.xlim(-0.02, 1.02)

        plt.grid(
            True,
            linestyle='--',
            linewidth=0.7,
            alpha=0.6,
            color='#dddddd'
        )
        plt.legend(
            loc='upper left',
            frameon=True,
            framealpha=0.9,
            edgecolor='#333333',
            fontsize=10
        )

        plt.tight_layout()
        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
        plt.close()

        print(f"\nSemantic similarity histogram saved to: {plot_path}")

        print("\nCalculating perplexity...")
        perplexity = calculate_perplexity(gen_model, all_gen_texts, tokenizer, batch_size)

        print("Calculating text matching metrics...")
        bleu_score = calculate_bleu(all_gen_texts, all_blackbox_texts)
        rouge_scores = calculate_rouge(all_gen_texts, all_blackbox_texts)

        print(f"\n{'='*50}\nValidation Results Epoch {epoch+1}")
        print(f"{'Metric':<20} | {'Value':<10}")
        print(f"{'BLEU':<20} | {bleu_score:.4f}")
        print(f"{'Perplexity':<20} | {perplexity:.2f}")
        print(f"{'ROUGE-1':<20} | {rouge_scores['rouge1']:.4f}")
        print(f"{'ROUGE-L':<20} | {rouge_scores['rougeL']:.4f}")
        print(f"{'Blackbox Prediction':<20} | {real_pred_mean:.4f}")
        print(f"{'Generated Prediction':<20} | {fake_pred_mean:.4f}")
        print(f"{'Prediction Gap':<20} | {abs(real_pred_mean - fake_pred_mean):.4f}")
        print(f"{'Semantic Similarity':<20} | {np.mean(semantic_sims):.4f}")
        print(f"{'Sample Count':<20} | {len(all_gen_texts)}")

        print("\nGenerated Samples Comparison:")
        for i in range(min(2, len(all_gen_texts))):
            print(f"[Prompt]: {all_prompts[i]}...")
            print(f"[Generated]: {all_gen_texts[i]}...")
            print(f"[Blackbox]: {all_blackbox_texts[i]}...")
            print("-"*80)

        return {
            "bleu": float(bleu_score),
            "perplexity": float(perplexity),
            "pred_gap": float(abs(real_pred_mean - fake_pred_mean)),
            "real_pred": float(real_pred_mean),
            "fake_pred": float(fake_pred_mean),
            "target_gap": float(abs(real_pred_mean - 0.5) + abs(fake_pred_mean - 0.5)),
            "rouge1": float(rouge_scores['rouge1']),
            "rougeL": float(rouge_scores['rougeL']),
            "semantic_sim": float(np.mean(semantic_sims))
        }

    def calculate_perplexity(model, texts, tokenizer, batch_size=4):
        total_loss = 0.0
        valid_samples = 0
        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device("cuda")

        print("Calculating perplexity:")
        with tqdm(total=len(texts)) as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                try:
                    if not batch_texts:
                        continue

                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_length=True
                    ).to(device)

                    valid_indices = [idx for idx, length in enumerate(inputs['length']) if length > 0]
                    if not valid_indices:
                        continue

                    filtered_inputs = {
                        k: v[valid_indices] for k, v in inputs.items()
                        if k != 'length'
                    }

                    with torch.no_grad():
                        outputs = model(**filtered_inputs, labels=filtered_inputs["input_ids"])

                        if isinstance(outputs.loss, torch.Tensor) and outputs.loss.dim() > 0:
                            loss = outputs.loss.mean()
                        else:
                            loss = outputs.loss

                        total_loss += loss.item() * len(filtered_inputs["input_ids"])
                        valid_samples += len(filtered_inputs["input_ids"])

                    del filtered_inputs, outputs
                except Exception as e:
                    print(f"Skipping batch {i//batch_size} (error: {str(e)})")
                pbar.update(len(batch_texts))
                torch.cuda.empty_cache()

        if valid_samples == 0:
            return float('inf')
        return torch.exp(torch.tensor(total_loss / valid_samples)).item()

    def generate_texts(model, tokenizer, input_ids, attention_mask, max_new_tokens=512):
        was_training = model.training
        model.eval()
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": 0.96,
            "temperature": 0.8,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "num_return_sequences": 1,
            "repetition_penalty": 1.5,
            "length_penalty": 0.9,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        }
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        if was_training:
            model.train()
        torch.cuda.empty_cache()
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def calculate_rouge(generated, references):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = []
        for gen, ref in zip(generated, references):
            scores.append(scorer.score(ref, gen))
        return {
            'rouge1': np.mean([s['rouge1'].fmeasure for s in scores]),
            'rougeL': np.mean([s['rougeL'].fmeasure for s in scores])
        }

    def calculate_bleu(generated, references):
        refs = [[word_tokenize(ref)] for ref in references]
        gens = [word_tokenize(gen) for gen in generated]
        return corpus_bleu(refs, gens)

    custom_train_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_name', type=str, default="wild_sft")
    parser.add_argument('--train_dataset_path', type=str, default="./datasets/train_data.json")
    parser.add_argument('--target_model_name', type=str, default="llama2-70b")
    parser.add_argument('--train_num_samples', type=int, default=150)
    parser.add_argument('--val_num_samples', type=int, default=50)
    parser.add_argument('--scoring_model_name', type=str, default="llama3.2-3b")
    parser.add_argument('--output_path', type=str, default="./output")
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="cache")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu_ids', type=str, default="1,2,3,4")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    train(args)
