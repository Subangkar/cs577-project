import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline,
    MarianMTModel,
    MarianTokenizer
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import argparse
import random

# ------------------ Config ------------------ #
MODEL_CONFIG = {
    "dolly": "sshleifer/tiny-gpt2",
    "gptj": "sshleifer/tiny-gpt2"
}
TARGET_LLM = MODEL_CONFIG["dolly"]
PARAPHRASER_MODEL = "t5-small"
DETECTOR_MODEL = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Dataset Class ------------------ #
class RadarDataset(Dataset):
    def __init__(self, human_texts, ai_texts):
        self.human_texts = human_texts
        self.ai_texts = ai_texts

    def __len__(self):
        return len(self.human_texts)

    def __getitem__(self, idx):
        return self.human_texts[idx], self.ai_texts[idx]

# ------------------ Text Generation ------------------ #
def generate_ai_corpus(human_texts, model_name, max_length=128):
    generator = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
    ai_texts = []
    for text in tqdm(human_texts, desc="Generating AI completions"):
        prompt = text[:30]
        result = generator(prompt, do_sample=True, max_length=max_length)[0]['generated_text']
        ai_texts.append(result)
    return ai_texts

# ------------------ Hybrid Paraphraser Components ------------------ #
BT_EN_TO_FR = "Helsinki-NLP/opus-mt-en-fr"
BT_FR_TO_EN = "Helsinki-NLP/opus-mt-fr-en"

bt_tokenizer_en_fr = MarianTokenizer.from_pretrained(BT_EN_TO_FR)
bt_model_en_fr = MarianMTModel.from_pretrained(BT_EN_TO_FR).to(DEVICE)

bt_tokenizer_fr_en = MarianTokenizer.from_pretrained(BT_FR_TO_EN)
bt_model_fr_en = MarianMTModel.from_pretrained(BT_FR_TO_EN).to(DEVICE)

def backtranslate(texts):
    inputs = bt_tokenizer_en_fr(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        translated = bt_model_en_fr.generate(**inputs, max_length=128)
    fr_texts = bt_tokenizer_en_fr.batch_decode(translated, skip_special_tokens=True)
    inputs = bt_tokenizer_fr_en(fr_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        backtranslated = bt_model_fr_en.generate(**inputs, max_length=128)
    en_texts = bt_tokenizer_fr_en.batch_decode(backtranslated, skip_special_tokens=True)
    return en_texts

def insert_filler(text):
    fillers = ["well", "actually", "you know", "I mean", "to be honest"]
    words = text.split()
    if len(words) > 5:
        idx = random.randint(1, len(words) - 2)
        words.insert(idx, random.choice(fillers))
    return " ".join(words)

# ------------------ Reward Calculation ------------------ #
def compute_reward(detector, tokenizer, texts):
    detector.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = detector(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1]

def normalize_rewards(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-7)

# ------------------ PPO-style Updates ------------------ #
def update_paraphraser(model, tokenizer, optimizer, xm, xp, adv):
    model.train()
    inputs = tokenizer(xm, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    targets = tokenizer(xp, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to(DEVICE)
    outputs = model(**inputs, labels=targets)
    loss = -(outputs.loss * adv.to(DEVICE)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_detector(detector, tokenizer, optimizer, xh, xm, xp):
    detector.train()
    texts = xh + xm + xp
    labels = torch.tensor([0]*len(xh) + [1]*len(xm) + [1]*len(xp)).to(DEVICE)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    outputs = detector(**inputs)
    loss = nn.CrossEntropyLoss()(outputs.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ------------------ Evaluation ------------------ #
def evaluate(detector, tokenizer, xh_val, xm_val):
    detector.eval()
    texts = xh_val + xm_val
    labels = [0]*len(xh_val) + [1]*len(xm_val)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        logits = detector(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return roc_auc_score(labels, probs)

# ------------------ Training Loop ------------------ #
def train_radar(human_texts, ai_texts, epochs=2, paraphraser_mode="vanilla"):
    tokenizer_p = AutoTokenizer.from_pretrained(PARAPHRASER_MODEL)
    tokenizer_d = AutoTokenizer.from_pretrained(DETECTOR_MODEL)

    paraphraser = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_MODEL).to(DEVICE)
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_MODEL, num_labels=2).to(DEVICE)

    optimizer_p = optim.AdamW(paraphraser.parameters(), lr=2e-5)
    optimizer_d = optim.AdamW(detector.parameters(), lr=2e-5)

    split = int(len(human_texts) * 0.9)
    xh_train, xh_val = human_texts[:split], human_texts[split:]
    xm_train, xm_val = ai_texts[:split], ai_texts[split:]

    dataset = RadarDataset(xh_train, xm_train)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    best_auc = 0
    best_detector, best_paraphraser = None, None

    for epoch in range(epochs):
        buffer = []
        for xh, xm in tqdm(loader, desc=f"Epoch {epoch+1}"):
            inputs = tokenizer_p(xm, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
            paraphrased_ids = paraphraser.generate(**inputs, max_length=128)
            xp_raw = tokenizer_p.batch_decode(paraphrased_ids, skip_special_tokens=True)

            if paraphraser_mode == "hybrid":
                xp_bt = backtranslate(xp_raw)
                xp = [insert_filler(x) for x in xp_bt]
            else:
                xp = xp_raw

            rewards = compute_reward(detector, tokenizer_d, xp)
            adv = normalize_rewards(rewards)
            buffer.append((list(xh), list(xm), xp, adv))

        for xh, xm, xp, adv in buffer:
            update_paraphraser(paraphraser, tokenizer_p, optimizer_p, xm, xp, adv)
            update_detector(detector, tokenizer_d, optimizer_d, xh, xm, xp)

        auc = evaluate(detector, tokenizer_d, xh_val, xm_val)
        print(f"Validation AUROC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_detector = detector.state_dict()
            best_paraphraser = paraphraser.state_dict()

    if best_detector and best_paraphraser:
        detector.load_state_dict(best_detector)
        paraphraser.load_state_dict(best_paraphraser)
    return detector, paraphraser

# ------------------ Entry Point ------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphraser_mode", choices=["vanilla", "hybrid"], default="vanilla", help="Choose paraphraser type")
    args = parser.parse_args()

    print("Loading WebText subset...")
    dataset = load_dataset("openwebtext", split="train[:1%]")
    texts = [item['text'] for item in dataset if len(item['text']) > 200][:30]

    ai_corpus = generate_ai_corpus(texts, TARGET_LLM)
    detector, paraphraser = train_radar(texts, ai_corpus, epochs=2, paraphraser_mode=args.paraphraser_mode)
