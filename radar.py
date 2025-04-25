import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import random

# ------------------ Config ------------------ #
MODEL_CONFIG = {
    "dolly": "databricks/dolly-v2-3b",
    "gptj": "EleutherAI/gpt-j-6B"
}
TARGET_LLM = MODEL_CONFIG["dolly"]  # Change to "gptj" if needed
PARAPHRASER_MODEL = "t5-large"
DETECTOR_MODEL = "distilroberta-base"
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

# ------------------ Generate AI-text Corpus ------------------ #
def generate_ai_corpus(human_texts, model_name, max_length=200):
    generator = pipeline("text-generation", model=model_name, device=0, trust_remote_code=True)
    ai_texts = []
    for text in tqdm(human_texts, desc="Generating AI completions"):
        prompt = text[:30]
        result = generator(prompt, do_sample=True)[0]['generated_text'] # , max_length=max_length
        ai_texts.append(result)
    return ai_texts

# ------------------ Reward Calculation ------------------ #
def compute_reward(detector, tokenizer, texts):
    detector.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = detector(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1]

def normalize_rewards(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-7)

# ------------------ PPO-style Paraphraser Update ------------------ #
def update_paraphraser(model, tokenizer, optimizer, xm, xp, adv):
    model.train()
    inputs = tokenizer(xm, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    targets = tokenizer(xp, return_tensors="pt", padding=True, truncation=True).input_ids.to(DEVICE)
    outputs = model(**inputs, labels=targets)
    loss = -(outputs.loss * adv.to(DEVICE)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ------------------ Detector Update ------------------ #
def update_detector(detector, tokenizer, optimizer, xh, xm, xp):
    detector.train()
    texts = xh + xm + xp
    labels = torch.tensor([0]*len(xh) + [1]*len(xm) + [1]*len(xp)).to(DEVICE)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
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
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        logits = detector(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return roc_auc_score(labels, probs)

# ------------------ RADAR Training Loop ------------------ #
def train_radar(human_texts, ai_texts, epochs=3):
    tokenizer_paraphraser = AutoTokenizer.from_pretrained(PARAPHRASER_MODEL)
    tokenizer_detector = AutoTokenizer.from_pretrained(DETECTOR_MODEL)

    paraphraser = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_MODEL).to(DEVICE)
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_MODEL, num_labels=2).to(DEVICE)

    optimizer_p = optim.AdamW(paraphraser.parameters(), lr=1e-5)
    optimizer_d = optim.AdamW(detector.parameters(), lr=1e-5)

    split = int(len(human_texts) * 0.9)
    xh_train, xh_val = human_texts[:split], human_texts[split:]
    xm_train, xm_val = ai_texts[:split], ai_texts[split:]

    dataset = RadarDataset(xh_train, xm_train)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    best_auc = 0
    for epoch in range(epochs):
        buffer = []
        for xh, xm in tqdm(loader, desc=f"Epoch {epoch+1}"):
            inputs = tokenizer_paraphraser(xm, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            paraphrased_ids = paraphraser.generate(**inputs)
            xp = tokenizer_paraphraser.batch_decode(paraphrased_ids, skip_special_tokens=True)
            rewards = compute_reward(detector, tokenizer_detector, xp)
            adv = normalize_rewards(rewards)
            buffer.append((list(xh), list(xm), xp, adv))

        for xh, xm, xp, adv in buffer:
            update_paraphraser(paraphraser, tokenizer_paraphraser, optimizer_p, xm, xp, adv)
            update_detector(detector, tokenizer_detector, optimizer_d, xh, xm, xp)

        auc = evaluate(detector, tokenizer_detector, xh_val, xm_val)
        print(f"Validation AUROC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_detector = detector.state_dict()
            best_paraphraser = paraphraser.state_dict()

    detector.load_state_dict(best_detector)
    paraphraser.load_state_dict(best_paraphraser)
    return detector, paraphraser

# ------------------ Entry Point ------------------ #
if __name__ == '__main__':
    print("Loading WebText...")
    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    texts = [item['text'] for item in dataset.select(range(100)) if len(item['text']) > 200]
    ai_corpus = generate_ai_corpus(texts, TARGET_LLM)
    detector, paraphraser = train_radar(texts, ai_corpus, epochs=2)
