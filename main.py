#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
#%%
# === Configurations ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
MAX_LEN = 256
LR = 1e-5
PPO_EPOCHS = 3
TRAIN_STEPS = 5
GAMMA = 0.01  # Entropy penalty
EPSILON = 0.2  # PPO clip threshold
LAMBDA = 0.5  # Detector loss balance
#%%
# === Load Tokenizers and Models ===
t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-large").to(DEVICE)
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-large").to(DEVICE)
#%%
# === Sample Synthetic Dataset ===
def generate_synthetic_samples(num_samples=1000):
    prompts = ["The future of AI is", "The economic impact of technology", "Climate change poses challenges"]
    human_texts = [random.choice(prompts) + " " + " ".join([random.choice(["great", "unknown", "exciting", "dangerous"]) for _ in range(30)]) for _ in range(num_samples)]
    ai_texts = [random.choice(prompts) + " " + " ".join([random.choice(["remarkable", "innovative", "risky", "surprising"]) for _ in range(30)]) for _ in range(num_samples)]
    return human_texts, ai_texts

prompts = ["The future of AI is", "The economic impact of technology", "Climate change poses challenges"]
human_texts = [random.choice(prompts) + " " + " ".join([random.choice(["great", "unknown", "exciting", "dangerous"]) for _ in range(3)]) for _ in range(3)]
# ai_texts = [random.choice(prompts) + " " + " ".join([random.choice(["remarkable", "innovative", "risky", "surprising"]) for _ in range(30)]) for _ in range(num_samples)]
human_texts

#%%
class TextDataset(Dataset):
    def __init__(self, human, ai):
        self.human = human
        self.ai = ai
        self.data = [(t, 1) for t in human] + [(t, 0) for t in ai]
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding="max_length")
        return encoding.input_ids.squeeze(0), encoding.attention_mask.squeeze(0), torch.tensor(label)
#%%
# === Training Loop Placeholder ===
def train():
    human_texts, ai_texts = generate_synthetic_samples(500)
    dataset = TextDataset(human_texts, ai_texts)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer_D = AdamW(roberta_model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in tqdm(range(TRAIN_STEPS)):
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            outputs = roberta_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, 1]  # Use only the logit corresponding to 'human' class
            loss = loss_fn(logits, labels.float())

            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

        if step % 100 == 0:
            print(f"Step {step}: Detector Loss = {loss.item():.4f}")
#%%
train()
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import random

# ----------------------- Dataset ----------------------- #
class TextDataset(Dataset):
    def __init__(self, texts_human, texts_ai):
        self.texts_human = texts_human
        self.texts_ai = texts_ai

    def __len__(self):
        return len(self.texts_human)

    def __getitem__(self, idx):
        return self.texts_human[idx], self.texts_ai[idx]

# ----------------------- Reward Function ----------------------- #
def compute_reward(detector, tokenizer, texts):
    detector.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(detector.device)
    with torch.no_grad():
        outputs = detector(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1]  # Assuming label 1 is AI text

def normalize_rewards(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-7)

# ----------------------- Training Functions ----------------------- #
def update_paraphraser(paraphraser, tokenizer, optimizer, xm, xp, advantages):
    paraphraser.train()
    inputs = tokenizer(xm, return_tensors="pt", padding=True, truncation=True).to(paraphraser.device)
    targets = tokenizer(xp, return_tensors="pt", padding=True, truncation=True).input_ids.to(paraphraser.device)

    outputs = paraphraser(**inputs, labels=targets)
    log_probs = -outputs.loss  # Negative loss as log prob approximation
    loss = -(log_probs * advantages.to(paraphraser.device)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_detector(detector, tokenizer, optimizer, xh, xm, xp):
    detector.train()
    texts = xh + xm + xp
    labels = torch.tensor([0]*len(xh) + [1]*len(xm) + [1]*len(xp)).to(detector.device)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(detector.device)
    outputs = detector(**inputs)
    loss = nn.CrossEntropyLoss()(outputs.logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ----------------------- Evaluation ----------------------- #
def evaluate_auroc(detector, tokenizer, xh_val, xm_val):
    detector.eval()
    texts = xh_val + xm_val
    labels = [0]*len(xh_val) + [1]*len(xm_val)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(detector.device)
    with torch.no_grad():
        logits = detector(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return roc_auc_score(labels, probs)

# ----------------------- Main RADAR Training Loop ----------------------- #
def train_radar(texts_human, texts_ai, val_ratio=0.1, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split validation
    split = int(len(texts_human) * (1 - val_ratio))
    xh_train, xh_val = texts_human[:split], texts_human[split:]
    xm_train, xm_val = texts_ai[:split], texts_ai[split:]

    dataset = TextDataset(xh_train, xm_train)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    detector = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
    paraphraser = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)

    optimizer_detector = optim.Adam(detector.parameters(), lr=1e-5)
    optimizer_paraphraser = optim.Adam(paraphraser.parameters(), lr=1e-5)

    best_auroc = 0
    best_detector_state = None
    best_paraphraser_state = None

    for epoch in range(epochs):
        replay_buffer = []
        for xh_batch, xm_batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            # Paraphrase
            inputs = tokenizer(list(xm_batch), return_tensors="pt", padding=True, truncation=True).to(device)
            paraphrased_ids = paraphraser.generate(**inputs)
            xp_batch = tokenizer.batch_decode(paraphrased_ids, skip_special_tokens=True)

            # Reward and Advantage
            rewards = compute_reward(detector, tokenizer, xp_batch)
            advantages = normalize_rewards(rewards)

            # Store in buffer
            replay_buffer.append((list(xh_batch), list(xm_batch), xp_batch, advantages))

        # PPO-like training phase
        for xh, xm, xp, adv in replay_buffer:
            update_paraphraser(paraphraser, tokenizer, optimizer_paraphraser, xm, xp, adv)
            update_detector(detector, tokenizer, optimizer_detector, xh, xm, xp)

        # Evaluate
        auroc = evaluate_auroc(detector, tokenizer, xh_val, xm_val)
        print(f"Validation AUROC: {auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_detector_state = detector.state_dict()
            best_paraphraser_state = paraphraser.state_dict()

    # Return best models
    detector.load_state_dict(best_detector_state)
    paraphraser.load_state_dict(best_paraphraser_state)
    return detector, paraphraser

# ----------------------- Example Usage ----------------------- #
if __name__ == "__main__":
    # Placeholder data
    texts_human = ["This is a human sentence." for _ in range(100)]
    texts_ai = ["This is a generated sentence." for _ in range(100)]
    detector, paraphraser = train_radar(texts_human, texts_ai, epochs=2)

#%%
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
DETECTOR_MODEL = "roberta-large"
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
    generator = pipeline("text-generation", model=model_name, device=0)
    ai_texts = []
    for text in tqdm(human_texts, desc="Generating AI completions"):
        prompt = text[:30]
        result = generator(prompt, max_length=max_length, do_sample=True)[0]['generated_text']
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
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

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
    dataset = load_dataset("openwebtext", split="train")
    texts = [item['text'] for item in dataset.select(range(1000)) if len(item['text']) > 200]
    ai_corpus = generate_ai_corpus(texts, TARGET_LLM)
    detector, paraphraser = train_radar(texts, ai_corpus, epochs=2)

#%%
