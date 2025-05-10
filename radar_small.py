import pickle
import traceback
import warnings

import utils

warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
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

from paraphrase import paraphrase

# ------------------ Config ------------------ #
MODEL_CONFIG = {
    "dolly": "sshleifer/tiny-gpt2",
    "gptj": "sshleifer/tiny-gpt2",
    "distilroberta": "philschmid/tiny-bert-sst2-distilled",
    "gemma": "google/gemma-2-2b",
    "parrot": "prithivida/parrot_paraphraser_on_T5",
    "t5": "ramsrigouthamg/t5_paraphraser",
}
TARGET_LLM = MODEL_CONFIG["t5"]
PARAPHRASER_MODEL = "ramsrigouthamg/t5_paraphraser"
DETECTOR_MODEL = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_IDS = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
PARAPHRASER_MODE = "vanilla"

BATCH_SIZE = 16
N_EPOCHS = 8
MAX_LENGTH = 2048
N_SENTENCES = 10000


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
    generator = pipeline("text2text-generation", model=model_name, framework="pt")
    ai_texts = []
    for text in tqdm(human_texts, desc="Generating AI completions"):
        prompt = "write the following text in your own words and keep all the contents and keep the text length close to original text: " + text.strip().replace(
            "\n", " ")

        out = generator(prompt,
                        max_length=MAX_LENGTH,
                        num_return_sequences=1,
                        do_sample=False,
                        # top_k=50,
                        # top_p=0.95,
                        )[0]["generated_text"]
        ai_texts.append(out)
    return ai_texts


# ------------------ Hybrid Paraphraser Components ------------------ #
BT_EN_TO_FR = "Helsinki-NLP/opus-mt-en-fr"
BT_FR_TO_EN = "Helsinki-NLP/opus-mt-fr-en"

bt_tokenizer_en_fr = MarianTokenizer.from_pretrained(BT_EN_TO_FR)
bt_model_en_fr = MarianMTModel.from_pretrained(BT_EN_TO_FR).to(DEVICE)

bt_tokenizer_fr_en = MarianTokenizer.from_pretrained(BT_FR_TO_EN)
bt_model_fr_en = MarianMTModel.from_pretrained(BT_FR_TO_EN).to(DEVICE)


def backtranslate(texts):
    inputs = bt_tokenizer_en_fr(texts, return_tensors="pt", padding=True, truncation=True).to(
        DEVICE)
    with torch.no_grad():
        translated = bt_model_en_fr.generate(**inputs, max_length=bt_tokenizer_en_fr.model_max_length)
    fr_texts = bt_tokenizer_en_fr.batch_decode(translated, skip_special_tokens=True)
    inputs = bt_tokenizer_fr_en(fr_texts, return_tensors="pt", padding=True, truncation=True).to(
        DEVICE)
    with torch.no_grad():
        backtranslated = bt_model_fr_en.generate(**inputs, max_length=bt_tokenizer_fr_en.model_max_length)
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
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=tokenizer.model_max_length).to(DEVICE)
    with torch.no_grad():
        outputs = detector(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[:, 1]


def normalize_rewards(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-7)


# ------------------ PPO-style Updates ------------------ #
def update_paraphraser(model, tokenizer, optimizer, xm, xp, adv):
    model.train()
    # tokenize source (xm)
    inputs = tokenizer(
        xm,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(MAX_LENGTH, tokenizer.model_max_length)
    ).to(DEVICE)
    # tokenize targets (xp)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            xp,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(MAX_LENGTH, tokenizer.model_max_length)
        ).input_ids.to(DEVICE)

    # *** Modified: compute logits and per‐sample loss manually for DataParallel ***
    outputs = model(**inputs, labels=targets)
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    vocab_size = logits.size(-1)

    # Flatten logits/targets for token‐level CE, no reduction
    loss_fct = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        reduction="none"
    )
    flat_logits = logits.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
    flat_targets = targets.view(-1)  # [batch*seq_len]
    flat_loss = loss_fct(flat_logits, flat_targets)  # [batch*seq_len]

    # Reshape to [batch, seq_len] then average per sample
    per_sample_loss = (
        flat_loss
        .view(targets.size(0), -1)
        .mean(dim=1)
    )  # [batch]

    # Weight by adv and take final mean
    loss = -(per_sample_loss * adv.to(DEVICE)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_detector(detector, tokenizer, optimizer, xh, xm, xp):
    detector.train()
    batch_size = len(xh)
    texts_all = [t for triple in zip(xh, xm, xp) for t in triple]
    labels_all = torch.tensor(
        [lab for _ in range(batch_size) for lab in (0, 1, 1)],
        dtype=torch.long,
        device=DEVICE)
    # texts = xh + xm + xp
    # labels = torch.tensor([0] * len(xh) + [1] * len(xm) + [1] * len(xp)).to(DEVICE)

    for i in range(3):
        texts = texts_all[i * batch_size:(i + 1) * batch_size]
        labels = labels_all[i * batch_size:(i + 1) * batch_size]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=min(MAX_LENGTH, tokenizer.model_max_length)).to(DEVICE)
        # print(type(inputs), inputs)
        # exit(0)
        try:
            outputs = detector(**inputs)
        except RuntimeError as e:
            print(inputs["input_ids"].size(), inputs["attention_mask"].size())
            print(texts)
            traceback.print_exc()
            exit(0)
        # outputs = detector(**inputs)
        loss = nn.CrossEntropyLoss()(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ------------------ Evaluation ------------------ #
def evaluate(detector, tokenizer, xh_val, xm_val):
    detector.eval()
    probs_all = []
    # texts = xh_val + xm_val
    labels = [0] * len(xh_val) + [1] * len(xm_val)

    for texts in [xh_val, xm_val]:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=tokenizer.model_max_length).to(DEVICE)
        with torch.no_grad():
            logits = detector(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probs_all.append(probs)
    return roc_auc_score(labels, np.concatenate(probs_all))


# ------------------ Training Loop ------------------ #
def train_radar(human_texts, ai_texts, epochs=2, paraphraser_mode="vanilla"):
    tokenizer_p = AutoTokenizer.from_pretrained(PARAPHRASER_MODEL, use_fast=True)
    tokenizer_d = AutoTokenizer.from_pretrained(DETECTOR_MODEL, use_fast=True)

    paraphraser = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_MODEL).to(DEVICE)
    if GPU_IDS:
        paraphraser = nn.DataParallel(paraphraser, device_ids=GPU_IDS)
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_MODEL, num_labels=2).to(DEVICE)
    if GPU_IDS:
        detector = nn.DataParallel(detector, device_ids=GPU_IDS)

    optimizer_p = optim.AdamW(paraphraser.parameters(), lr=2e-5)
    optimizer_d = optim.AdamW(detector.parameters(), lr=2e-5)

    split = int(len(human_texts) * 0.9)
    xh_train, xh_val = human_texts[:split], human_texts[split:]
    xm_train, xm_val = ai_texts[:split], ai_texts[split:]

    dataset = RadarDataset(xh_train, xm_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_auc = 0
    best_detector, best_paraphraser = None, None
    print(">>", epochs)
    for epoch in range(epochs):
        buffer = []
        for xh, xm in tqdm(loader, desc=f"Epoch {epoch + 1}/{N_EPOCHS}"):
            inputs = tokenizer_p(xm, return_tensors="pt", padding=True, truncation=True,
                                 max_length=tokenizer_p.model_max_length).to(DEVICE)
            gen_model = paraphraser.module if isinstance(paraphraser, nn.DataParallel) else paraphraser
            paraphrased_ids = gen_model.generate(**inputs, max_length=tokenizer_p.model_max_length)
            # paraphrased_ids = paraphraser.generate(**inputs, max_length=tokenizer_p.model_max_length)
            xp_raw = tokenizer_p.batch_decode(paraphrased_ids, skip_special_tokens=True)
            if paraphraser_mode == "hybrid":
                xp = backtranslate(xp_raw)
                xp = paraphrase(xp)
            elif paraphraser_mode == "bt":
                xp = backtranslate(xp_raw)
            elif paraphraser_mode == "lex":
                xp = paraphrase(xp_raw)
            else:
                xp = xp_raw

            rewards = compute_reward(detector, tokenizer_d, xp)
            adv = normalize_rewards(rewards)
            buffer.append((list(xh), list(xm), xp, adv))

        # torch.cuda.empty_cache()

        for xh, xm, xp, adv in tqdm(buffer, desc=f"Epoch {epoch + 1}/{N_EPOCHS} Updating"):
            update_paraphraser(paraphraser, tokenizer_p, optimizer_p, xm, xp, adv)
            update_detector(detector, tokenizer_d, optimizer_d, xh, xm, xp)

        # torch.cuda.empty_cache()

        auc = evaluate(detector, tokenizer_d, xh_val, xm_val)
        print(f"Validation AUROC: {auc:.4f}")
        # if auc > best_auc:
        #     best_auc = auc
        #     best_detector = detector.state_dict()
        #     best_paraphraser = paraphraser.state_dict()

        # torch.cuda.empty_cache()

        utils.save_torch_model(
            detector.module if isinstance(detector, nn.DataParallel) else detector,
            optimizer_d, epoch=epoch,
            fname=f"saved_models/detector_{PARAPHRASER_MODE}_{N_SENTENCES}_{MAX_LENGTH}_{epoch}.pth")
        utils.save_torch_model(
            paraphraser.module if isinstance(detector, nn.DataParallel) else paraphraser,
            optimizer_p, epoch=epoch,
            fname=f"saved_models/paraphraser_{PARAPHRASER_MODE}_{N_SENTENCES}_{MAX_LENGTH}_{epoch}.pth")

    # if best_detector and best_paraphraser:
    #     detector.load_state_dict(best_detector)
    #     paraphraser.load_state_dict(best_paraphraser)
    return detector, paraphraser


# ------------------ Entry Point ------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphraser_mode", choices=["vanilla", "lex", "bt", "hybrid"], default="vanilla",
                        help="Choose paraphraser type")
    args = parser.parse_args()

    PARAPHRASER_MODE = args.paraphraser_mode

    print(DEVICE)

    # print("Loading WebText subset...")
    # dataset = load_dataset("openwebtext", split="train[:1%]")
    # print(len(dataset))
    # texts = [item['text'] for item in dataset if
    #          len(item['text']) < MAX_LENGTH and not utils.contains_non_english_letter(item['text'])]
    with open("texts.pkl", "rb") as f:
        texts = pickle.load(f)

    N_SENTENCES = min(N_SENTENCES, len(texts))
    texts = texts[:N_SENTENCES]
    # texts = texts[:481] + texts[483:5002]

    # ai_corpus = generate_ai_corpus(texts, TARGET_LLM)
    with open(f"ai_corpus_{N_SENTENCES}.pkl", "rb") as f:
        ai_corpus = pickle.load(f)

    # print(">>", texts[:5])
    # print(">>", ai_corpus[:5])

    # print(texts)
    # Save to a text file
    with open("human_texts.txt", "w") as f:
        for line in texts:
            f.write("\n>> " + line.strip().replace("\n", " "))  # Add newline after each string
    with open("ai_texts.txt", "w") as f:
        for line in ai_corpus:
            f.write("\n>> " + line)  # Add newline after each string
    detector, paraphraser = train_radar(texts, ai_corpus, epochs=N_EPOCHS, paraphraser_mode=args.paraphraser_mode)
