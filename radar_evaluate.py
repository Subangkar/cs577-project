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
PARAPHRASER_MODEL = "t5-small"
DETECTOR_MODEL = "distilbert-base-uncased"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PARAPHRASER_MODE = "vanilla"

BATCH_SIZE = 32

MAX_LENGTH = 1024
N_SENTENCES = 5000


# ------------------ Dataset Class ------------------ #
class RadarTestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]



# ------------------ Evaluation ------------------ #
def evaluate(detector, tokenizer, texts, labels: list[int]):
    detector.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    with torch.no_grad():
        logits = detector(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    # for i, t in enumerate(texts):
    #     print(probs[i], "->", t)
    print(probs)
    return roc_auc_score(labels, probs)


# ------------------ Training Loop ------------------ #
def test_radar(detector, texts, labels: list[int]):
    tokenizer_d = AutoTokenizer.from_pretrained(DETECTOR_MODEL)

    detector = detector.to(DEVICE)

    # xh_val = texts

    # dataset = RadarTestDataset(xh_val)
    # loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    auc = evaluate(detector, tokenizer_d, texts, labels)
    print(f"AUROC: {auc:.4f}")
    return auc


# ------------------ Entry Point ------------------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphraser_mode", choices=["vanilla", "hybrid"], default="vanilla",
                        help="Choose paraphraser type")
    args = parser.parse_args()

    PARAPHRASER_MODE = args.paraphraser_mode

    print(DEVICE)

    print("Loading WebText subset...")
    dataset = load_dataset("openwebtext", split="train[:1%]")
    print(len(dataset))
    texts = [item['text'] for item in dataset if
             len(item['text']) < MAX_LENGTH and not utils.contains_non_english_letter(item['text'])]

    N_SENTENCES = min(N_SENTENCES, len(texts))
    texts = texts[:N_SENTENCES]
    # texts = texts[:481] + texts[483:5002]

    # ai_corpus = generate_ai_corpus(texts, TARGET_LLM)

    checkpoint = torch.load("saved_models/detector_hybrid_4844_1024_12.pth")
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_MODEL, num_labels=2).to(DEVICE)
    detector.load_state_dict(checkpoint["model_state"]) #
    test_radar(detector=detector, texts=texts, labels=[0] * (len(texts)//2)) #  + [1]*(len(texts)-len(texts)//2)
