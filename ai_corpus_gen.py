import warnings

import utils
import pickle

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

MODEL_CONFIG = {
    "dolly": "sshleifer/tiny-gpt2",
    "gptj": "sshleifer/tiny-gpt2",
    "distilroberta": "philschmid/tiny-bert-sst2-distilled",
    "gemma": "google/gemma-2-2b",
    "parrot": "prithivida/parrot_paraphraser_on_T5",
    "t5": "ramsrigouthamg/t5_paraphraser",
}
TARGET_LLM = MODEL_CONFIG["t5"]

MAX_LENGTH = 2048
N_SENTENCES = 20000


# ------------------ Text Generation ------------------ #
def generate_ai_corpus(human_texts, model_name, max_length=128):
    generator = pipeline("text2text-generation", model=model_name, framework="pt")
    ai_texts = []
    for text in tqdm(human_texts, desc="Generating AI completions"):
        prompt = "write the following text in your own words and keep all the contents and keep the text length close to original text: " + text.strip().replace(
            "\n", " ")

        out = generator(prompt,
                        max_length=1024,
                        num_return_sequences=1,
                        do_sample=False,
                        # top_k=50,
                        # top_p=0.95,
                        )[0]["generated_text"]
        ai_texts.append(out)
    return ai_texts


print("Loading WebText subset...")
dataset = load_dataset("openwebtext", split="train[:1%]")
print(len(dataset))
texts = [item['text'] for item in dataset if
         len(item['text']) < MAX_LENGTH and not utils.contains_non_english_char(item['text'])]
texts_nonen = [item['text'] for item in dataset if
               len(item['text']) < MAX_LENGTH and utils.contains_non_english_char(item['text'])]

print(len(texts))
N_SENTENCES = min(N_SENTENCES, len(texts))
texts = texts[:N_SENTENCES]
# texts = texts[:481] + texts[483:5002]

with open("texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open("texts_nonen.pkl", "wb") as f:
    pickle.dump(texts_nonen, f)

total = 0
batch_size = 500
ai_corpus = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i: i + batch_size]
    ai_corpus_batch = generate_ai_corpus(batch_texts, TARGET_LLM)
    ai_corpus.extend(ai_corpus_batch)
    with open(f"ai_corpus_{len(ai_corpus)}.pkl", "wb") as f:
        pickle.dump(ai_corpus, f)
