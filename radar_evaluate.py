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
def evaluate(detector, tokenizer, texts, labels, batch_size=BATCH_SIZE):
    detector.eval()
    all_probs = []

    # split texts & labels into batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i: i + batch_size]
        batch_labels = labels[i: i + batch_size]

        # tokenize + move to device
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)

        # forward
        with torch.no_grad():
            logits = detector(**inputs).logits

        # grab P(model=1)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)

    # concatenate back and compute AUROC
    all_probs = np.concatenate(all_probs)
    # print(np.sort(all_probs)[::-1])
    print(list(all_probs))
    auc = roc_auc_score(labels, all_probs)
    print(f"AUROC: {auc:.4f}")
    return auc


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
    # dataset = load_dataset("openwebtext", split="train[:1%]")
    # print(len(dataset))
    # texts = [item['text'] for item in dataset if
    #          len(item['text']) < MAX_LENGTH and not utils.contains_non_english_letter(item['text'])]

    texts = [
        "Tufts University doctoral student Rümeysa Öztürk was released from a Louisiana detention center Friday, six weeks after masked federal agents took her into custody amid the Trump administration’s effort to deport noncitizens who have protested against the war in Gaza",
        "It is important to note that the integration of advanced methodologies significantly enhances the overall system performance. In particular, the utilization of hierarchical frameworks has the potential to yield an appreciable improvement in scalability metrics. Moreover, it should be emphasized that future iterations will undoubtedly benefit from a more robust parameterization strategy. Consequently, stakeholders must remain cognizant of evolving requirements and adapt their approaches accordingly. Overall, this paradigm shift may represent a seminal moment in the domain.",
        "The aforementioned phenomenon can be interpreted as a confluence of multiple latent variables that interact in a non-linear fashion. Specifically, one can observe that the emergent behaviors are contingent upon the dynamic interplay between these variables. From a methodological perspective, rigorous validation protocols are indispensable to ensure empirical reliability. Furthermore, it is worthy of mention that the prevailing conventions may not fully capture the nuances of real-world scenarios. Therefore, continuous refinement is recommended to achieve optimal outcomes.",
        "In order to fully comprehend the implications, one must systematically deconstruct each component of the system architecture. This deconstruction involves, but is not limited to, the assessment of data flow, control logic, and synchronization mechanisms. Additionally, it is critical to consider the broader socio-technical context in which these components operate. By doing so, practitioners can prevent potential misalignments between design intent and operational reality. Ultimately, such a holistic approach fosters resilient and adaptive solutions.",
        "It should be noted that the above guidelines are formulated based on a synthesis of extant literature and best practices. As a result, the recommendations herein are reflective of consensus-driven insights rather than anecdotal evidence. Nonetheless, further empirical investigation may be required to validate these propositions under varied conditions. Consequently, the research community is encouraged to engage in collaborative studies that span multiple disciplines. This collective endeavor will undoubtedly enrich the foundational knowledge base.",
        "The system's capacity to self-optimize is largely predicated on the continuous ingestion of high-fidelity telemetry data. However, the efficacy of such optimization routines is heavily influenced by the underlying algorithmic fidelity. In this context, it becomes imperative to implement adaptive learning rates and sophisticated regularization schemes. Moreover, the interplay between exploration and exploitation must be judiciously balanced to avoid convergence plateaus. In summary, the amalgamation of these strategies holds great promise for next-generation intelligent systems."]

    N_SENTENCES = min(N_SENTENCES, len(texts))
    texts = texts[:N_SENTENCES]
    # texts = texts[:481] + texts[483:5002]

    # ai_corpus = generate_ai_corpus(texts, TARGET_LLM)

    checkpoint = torch.load("saved_models_0/detector_hybrid_4844_1024_12.pth")
    detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_MODEL, num_labels=2).to(DEVICE)
    detector.load_state_dict(checkpoint["model_state"])  #
    test_radar(detector=detector, texts=texts, labels=[0] + [1] * (len(texts) - 1))  # + [1]*(len(texts)-len(texts)//2)
