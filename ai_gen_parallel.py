import pickle
from multiprocessing import Pool
from tqdm import tqdm
from transformers import pipeline

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


# 1) Load texts once from pickle
with open("texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open("ai_corpus_8000.pkl", "rb") as f:
    ai_corpus = pickle.load(f)

# texts = texts[:1500]

# 2) Worker initializer: load the HF pipeline once per process
def init_worker():
    global generator
    generator = pipeline(
        "text2text-generation",
        model=TARGET_LLM,
        framework="pt",
        device_map="auto",
        # device=0  # uncomment if you want to use GPU
    )

# 3) Batch-generation function
def generate_ai_batch(batch_texts):
    prompts = [
        "write the following text in your own words and keep all the contents "
        "and keep the text length close to original text: "
        + txt.strip().replace("\n", " ")
        for txt in batch_texts
    ]
    outputs = generator(
        prompts,
        max_length=1024,
        num_return_sequences=1,
        do_sample=False,
        batch_size=16
    )
    return [out["generated_text"] for out in outputs]

if __name__ == "__main__":
    batch_size = 500
    # split texts into sequential batches
    batches = [texts[i : i + batch_size] for i in range(16*batch_size, len(texts), batch_size)]

    # ai_corpus = []
    with Pool(processes=4, initializer=init_worker) as pool:
        for ai_batch in tqdm(pool.imap(generate_ai_batch, batches),
                             total=len(batches),
                             desc="Generating AI corpus"):
            ai_corpus.extend(ai_batch)
            # checkpoint after each batch
            with open(f"ai_corpus_{len(ai_corpus)}.pkl", "wb") as f:
                pickle.dump(ai_corpus, f)
