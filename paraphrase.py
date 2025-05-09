import random, nltk, spacy, pyinflect
from nltk.corpus import wordnet as wn

# one‑time NLTK resources download
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

nlp = spacy.load("en_core_web_sm")

POS_MAP = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}

def get_synonyms(tok):
    """Return a list of WordNet synonyms (same POS) excluding the original word."""
    wn_pos = POS_MAP.get(tok.pos_)
    if not wn_pos:
        return []
    lemmas = {l.name().replace("_", " ")
              for s in wn.synsets(tok.lemma_, pos=wn_pos)
              for l in s.lemmas()}
    lemmas.discard(tok.lemma_)
    lemmas.discard(tok.text.lower())
    return list(lemmas)

def inflect(syn, tok):
    """Match the replacement’s morphology (plural, tense, etc.)."""
    doc = nlp(syn)              # analyse the synonym
    for w in doc:
        inf = w._.inflect(tok.tag_)   # try to inflect to target tag (e.g. VBD, NNS)
        if inf:
            return inf
    return syn                   # fallback: use as‑is


def paraphrase(sents, p=0.4):
    """Randomly replace ≈p proportion of eligible tokens with inflected synonyms."""
    outputs = []
    for sent in sents:
        doc = nlp(sent)
        out = []
        for tok in doc:
            if random.random() < p:
                syns = get_synonyms(tok)
                if syns:
                    out.append(inflect(random.choice(syns), tok))
                    continue
            out.append(tok.text)
        outputs.append(spacy.tokens.Doc(doc.vocab, words=out).text)
    return outputs

# if __name__ == "__main__":
#     s = "The quick brown fox jumps over the lazy dog."
#     for _ in range(3):
#         print(paraphrase(s))
