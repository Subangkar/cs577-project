import string

import torch


def contains_non_english_letter(s: str) -> bool:
    """
    Returns True if s contains at least one alphabetic character
    that is NOT an ASCII English letter (A–Z, a–z).
    Punctuation, digits, whitespace, and emojis are all ignored.
    """
    for ch in s:
        if ch.isalpha() and ch not in string.ascii_letters:
            # ch is a letter (so not punctuation/emoji), but not in A–Z/a–z
            return True
    return False


def save_torch_model(model, optimizer, epoch, fname):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        # "loss": loss,
    }, fname)
