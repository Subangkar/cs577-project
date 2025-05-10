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


import unicodedata
import regex

# Compile once: anything NOT in Basic Latin block, nor Punctuation, nor Symbols, nor whitespace
_non_english_re = regex.compile(r'(?V1)[^\p{Block=BasicLatin}\p{P}\p{S}\s]')

def contains_non_english_char(s: str) -> bool:
    """
    Returns True if s contains any character that is NOT
     - an ASCII-range character (Basic Latin block: U+0000–U+007F),
     - punctuation (\p{P}),
     - symbol (\p{S}), or
     - whitespace (\s).
    That effectively treats A–Z/a–z, digits, ASCII punctuation,
    all Unicode punctuation, all symbols/emoji, and spaces/tabs/newlines
    as 'allowed', and flags everything else.
    """
    # 1) normalize to NFC so composed characters stay together
    s = unicodedata.normalize("NFC", s)
    # 2) search for any disallowed char
    return bool(_non_english_re.search(s))