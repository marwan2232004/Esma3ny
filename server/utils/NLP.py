from utils.Constants import *
import string
import torch


def preprocess_vocab():
    # Define the vocabulary: add lowercase letters, digits, and special tokens
    english_characters = list(
        string.ascii_lowercase + " "
    )  # List of English characters
    arabic_characters = list(
        "ابتثجحخدذرزسشصضطظعغفقكلمنهويئءىةؤ"
    )  # List of Arabic characters

    # Combine all characters
    characters = english_characters + arabic_characters
    vocab = special_tokens + characters  # Combine special tokens and characters

    # Build dictionaries for character-to-index and index-to-character
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    return char2idx, idx2char, vocab_size


def tokenize_text(
        text, char2idx, max_len=MAX_TEXT_LEN, start_token=True, end_token=True
):
    for char in text:
        idx = char2idx.get(char, char2idx["<UNK>"])
        if idx == 1:
            print(char)

    tokens = [char2idx.get(char, char2idx["<UNK>"]) for char in text]

    max_len -= start_token + end_token  # fot the start and end token

    # pad to a maximum length (for batching)
    if max_len is not None:
        text_len = len(tokens)
        if text_len < max_len:
            if end_token:
                tokens += [char2idx["<EOS>"]]
            tokens += [char2idx["<PAD>"]] * (max_len - text_len)  # Pad
        else:
            tokens = tokens[:max_len]  # Truncate if longer than max_len
            if end_token:
                tokens += [char2idx["<EOS>"]]

    if start_token:
        tokens.insert(0, char2idx["<SOS>"])

    return tokens


def TextDecoder(sentence, idx2char):
    out = ""
    for token in sentence:
        if isinstance(token, torch.Tensor):
            token = token.item()
        char = idx2char[token]
        if char == "<EOS>":
            return out
        if not (char in special_tokens):
            out += char
    return out
