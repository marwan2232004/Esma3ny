import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import gc
import string
from Transformer import *

############################################################################################


model_path = "D:\\DEPI\\trying ASR\\ASR.pth"
N_ROWS = 50000  # NUMBER OF ROWS TAKEN FROM THE DATA
MAX_TEXT_LEN = 70
MAX_SEQ_LEN = 70
N_MELS = 128
SAMPLE_RATE = 16000  # NUMBER OF SAMPLES PER SECOND
HOP_LENGTH = 512  # THE STEP LENGTH OF THE SLIDING WINDOW , Commonly set to one-fourth of N_FFT
N_FFT = 2048  # NUMBER OF FFT POINTS (WINDOW SIZE) THIS CONTROLS THE RESOLUTION OF THE FREQUENCY DOMAIN ANALYSIS
CHUNK_LENGTH = 15  # 15 SECOND CHUNK
N_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
N_FRAMES = math.ceil(N_SAMPLES / HOP_LENGTH)
N_SAMPLES_PER_TOKEN = 2 * HOP_LENGTH
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # OR N_FRAMES // CHUNK_LENGTH
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN
NEG_INFTY = -1e9
special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']


############################################################################################


def get_conv_Lout(L_in, conv):
    return math.floor(
        (L_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) / conv.stride[0] + 1)


class MMS(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, batch_first=True, max_encoder_seq_len=100, max_decoder_seq_len=100,
                 n_mels=N_MELS,
                 dropout=0.1):
        super(MMS, self).__init__()
        self.transformer = Transformer(d_model=d_model, num_heads=nhead, num_layers=num_encoder_layers,
                                       ffn_hidden=dim_feedforward,
                                       drop_prob=dropout)
        self.en_positional_encoding = PositionalEncoding(d_model, max_encoder_seq_len)
        self.de_positional_encoding = PositionalEncoding(d_model, max_decoder_seq_len)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.d_model = d_model
        self.ff = nn.Linear(d_model, vocab_size)
        self.device = get_device()

    def get_encoder_seq_len(self, L_in):
        return get_conv_Lout(get_conv_Lout(L_in, self.conv1), self.conv2)

    def forward(self, audio, audio_org_lens, text, encoder_self_attn_mask, decoder_self_attn_mask,
                decoder_padding_mask):
        batch_size = audio.size(0)

        audio = self.gelu(self.conv1(audio))
        audio = self.gelu(self.conv2(audio))

        audio = audio.permute(0, 2, 1)

        for i in range(batch_size):
            audio_new_len = self.get_encoder_seq_len(audio_org_lens[i])
            audio[i, audio_new_len:, :] = 0

        de_positional_encoding = self.de_positional_encoding().to(self.device)

        assert audio.shape[1:] == de_positional_encoding.shape[1:], "incorrect audio shape"

        audio += de_positional_encoding

        text = self.embedding(text)

        en_positional_encoding = self.en_positional_encoding().to(self.device)

        assert text.shape[1:] == en_positional_encoding.shape[1:], "incorrect text shape"

        text += en_positional_encoding

        out = self.transformer(src=audio, tgt=text,
                               encoder_self_attention_mask=encoder_self_attn_mask,
                               decoder_self_attention_mask=decoder_self_attn_mask,
                               decoder_cross_attention_mask=decoder_padding_mask)

        out = self.ff(out)

        del de_positional_encoding, en_positional_encoding
        gc.collect()

        return out


def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    return device


def pad_or_trim(array, length=N_SAMPLES, axis=-1, padding=True):
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if padding & (array.shape[axis] < length):
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array


# Function to load and preprocess audio
def preprocess_audio(file_path):
    audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE)

    duration = librosa.get_duration(y=audio_data, sr=SAMPLE_RATE)

    modified_audio = pad_or_trim(audio_data, padding=False)

    sgram = librosa.stft(y=modified_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)

    sgram_mag, _ = librosa.magphase(sgram)

    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                                     n_mels=N_MELS)

    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

    del audio_data, modified_audio, sgram, mel_scale_sgram

    return mel_sgram, duration


def preprocess_vocab():
    # Define the vocabulary: add lowercase letters, digits, and special tokens
    english_characters = list(string.ascii_lowercase + ' ')  # List of English characters
    arabic_characters = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويئءىةؤ")  # List of Arabic characters

    # Combine all characters
    characters = english_characters + arabic_characters
    vocab = special_tokens + characters  # Combine special tokens and characters

    # Build dictionaries for character-to-index and index-to-character
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(vocab)
    return char2idx, idx2char, vocab_size


def tokenize_text(text, char2idx, max_len=MAX_TEXT_LEN, start_token=True, end_token=True):
    for char in text:
        idx = char2idx.get(char, char2idx['<UNK>'])
        if idx == 1:
            print(char)

    tokens = [char2idx.get(char, char2idx['<UNK>']) for char in text]

    max_len -= start_token + end_token  # fot the start and end token

    # pad to a maximum length (for batching)
    if max_len is not None:
        text_len = len(tokens)
        if text_len < max_len:
            if end_token: tokens += [char2idx['<EOS>']]
            tokens += [char2idx['<PAD>']] * (max_len - text_len)  # Pad
        else:
            tokens = tokens[:max_len]  # Truncate if longer than max_len
            if end_token: tokens += [char2idx['<EOS>']]

    if start_token: tokens.insert(0, char2idx['<SOS>'])

    return tokens


def TextDecoder(sentence, idx2char):
    out = ''
    for token in sentence:
        if isinstance(token, torch.Tensor):
            token = token.item()
        char = idx2char[token]
        if char == '<EOS>':
            return out
        if not (char in special_tokens):
            out += char
    return out


def generate_padding_masks(transcription, audio_original_len, conv_func, frames=N_FRAMES):
    batch_size, seq_len = transcription.size()
    audio_len = conv_func(frames)

    look_ahead_mask = torch.full((seq_len, seq_len), True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)

    encoder_self_attn_mask = torch.full([batch_size, audio_len, audio_len], False)
    decoder_padding_self_attn_mask = torch.full([batch_size, seq_len, seq_len], False)
    decoder_padding_cross_attn_mask = torch.full([batch_size, seq_len, audio_len], False)

    for i in range(batch_size):
        audio_new_len = conv_func(audio_original_len[i])
        encoder_self_attn_mask[i, audio_new_len:, :] = True
        encoder_self_attn_mask[i, :, audio_new_len:] = True
        decoder_padding_cross_attn_mask[i, :, audio_new_len:] = True

        zero_indices = np.where(transcription[0].cpu().numpy() == 0)[0]
        if len(zero_indices) > 0:
            idx = zero_indices[0]
            decoder_padding_self_attn_mask[i, idx:, :] = True
            decoder_padding_self_attn_mask[i, :, idx:] = True
            decoder_padding_cross_attn_mask[i, idx:, :] = True

    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_self_attn_mask, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_cross_attn_mask, NEG_INFTY, 0)
    encoder_self_attn_mask = torch.where(encoder_self_attn_mask, NEG_INFTY, 0)

    return encoder_self_attn_mask, decoder_self_attention_mask, decoder_cross_attention_mask


def Convert(model, audio, org_len, char2idx, idx2char, device):
    with torch.no_grad():
        transcription = ""

        audio = [torch.tensor(aud) for aud in audio]
        audio = torch.stack(audio)
        audio = audio.unsqueeze(0)
        audio = audio.to(device)
        print("Audio size = ", audio.size())
        for i in range(MAX_TEXT_LEN):

            transcription_padded = torch.tensor(
                tokenize_text(transcription, char2idx, max_len=MAX_TEXT_LEN, end_token=False), dtype=torch.long)

            transcription_padded = transcription_padded.unsqueeze(0).to(device)

            encoder_self_attn_mask, decoder_self_attn_mask, decoder_padding_mask = generate_padding_masks(
                transcription_padded, [org_len], model.get_encoder_seq_len)

            decoder_self_attn_mask = decoder_self_attn_mask.to(device)
            decoder_padding_mask = decoder_padding_mask.to(device)
            encoder_self_attn_mask = encoder_self_attn_mask.to(device)
            # Forward pass
            output = model(audio, [org_len], transcription_padded, encoder_self_attn_mask, decoder_self_attn_mask,
                           decoder_padding_mask)

            output = F.softmax(output, dim=-1)

            next_token = torch.argmax(output[0, i, :], dim=-1).unsqueeze(0)  # Get last token prediction

            transcription += TextDecoder(next_token, idx2char)

            # If an end-of-sequence token is predicted, stop
            if next_token.item() == char2idx['<EOS>']:
                break

        return transcription


if __name__ == "__main__":
    device = get_device()
    # load model
    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    audio_path = 'D:\\DEPI\\trying ASR\\Test Arabic.mp3'
    processed_audios = []
    mel_spec, duration = preprocess_audio(audio_path)
    processed_audios.append(mel_spec)
    padded_audios = [
        (mel_spec.shape[-1], np.pad(mel_spec, ((0, 0), (0, N_FRAMES - mel_spec.shape[-1])), mode='constant'))
        for mel_spec in processed_audios]
    # print(padded_audios)
    print("--")
    char2idx, idx2char, vocab_size = preprocess_vocab()
    # print(padded_audios[0][1])
    print("--")
    # print(padded_audios[0][0])
    result = Convert(model, padded_audios[0][1], padded_audios[0][0], char2idx, idx2char, device)
    print(result)
