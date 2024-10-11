import numpy as np
from utils.Audio_Processing import preprocess_audio
from utils.Constants import *
from utils.MMS import get_device, MMS, greedyDecoder
from utils.NLP import preprocess_vocab
import torch

############################################################################################


model_path = "./ASR_2_1_220.pth"


############################################################################################


def predict(audio_file):
    device = get_device()

    processed_audios = []
    mel_spec, duration = preprocess_audio(audio_file)
    processed_audios.append(mel_spec)
    padded_audios = [
        (
            mel_spec.shape[-1],
            np.pad(
                mel_spec,
                ((0, 0), (0, N_FRAMES - mel_spec.shape[-1])),
                mode="constant",
            ),
        )
        for mel_spec in processed_audios
    ]

    char2idx, idx2char, vocab_size = preprocess_vocab()

    # load model

    model = MMS(
        vocab_size=vocab_size,
        max_encoder_seq_len=math.ceil(N_FRAMES / 2),
        max_decoder_seq_len=MAX_SEQ_LEN,
        num_encoder_layers=2,
        num_decoder_layers=1,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
    )

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    result = greedyDecoder(
        model, padded_audios[0][1], padded_audios[0][0], char2idx, idx2char, device
    )

    return result
