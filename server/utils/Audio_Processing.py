import librosa
import numpy as np
from utils.Constants import *


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

    mel_scale_sgram = librosa.feature.melspectrogram(
        S=sgram_mag, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

    del audio_data, modified_audio, sgram, mel_scale_sgram

    return mel_sgram, duration