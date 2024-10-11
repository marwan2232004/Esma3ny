from utils.Transformer import *
from utils.Constants import *
import gc
import numpy as np
from utils.NLP import TextDecoder, tokenize_text


def get_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    return torch.device("cpu")


def get_conv_Lout(L_in, conv):
    return math.floor(
        (L_in + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1)
        / conv.stride[0]
        + 1
    )


class MMS(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, max_encoder_seq_len=100, max_decoder_seq_len=100, n_mels=N_MELS,
                 dropout=0.1):
        super(MMS, self).__init__()

        self.transformer = Transformer(d_model=d_model,
                                       num_heads=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
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

    def get_encoder_seq_len(self, L_in):
        return get_conv_Lout(get_conv_Lout(L_in, self.conv1), self.conv2)

    def forward(self, audio, text, encoder_self_attn_mask, decoder_self_attn_mask, decoder_padding_mask,
                device):

        audio = self.gelu(self.conv1(audio))
        audio = self.gelu(self.conv2(audio))

        audio = audio.permute(0, 2, 1)

        en_positional_encoding = self.en_positional_encoding().to(device)

        assert audio.shape[1:] == en_positional_encoding.shape[1:], "incorrect audio shape"

        audio += en_positional_encoding

        text = self.embedding(text)

        de_positional_encoding = self.de_positional_encoding().to(device)

        assert text.shape[1:] == de_positional_encoding.shape[1:], "incorrect text shape"

        text += de_positional_encoding

        out = self.transformer(src=audio, tgt=text,
                               encoder_self_attention_mask=encoder_self_attn_mask,
                               decoder_self_attention_mask=decoder_self_attn_mask,
                               decoder_cross_attention_mask=decoder_padding_mask)

        out = self.ff(out)

        del de_positional_encoding, en_positional_encoding
        gc.collect()

        return out


def generate_padding_masks(
        transcription, audio_original_len, conv_func, frames=N_FRAMES
):
    batch_size, seq_len = transcription.size()
    audio_len = conv_func(frames)

    look_ahead_mask = torch.full((seq_len, seq_len), True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)

    encoder_self_attn_mask = torch.full([batch_size, audio_len, audio_len], False)
    decoder_padding_self_attn_mask = torch.full([batch_size, seq_len, seq_len], False)
    decoder_padding_cross_attn_mask = torch.full(
        [batch_size, seq_len, audio_len], False
    )

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

    decoder_self_attention_mask = torch.where(
        look_ahead_mask + decoder_padding_self_attn_mask, NEG_INFTY, 0
    )
    decoder_cross_attention_mask = torch.where(
        decoder_padding_cross_attn_mask, NEG_INFTY, 0
    )
    encoder_self_attn_mask = torch.where(encoder_self_attn_mask, NEG_INFTY, 0)

    return (
        encoder_self_attn_mask,
        decoder_self_attention_mask,
        decoder_cross_attention_mask,
    )


def greedyDecoder(model, audio, org_len, char2idx, idx2char, device):
    with torch.no_grad():
        transcription = ""

        audio = [torch.tensor(aud) for aud in audio]
        audio = torch.stack(audio)
        audio = audio.unsqueeze(0)
        audio = audio.to(device)

        for i in range(MAX_TEXT_LEN):

            transcription_padded = torch.tensor(
                tokenize_text(
                    transcription, char2idx, max_len=MAX_TEXT_LEN, end_token=False
                ),
                dtype=torch.long,
            )

            transcription_padded = transcription_padded.unsqueeze(0).to(device)

            encoder_self_attn_mask, decoder_self_attn_mask, decoder_padding_mask = (
                generate_padding_masks(
                    transcription_padded, [org_len], model.get_encoder_seq_len
                )
            )

            decoder_self_attn_mask = decoder_self_attn_mask.to(device)
            decoder_padding_mask = decoder_padding_mask.to(device)
            encoder_self_attn_mask = encoder_self_attn_mask.to(device)
            # Forward pass
            output = model(
                audio,
                transcription_padded,
                encoder_self_attn_mask,
                decoder_self_attn_mask,
                decoder_padding_mask,
                device
            )

            output = F.softmax(output, dim=-1)

            next_token = torch.argmax(output[0, i, :], dim=-1).unsqueeze(
                0
            )  # Get last token prediction

            transcription += TextDecoder(next_token, idx2char)

            # If an end-of-sequence token is predicted, stop
            if next_token.item() == char2idx["<EOS>"]:
                break

        return transcription
