import pytest
import torch
from transformers import WhisperConfig

from wge.data import Data


@pytest.fixture
def tiny_config():
    return WhisperConfig(
        vocab_size=100, num_mel_bins=80, d_model=32, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
        max_source_positions=50, max_target_positions=64, decoder_start_token_id=1, pad_token_id=0,
        bos_token_id=1, eos_token_id=2, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
    )


@pytest.fixture
def tiny_data():
    g = torch.Generator().manual_seed(0)
    n = 16
    lens = torch.randint(4, 12, (n,), generator=g)
    labels = torch.full((n, 12), -100, dtype=torch.long)
    for i, k in enumerate(lens):
        labels[i, :k] = torch.randint(3, 100, (int(k),), generator=g)
    return Data(torch.randn(n, 80, 100, generator=g), labels, [str(i) for i in range(n)], {})
