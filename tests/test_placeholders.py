from __future__ import annotations

import pytest
from PIL import Image

from src.detection.srm import (
    SRMModelArtifact,
    SRMTrainingInput,
    extract_srm_features,
    score_srm_ec_model,
    train_srm_ec_model,
)
from src.detection.statistical import (
    block_dct_shift_score,
    chi_square_score,
    rs_analysis_score,
)
from src.embedding.dct import embed_dct_qim
from src.embedding.encryption import (
    decrypt_payload_aes_256_cbc,
    encrypt_payload_aes_256_cbc,
)
from src.embedding.lsb import embed_lsb


@pytest.mark.parametrize(
    ("fn", "args", "match"),
    [
        (
            encrypt_payload_aes_256_cbc,
            (b"abc", b"k" * 32, b"i" * 16),
            "AES-256-CBC encryption",
        ),
        (
            decrypt_payload_aes_256_cbc,
            (b"abc", b"k" * 32, b"i" * 16),
            "AES-256-CBC decryption",
        ),
        (
            embed_lsb,
            (Image.new("RGB", (8, 8), color=(0, 0, 0)), b"abc", "low", 123),
            "LSB embedding",
        ),
        (
            embed_dct_qim,
            (Image.new("RGB", (8, 8), color=(0, 0, 0)), b"abc", "low", 20.0),
            "DCT-QIM embedding",
        ),
        (
            extract_srm_features,
            (Image.new("RGB", (8, 8), color=(0, 0, 0)),),
            "SRM feature extraction",
        ),
        (rs_analysis_score, (Image.new("RGB", (8, 8), color=(0, 0, 0)),), "RS analysis"),
        (chi_square_score, (Image.new("RGB", (8, 8), color=(0, 0, 0)),), "Chi-square"),
        (
            block_dct_shift_score,
            (Image.new("RGB", (8, 8), color=(0, 0, 0)),),
            "Block-DCT shift test",
        ),
        (
            train_srm_ec_model,
            (
                SRMTrainingInput(
                    method="lsb",
                    fold=0,
                    x_train=[[0.1, 0.2]],
                    y_train=[0],
                    x_val=[[0.1, 0.2]],
                    y_val=[1],
                ),
            ),
            r"SRM\+EC training",
        ),
        (
            score_srm_ec_model,
            (
                SRMModelArtifact(method="lsb", fold=0, model_state=None, hyperparams={}),
                [[0.1, 0.2]],
            ),
            r"SRM\+EC inference",
        ),
    ],
)
def test_deferred_functions_raise_not_implemented(fn, args, match: str) -> None:
    with pytest.raises(NotImplementedError, match=match):
        fn(*args)
