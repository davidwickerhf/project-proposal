from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

from PIL import Image

from src.detection.statistical import (
    calibration_chi_square_score,
    chi_square_dct_score,
    chi_square_spatial_score,
    rs_analysis_score,
    sample_pairs_score,
)
from src.detection.srnet import SRNetModelArtifact, SRNetTrainingInput, score_srnet_model, train_srnet_model
from src.embedding.dct import embed_dct_lsb_jpeg
from src.embedding.encryption import decrypt_payload_aes_256_cbc, encrypt_payload_aes_256_cbc
from src.embedding.lsb import embed_lsb


def test_statistical_detectors_accept_in_memory_artifacts() -> None:
    for fn in [rs_analysis_score, chi_square_spatial_score, sample_pairs_score]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        assert len(params) == 1
        assert params[0].name == "image"

    for fn in [chi_square_dct_score, calibration_chi_square_score]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        assert params[0].name == "jpeg_bytes"


def test_embedding_and_encryption_contracts_are_closed_loop() -> None:
    for fn in [
        encrypt_payload_aes_256_cbc,
        decrypt_payload_aes_256_cbc,
        embed_lsb,
        embed_dct_lsb_jpeg,
    ]:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            assert "path" not in p.name.lower()


def test_srnet_extension_contract_uses_input_output_artifacts() -> None:
    train_sig = inspect.signature(train_srnet_model)
    train_params = list(train_sig.parameters.values())
    assert len(train_params) == 1
    assert train_params[0].name == "training_input"

    score_sig = inspect.signature(score_srnet_model)
    score_params = list(score_sig.parameters.values())
    assert [p.name for p in score_params] == ["model", "x_samples"]

    train_input = SRNetTrainingInput(
        method="lsb",
        fold=0,
        x_train=[[0.1, 0.2]],
        y_train=[0],
        x_val=[[0.1, 0.2]],
        y_val=[1],
    )
    model = SRNetModelArtifact(method="lsb", fold=0, model_state=None, hyperparams={})

    assert train_input.method == "lsb"
    assert model.fold == 0


def test_contracts_do_not_require_path_objects() -> None:
    for fn in [
        encrypt_payload_aes_256_cbc,
        decrypt_payload_aes_256_cbc,
        embed_lsb,
        embed_dct_lsb_jpeg,
        rs_analysis_score,
        chi_square_spatial_score,
        sample_pairs_score,
        chi_square_dct_score,
        calibration_chi_square_score,
        train_srnet_model,
        score_srnet_model,
    ]:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            ann = get_type_hints(fn).get(p.name)
            assert ann is not Path

    assert get_type_hints(rs_analysis_score)["image"] is Image.Image
