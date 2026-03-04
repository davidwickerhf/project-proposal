from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

from PIL import Image

from src.detection.srm import SRMModelArtifact, SRMTrainingInput, extract_srm_features, score_srm_ec_model, train_srm_ec_model
from src.detection.statistical import block_dct_shift_score, chi_square_score, rs_analysis_score
from src.embedding.dct import embed_dct_qim
from src.embedding.encryption import decrypt_payload_aes_256_cbc, encrypt_payload_aes_256_cbc
from src.embedding.lsb import embed_lsb


def test_extract_srm_features_accepts_in_memory_image() -> None:
    sig = inspect.signature(extract_srm_features)
    params = list(sig.parameters.values())
    assert len(params) == 1
    assert params[0].name == "image"
    assert get_type_hints(extract_srm_features)["image"] is Image.Image
    ret = get_type_hints(extract_srm_features)["return"]
    assert getattr(ret, "__origin__", ret) is list


def test_statistical_detectors_accept_in_memory_images() -> None:
    for fn in [rs_analysis_score, chi_square_score, block_dct_shift_score]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        assert len(params) == 1
        assert params[0].name == "image"


def test_embedding_and_encryption_contracts_are_closed_loop() -> None:
    for fn in [encrypt_payload_aes_256_cbc, decrypt_payload_aes_256_cbc, embed_lsb, embed_dct_qim]:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            assert "path" not in p.name.lower()


def test_srm_contract_uses_input_output_artifacts() -> None:
    train_sig = inspect.signature(train_srm_ec_model)
    train_params = list(train_sig.parameters.values())
    assert len(train_params) == 1
    assert train_params[0].name == "training_input"

    score_sig = inspect.signature(score_srm_ec_model)
    score_params = list(score_sig.parameters.values())
    assert [p.name for p in score_params] == ["model", "x_samples"]

    # Smoke instantiation for contract dataclasses.
    train_input = SRMTrainingInput(
        method="lsb",
        fold=0,
        x_train=[[0.1, 0.2]],
        y_train=[0],
        x_val=[[0.1, 0.2]],
        y_val=[1],
    )
    model = SRMModelArtifact(method="lsb", fold=0, model_state=None, hyperparams={})

    assert train_input.method == "lsb"
    assert model.fold == 0


def test_contracts_do_not_require_path_objects() -> None:
    # Enforce no deferred function takes a ``Path`` argument.
    for fn in [
        encrypt_payload_aes_256_cbc,
        decrypt_payload_aes_256_cbc,
        embed_lsb,
        embed_dct_qim,
        rs_analysis_score,
        chi_square_score,
        block_dct_shift_score,
        extract_srm_features,
        train_srm_ec_model,
        score_srm_ec_model,
    ]:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            ann = get_type_hints(fn).get(p.name)
            assert ann is not Path

    # Expected in-memory image type for statistical detector input.
    assert get_type_hints(rs_analysis_score)["image"] is Image.Image
