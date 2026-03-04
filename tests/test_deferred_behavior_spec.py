from __future__ import annotations

import random
from typing import Callable

import pytest
from PIL import Image

from src.detection.srm import SRMTrainingInput, extract_srm_features, score_srm_ec_model, train_srm_ec_model
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


def _call_or_xfail(fn: Callable, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.xfail(f"Deferred implementation pending: {exc}")


def _sample_payload(n: int = 128, seed: int = 7) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(n))


def _make_cover(size: tuple[int, int] = (64, 64)) -> Image.Image:
    img = Image.new("RGB", size)
    pix = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            v = (x * 3 + y * 5) % 256
            pix[x, y] = (v, (v + 83) % 256, (v + 157) % 256)
    return img


def _assert_float(x: object) -> None:
    assert isinstance(x, float)
    assert x == x  # NaN check
    assert abs(x) != float("inf")


def test_encrypt_roundtrip_and_determinism_spec() -> None:
    payload = _sample_payload(256)
    key = b"k" * 32
    iv = b"i" * 16

    c1 = _call_or_xfail(encrypt_payload_aes_256_cbc, payload, key, iv)
    c2 = _call_or_xfail(encrypt_payload_aes_256_cbc, payload, key, iv)

    assert isinstance(c1, bytes)
    assert c1 == c2
    assert c1 != payload

    p1 = _call_or_xfail(decrypt_payload_aes_256_cbc, c1, key, iv)
    assert p1 == payload


def test_encrypt_rejects_invalid_key_or_iv_spec() -> None:
    payload = _sample_payload(32)

    try:
        encrypt_payload_aes_256_cbc(payload, b"short", b"i" * 16)
    except NotImplementedError as exc:
        pytest.xfail(f"Deferred implementation pending: {exc}")
    except (ValueError, AssertionError, TypeError):
        pass
    else:
        pytest.fail("Encryption should reject invalid key length")

    try:
        encrypt_payload_aes_256_cbc(payload, b"k" * 32, b"short")
    except NotImplementedError as exc:
        pytest.xfail(f"Deferred implementation pending: {exc}")
    except (ValueError, AssertionError, TypeError):
        pass
    else:
        pytest.fail("Encryption should reject invalid IV length")


def test_embed_lsb_contract_and_determinism_spec() -> None:
    cover = _make_cover()
    cover_before = cover.tobytes()
    payload = _sample_payload(64)

    s1 = _call_or_xfail(embed_lsb, cover, payload, "low", 12345)
    s2 = _call_or_xfail(embed_lsb, cover, payload, "low", 12345)
    s3 = _call_or_xfail(embed_lsb, cover, payload, "high", 12345)

    assert isinstance(s1, Image.Image)
    assert s1.size == cover.size
    assert s1.mode == cover.mode
    assert cover.tobytes() == cover_before  # input image not mutated in-place
    assert s1.tobytes() == s2.tobytes()  # deterministic with same parameters
    assert s1.tobytes() != s3.tobytes()  # payload-level should alter embedding intensity


def test_embed_dct_contract_and_determinism_spec() -> None:
    cover = _make_cover()
    cover_before = cover.tobytes()
    payload = _sample_payload(64)

    s1 = _call_or_xfail(embed_dct_qim, cover, payload, "low", 20.0)
    s2 = _call_or_xfail(embed_dct_qim, cover, payload, "low", 20.0)
    s3 = _call_or_xfail(embed_dct_qim, cover, payload, "high", 20.0)

    assert isinstance(s1, Image.Image)
    assert s1.size == cover.size
    assert s1.mode == cover.mode
    assert cover.tobytes() == cover_before
    assert s1.tobytes() == s2.tobytes()
    assert s1.tobytes() != s3.tobytes()


def test_extract_srm_features_returns_deterministic_float_list_spec() -> None:
    cover = _make_cover()

    f1 = _call_or_xfail(extract_srm_features, cover)
    f2 = _call_or_xfail(extract_srm_features, cover)

    assert isinstance(f1, list)
    assert len(f1) > 0
    assert all(isinstance(v, float) for v in f1)
    assert f1 == f2  # deterministic


def test_statistical_detectors_return_finite_deterministic_scores_spec() -> None:
    cover = _make_cover()

    for fn in [rs_analysis_score, chi_square_score, block_dct_shift_score]:
        a = _call_or_xfail(fn, cover)
        b = _call_or_xfail(fn, cover)
        _assert_float(a)
        _assert_float(b)
        assert a == b


def test_statistical_detectors_signal_change_on_modified_image_spec() -> None:
    clean = _make_cover()
    modified = clean.copy()
    pix = modified.load()
    for y in range(modified.size[1]):
        for x in range(0, modified.size[0], 2):
            r, g, b = pix[x, y]
            pix[x, y] = (r ^ 1, g, b)

    for fn in [rs_analysis_score, chi_square_score, block_dct_shift_score]:
        c = _call_or_xfail(fn, clean)
        m = _call_or_xfail(fn, modified)
        _assert_float(c)
        _assert_float(m)
        assert c != m


def test_srm_train_score_shape_and_determinism_spec() -> None:
    training_input = SRMTrainingInput(
        method="lsb",
        fold=0,
        x_train=[[-2.0, -1.0], [-1.5, -0.8], [1.2, 0.9], [2.0, 1.1]],
        y_train=[0, 0, 1, 1],
        x_val=[[-1.2, -1.1], [1.4, 1.2]],
        y_val=[0, 1],
        random_seed=42,
    )

    model_a = _call_or_xfail(train_srm_ec_model, training_input)
    model_b = _call_or_xfail(train_srm_ec_model, training_input)

    assert model_a.method == "lsb"
    assert model_a.fold == 0
    assert model_b.method == model_a.method

    x_test = [[-1.0, -0.9], [1.5, 1.3], [0.0, 0.1]]
    scores_a = _call_or_xfail(score_srm_ec_model, model_a, x_test)
    scores_b = _call_or_xfail(score_srm_ec_model, model_b, x_test)

    assert isinstance(scores_a, list)
    assert len(scores_a) == len(x_test)
    assert all(isinstance(s, float) for s in scores_a)
    assert scores_a == scores_b


def test_srm_rejects_mismatched_train_shapes_spec() -> None:
    bad_input = SRMTrainingInput(
        method="dct",
        fold=1,
        x_train=[[0.0, 0.1], [1.0, 1.1]],
        y_train=[0],
        x_val=[[0.2, 0.3]],
        y_val=[1],
        random_seed=42,
    )

    try:
        train_srm_ec_model(bad_input)
    except NotImplementedError as exc:
        pytest.xfail(f"Deferred implementation pending: {exc}")
    except (ValueError, AssertionError):
        pass
    else:
        pytest.fail("SRM training should reject mismatched x/y lengths")
