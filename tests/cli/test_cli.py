import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner


@pytest.fixture
def app(monkeypatch):
    """Load CLI app with stubbed dependencies."""
    gv_pkg = types.ModuleType("genomevault")
    gv_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "genomevault", gv_pkg)

    hvt_pkg = types.ModuleType("genomevault.hypervector_transform")
    hvt_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "genomevault.hypervector_transform", hvt_pkg)

    zk_pkg = types.ModuleType("genomevault.zk_proofs")
    zk_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "genomevault.zk_proofs", zk_pkg)

    class FakeEncoder:
        def __init__(self, dimension: int = 10000) -> None:
            self.dimension = dimension

        def encode(self, data):
            return np.ones(self.dimension)

        def encode_genomic_variants(self, variants):  # pragma: no cover - simple stub
            return np.ones(self.dimension)

    class FakeProver:
        def prove_variant(self, public_input, private_input):
            class Proof:
                def dict(self) -> dict:
                    return {"proof": "stub"}

            return Proof()

        def generate_proof(self, **kwargs):  # pragma: no cover - simple stub
            class Proof:
                def dict(self) -> dict:
                    return {"proof": "stub"}

            return Proof()

    class FakeVerifier:
        def verify_variant(self, proof, public_input):
            return True

        def verify(self, **kwargs):  # pragma: no cover - simple stub
            return True

    encoder_mod = types.ModuleType("genomevault.hypervector_transform.hdc_encoder")
    encoder_mod.HypervectorEncoder = FakeEncoder
    monkeypatch.setitem(
        sys.modules, "genomevault.hypervector_transform.hdc_encoder", encoder_mod
    )
    setattr(hvt_pkg, "hdc_encoder", encoder_mod)

    prover_mod = types.ModuleType("genomevault.zk_proofs.prover")
    prover_mod.Prover = FakeProver
    monkeypatch.setitem(sys.modules, "genomevault.zk_proofs.prover", prover_mod)
    setattr(zk_pkg, "prover", prover_mod)

    verifier_mod = types.ModuleType("genomevault.zk_proofs.verifier")
    verifier_mod.Verifier = FakeVerifier
    monkeypatch.setitem(sys.modules, "genomevault.zk_proofs.verifier", verifier_mod)
    setattr(zk_pkg, "verifier", verifier_mod)

    spec = importlib.util.spec_from_file_location(
        "cli_main", Path(__file__).resolve().parents[2] / "genomevault" / "cli" / "main.py"
    )
    cli_main = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(cli_main)
    return cli_main.app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_encode_happy(app, runner):
    result = runner.invoke(app, ["encode", "--data", "{\"a\":1}", "--dimension", "4"])
    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["dimension"] == 4
    assert output["type"] == "hypervector"
    assert len(output["vector"]) == 4


def test_encode_error(app, runner):
    result = runner.invoke(app, ["encode"])
    assert result.exit_code != 0
    output = json.loads(result.stdout)
    assert "error" in output


def test_sim_happy(app, runner, tmp_path):
    v1 = tmp_path / "v1.json"
    v2 = tmp_path / "v2.json"
    json.dump({"vector": [0, 1, 0, 1]}, v1.open("w"))
    json.dump({"vector": [0, 1, 0, 1]}, v2.open("w"))
    result = runner.invoke(app, ["sim", "--v1", str(v1), "--v2", str(v2)])
    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert "similarity" in output


def test_sim_error(app, runner, tmp_path):
    v1 = tmp_path / "v1.json"
    v2 = tmp_path / "v2.json"
    json.dump({"vector": [0, 1]}, v1.open("w"))
    json.dump({"vector": [0, 1]}, v2.open("w"))
    result = runner.invoke(
        app, ["sim", "--v1", str(v1), "--v2", str(v2), "--metric", "unknown"]
    )
    assert result.exit_code != 0
    output = json.loads(result.stdout)
    assert "error" in output


def test_prove_happy(app, runner, tmp_path):
    pub = tmp_path / "pub.json"
    priv = tmp_path / "priv.json"
    json.dump({}, pub.open("w"))
    json.dump({}, priv.open("w"))
    result = runner.invoke(app, ["prove", "--public", str(pub), "--private", str(priv)])
    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["success"] is True
    assert output["circuit_type"] == "variant"


def test_prove_error(app, runner, tmp_path):
    priv = tmp_path / "priv.json"
    json.dump({}, priv.open("w"))
    missing_public = tmp_path / "missing.json"
    result = runner.invoke(
        app, ["prove", "--public", str(missing_public), "--private", str(priv)]
    )
    assert result.exit_code != 0
    output = json.loads(result.stdout)
    assert "error" in output


def test_verify_happy(app, runner, tmp_path):
    public = tmp_path / "public.json"
    proof = tmp_path / "proof.json"
    json.dump({}, public.open("w"))
    json.dump({"proof": {"data": "proof"}, "circuit_type": "variant"}, proof.open("w"))
    result = runner.invoke(app, ["verify", "--proof", str(proof), "--public", str(public)])
    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["valid"] is True


def test_verify_error(app, runner, tmp_path):
    public = tmp_path / "public.json"
    invalid_proof = tmp_path / "proof.json"
    json.dump({}, public.open("w"))
    invalid_proof.write_text("{invalid")
    result = runner.invoke(app, ["verify", "--proof", str(invalid_proof), "--public", str(public)])
    assert result.exit_code != 0
    output = json.loads(result.stdout)
    assert "error" in output
