from genomevault.pipelines.e2e_pipeline import run_e2e



def test_e2e_pipeline_smoke():
    out = run_e2e()
    assert isinstance(out, dict) and set(out.keys()) == {
        "encoded",
        "pir",
        "proof",
        "tx",
    }
