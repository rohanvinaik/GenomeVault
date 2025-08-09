import pandas as pd


from genomevault.pipelines.etl import run_etl, transform


def test_transform_standardizes_and_renames():
    df = pd.DataFrame(
        {
            "Sample": ["s1"],
            "Chromosome": ["chr1"],
            "Position": [10],
            "Ref": ["A"],
            "Alt": ["T"],
        }
    )
    out = transform(df)
    assert set(out.columns) >= {"sample_id", "chrom", "pos", "ref", "alt"}


def test_run_etl_validates(tmp_path):
    csv = tmp_path / "in.csv"
    csv.write_text("sample_id,chrom,pos,ref,alt\ns1,chr1,10,A,T\n", encoding="utf-8")
    # write minimal contract
    cjson = tmp_path / "contract.json"
    cjson.write_text(
        '{"name":"t","columns":[{"name":"sample_id","dtype":"string","required":true,"allow_null":false},{"name":"chrom","dtype":"string","required":true,"allow_null":false},{"name":"pos","dtype":"int","required":true,"allow_null":false},{"name":"ref","dtype":"string","required":true,"allow_null":false},{"name":"alt","dtype":"string","required":true,"allow_null":false}], "unique_key": ["sample_id","chrom","pos"]}',
        encoding="utf-8",
    )
    res = run_etl(str(csv), contract_path=str(cjson))
    assert res["ok"] is True and res["rows"] == 1
