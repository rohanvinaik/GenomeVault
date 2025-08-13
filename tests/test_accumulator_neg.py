from genomevault.zk_proofs.core.accumulator import INITIAL_ACC, step, verify_chain


def test_accumulator_drop_link_fails():
    # build small chain
    acc = INITIAL_ACC
    chain = []
    for k in range(3):
        pc = bytes([k]) * 32
        vk = bytes([255 - k]) * 32
        acc = step(acc, pc, vk)
        chain.append({"proof_commit": pc.hex(), "vk_hash": vk.hex(), "proof_id": str(k)})

    assert verify_chain(chain, acc.hex())
    assert not verify_chain(chain[:-1], acc.hex())
