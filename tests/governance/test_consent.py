from genomevault.governance.consent.store import ConsentStore



def test_consent_grant_revoke_check():
    cs = ConsentStore()
    cs.grant("subj1", "research", ttl_days=1)
    assert cs.has_consent("subj1", "research") is True
    cs.revoke("subj1", "research")
    assert cs.has_consent("subj1", "research") is False
