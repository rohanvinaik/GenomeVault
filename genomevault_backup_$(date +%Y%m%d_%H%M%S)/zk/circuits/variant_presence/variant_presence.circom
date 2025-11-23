pragma circom 2.0.0;

include "poseidon.circom";
include "comparators.circom";

template VariantPresence() {
    // Public inputs
    signal input variant_hash;
    signal input reference_hash;
    signal input commitment_root;

    // Private inputs
    signal input chr;
    signal input position;
    signal input ref_allele;
    signal input alt_allele;
    signal input merkle_proof[20];
    signal input merkle_indices[20];
    signal input witness_randomness;

    // Output
    signal output valid;

    // Hash the variant data
    component hasher = Poseidon(5);
    hasher.inputs[0] <== chr;
    hasher.inputs[1] <== position;
    hasher.inputs[2] <== ref_allele;
    hasher.inputs[3] <== alt_allele;
    hasher.inputs[4] <== witness_randomness;

    // Verify the hash matches the public input
    component eq = IsEqual();
    eq.in[0] <== hasher.out;
    eq.in[1] <== variant_hash;

    // Simplified Merkle proof verification
    // In production, would verify full Merkle path
    signal current_hash;
    current_hash <== hasher.out;

    // Check against commitment root (simplified)
    component root_check = IsEqual();
    root_check.in[0] <== commitment_root;
    root_check.in[1] <== reference_hash + current_hash; // Simplified

    valid <== eq.out * root_check.out;
}

component main {public [variant_hash, reference_hash, commitment_root]} = VariantPresence();
