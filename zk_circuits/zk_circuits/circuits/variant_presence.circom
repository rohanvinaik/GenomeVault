pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/comparators.circom";
include "../node_modules/circomlib/circuits/poseidon.circom";
include "./lib/poseidon.circom";
include "./lib/merkle.circom";

// Complete genomic variant presence proof with full security guarantees
template VariantPresence(merkle_levels) {
    // Public inputs (known to verifier)
    signal input variant_hash;           // Hash of the variant being queried
    signal input database_commitment;    // Commitment to the genomic database
    signal input merkle_root;           // Merkle root of the variant database
    signal input nullifier;             // Prevents double-spending/queries

    // Private inputs (known only to prover)
    signal input chromosome;            // Chromosome number (1-24)
    signal input position;             // Genomic position
    signal input ref_allele;           // Reference allele (encoded)
    signal input alt_allele;           // Alternative allele (encoded)
    signal input sample_id;            // Sample identifier
    signal input commitment_randomness; // Randomness used in commitment
    signal input merkle_proof[merkle_levels];   // Merkle inclusion proof
    signal input merkle_indices[merkle_levels]; // Merkle path indices
    signal input witness_randomness;    // Additional randomness for privacy

    // Public outputs
    signal output variant_present;      // 1 if variant is present, 0 otherwise
    signal output privacy_nullifier;   // Prevents linkability

    // Step 1: Verify the variant hash matches the provided variant data
    component variant_hasher = VariantHasher();
    variant_hasher.chromosome <== chromosome;
    variant_hasher.position <== position;
    variant_hasher.ref_allele <== ref_allele;
    variant_hasher.alt_allele <== alt_allele;
    variant_hasher.sample_id <== sample_id;

    component hash_check = IsEqual();
    hash_check.in[0] <== variant_hasher.hash;
    hash_check.in[1] <== variant_hash;

    // Step 2: Verify the variant is included in the committed database
    component merkle_verification = GenomicMerkleTree(merkle_levels, 1);
    merkle_verification.variants[0][0] <== chromosome;
    merkle_verification.variants[0][1] <== position;
    merkle_verification.variants[0][2] <== ref_allele;
    merkle_verification.variants[0][3] <== alt_allele;
    merkle_verification.merkle_root <== merkle_root;
    merkle_verification.query_variant[0] <== chromosome;
    merkle_verification.query_variant[1] <== position;
    merkle_verification.query_variant[2] <== ref_allele;
    merkle_verification.query_variant[3] <== alt_allele;

    for (var i = 0; i < merkle_levels; i++) {
        merkle_verification.merkle_proof[i] <== merkle_proof[i];
        merkle_verification.merkle_indices[i] <== merkle_indices[i];
    }

    // Step 3: Verify database commitment
    component commitment_hasher = GenomeVaultPoseidon(3);
    commitment_hasher.inputs[0] <== merkle_root;
    commitment_hasher.inputs[1] <== sample_id;
    commitment_hasher.inputs[2] <== commitment_randomness;

    component commitment_check = IsEqual();
    commitment_check.in[0] <== commitment_hasher.out;
    commitment_check.in[1] <== database_commitment;

    // Step 4: Generate privacy nullifier to prevent linkability
    component nullifier_generator = GenomeVaultPoseidon(4);
    nullifier_generator.inputs[0] <== nullifier;
    nullifier_generator.inputs[1] <== variant_hash;
    nullifier_generator.inputs[2] <== sample_id;
    nullifier_generator.inputs[3] <== witness_randomness;
    privacy_nullifier <== nullifier_generator.out;

    // Step 5: Final verification - all conditions must be satisfied
    signal verification_steps[3];
    verification_steps[0] <== hash_check.out;
    verification_steps[1] <== merkle_verification.inclusion_verified;
    verification_steps[2] <== commitment_check.out;

    // AND all verification steps
    signal partial_and;
    partial_and <== verification_steps[0] * verification_steps[1];
    variant_present <== partial_and * verification_steps[2];
}

// Main component with 20 levels (supports up to 2^20 = ~1M variants)
component main {public [variant_hash, database_commitment, merkle_root, nullifier]} = VariantPresence(20);
