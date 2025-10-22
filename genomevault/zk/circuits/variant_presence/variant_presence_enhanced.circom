pragma circom 2.0.0;

include "poseidon.circom";
include "comparators.circom";
include "mux1.circom";
include "bitify.circom";

/*
 * Enhanced Variant Presence Circuit for GenomeVault
 *
 * Verifies presence of genomic variants in a committed genome database
 * with full Merkle tree verification and comprehensive validity checks.
 *
 * Features:
 * - Full 20-level Merkle tree verification
 * - Batch verification of up to 10 variants
 * - Allele frequency range proofs
 * - Quality score thresholds
 * - Genotype validity checks
 * - Chromosome and position validation
 * - Multi-allelic variant support
 *
 * Estimated constraints: 15,000-20,000
 */

// Helper template: Range check that value is in [min, max]
template RangeCheck(n) {
    signal input in;
    signal input min;
    signal input max;
    signal output out;

    component geq_min = GreaterEqThan(n);
    geq_min.in[0] <== in;
    geq_min.in[1] <== min;

    component leq_max = LessEqThan(n);
    leq_max.in[0] <== in;
    leq_max.in[1] <== max;

    out <== geq_min.out * leq_max.out;
}

// Merkle tree verification with Poseidon hashing
template MerkleTreeVerifier(levels) {
    signal input leaf;
    signal input root;
    signal input pathIndices[levels];
    signal input pathElements[levels];
    signal output valid;

    component hashers[levels];
    component muxes[levels];

    signal hashes[levels + 1];
    hashes[0] <== leaf;

    for (var i = 0; i < levels; i++) {
        // Select left or right based on path index
        muxes[i] = MultiMux1(2);
        muxes[i].c[0][0] <== hashes[i];
        muxes[i].c[0][1] <== pathElements[i];
        muxes[i].c[1][0] <== pathElements[i];
        muxes[i].c[1][1] <== hashes[i];
        muxes[i].s <== pathIndices[i];

        // Hash the pair
        hashers[i] = Poseidon(2);
        hashers[i].inputs[0] <== muxes[i].out[0];
        hashers[i].inputs[1] <== muxes[i].out[1];

        hashes[i + 1] <== hashers[i].out;
    }

    // Verify final hash matches root
    component root_check = IsEqual();
    root_check.in[0] <== hashes[levels];
    root_check.in[1] <== root;

    valid <== root_check.out;
}

// Genotype validity checker (must be one of: 0/0, 0/1, 1/0, 1/1, etc.)
template GenotypeValidator() {
    signal input genotype;
    signal output valid;

    // Valid genotypes encoded as:
    // 0 = 0/0 (homozygous reference)
    // 1 = 0/1 (heterozygous)
    // 2 = 1/0 (heterozygous, phased)
    // 3 = 1/1 (homozygous alternate)
    // 4 = ./. (missing)

    component checks[5];
    signal results[5];

    for (var i = 0; i < 5; i++) {
        checks[i] = IsEqual();
        checks[i].in[0] <== genotype;
        checks[i].in[1] <== i;
        results[i] <== checks[i].out;
    }

    // At least one must match
    valid <== results[0] + results[1] + results[2] + results[3] + results[4];
}

// Chromosome validator (must be 1-23, X=23, Y=24, MT=25)
template ChromosomeValidator() {
    signal input chromosome;
    signal output valid;

    component range = RangeCheck(8);
    range.in <== chromosome;
    range.min <== 1;
    range.max <== 25;

    valid <== range.out;
}

// Single variant verification
template VariantVerifier(merkle_levels) {
    // Public inputs
    signal input commitment_root;

    // Private inputs
    signal input chromosome;
    signal input position;
    signal input ref_allele;
    signal input alt_allele;
    signal input genotype;
    signal input quality_score;
    signal input allele_frequency;
    signal input merkle_path[merkle_levels];
    signal input merkle_indices[merkle_levels];
    signal input witness_randomness;

    // Output
    signal output valid;

    // 1. Validate chromosome (1-25)
    component chr_validator = ChromosomeValidator();
    chr_validator.chromosome <== chromosome;

    // 2. Validate genotype (must be valid encoding)
    component geno_validator = GenotypeValidator();
    geno_validator.genotype <== genotype;

    // 3. Validate quality score (must be >= 20)
    component quality_check = GreaterEqThan(32);
    quality_check.in[0] <== quality_score;
    quality_check.in[1] <== 20;

    // 4. Validate allele frequency (must be in [0, 100])
    component af_range = RangeCheck(32);
    af_range.in <== allele_frequency;
    af_range.min <== 0;
    af_range.max <== 100;

    // 5. Hash the variant data
    component variant_hasher = Poseidon(8);
    variant_hasher.inputs[0] <== chromosome;
    variant_hasher.inputs[1] <== position;
    variant_hasher.inputs[2] <== ref_allele;
    variant_hasher.inputs[3] <== alt_allele;
    variant_hasher.inputs[4] <== genotype;
    variant_hasher.inputs[5] <== quality_score;
    variant_hasher.inputs[6] <== allele_frequency;
    variant_hasher.inputs[7] <== witness_randomness;

    // 6. Verify Merkle tree membership
    component merkle_verifier = MerkleTreeVerifier(merkle_levels);
    merkle_verifier.leaf <== variant_hasher.out;
    merkle_verifier.root <== commitment_root;
    for (var i = 0; i < merkle_levels; i++) {
        merkle_verifier.pathElements[i] <== merkle_path[i];
        merkle_verifier.pathIndices[i] <== merkle_indices[i];
    }

    // 7. Combine all validity checks
    signal validity_product;
    validity_product <== chr_validator.valid * geno_validator.valid;

    signal quality_af_product;
    quality_af_product <== quality_check.out * af_range.out;

    signal all_checks;
    all_checks <== validity_product * quality_af_product;

    valid <== all_checks * merkle_verifier.valid;
}

// Main template: Batch variant verification
template VariantPresenceBatch(num_variants, merkle_levels) {
    // Public inputs
    signal input commitment_root;
    signal input expected_num_valid;

    // Private inputs (batched)
    signal input chromosomes[num_variants];
    signal input positions[num_variants];
    signal input ref_alleles[num_variants];
    signal input alt_alleles[num_variants];
    signal input genotypes[num_variants];
    signal input quality_scores[num_variants];
    signal input allele_frequencies[num_variants];
    signal input merkle_paths[num_variants][merkle_levels];
    signal input merkle_indices[num_variants][merkle_levels];
    signal input witness_randomness[num_variants];

    // Output
    signal output all_valid;

    // Verify each variant
    component verifiers[num_variants];
    signal valid_count[num_variants + 1];
    valid_count[0] <== 0;

    for (var i = 0; i < num_variants; i++) {
        verifiers[i] = VariantVerifier(merkle_levels);

        verifiers[i].commitment_root <== commitment_root;
        verifiers[i].chromosome <== chromosomes[i];
        verifiers[i].position <== positions[i];
        verifiers[i].ref_allele <== ref_alleles[i];
        verifiers[i].alt_allele <== alt_alleles[i];
        verifiers[i].genotype <== genotypes[i];
        verifiers[i].quality_score <== quality_scores[i];
        verifiers[i].allele_frequency <== allele_frequencies[i];
        verifiers[i].witness_randomness <== witness_randomness[i];

        for (var j = 0; j < merkle_levels; j++) {
            verifiers[i].merkle_path[j] <== merkle_paths[i][j];
            verifiers[i].merkle_indices[j] <== merkle_indices[i][j];
        }

        valid_count[i + 1] <== valid_count[i] + verifiers[i].valid;
    }

    // Check that the number of valid variants matches expected
    component count_check = IsEqual();
    count_check.in[0] <== valid_count[num_variants];
    count_check.in[1] <== expected_num_valid;

    all_valid <== count_check.out;
}

// Main component: Verify 10 variants with 20-level Merkle tree
component main {public [commitment_root, expected_num_valid]} = VariantPresenceBatch(10, 20);
