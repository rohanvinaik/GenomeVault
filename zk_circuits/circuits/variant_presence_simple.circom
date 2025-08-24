pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/comparators.circom";
include "./lib/poseidon.circom";

template VariantPresence() {
    // Simple variant matching without complex loops
    signal input variant_chr;
    signal input variant_pos;
    signal input variant_alt;

    signal input query_chr;
    signal input query_pos;
    signal input query_alt;

    signal output found;
    signal output variant_hash;

    // Hash the query variant using Poseidon
    component hasher = Poseidon(3);
    hasher.inputs[0] <== query_chr;
    hasher.inputs[1] <== query_pos;
    hasher.inputs[2] <== query_alt;
    variant_hash <== hasher.out;

    // Check if variant matches query
    component chrEq = IsEqual();
    component posEq = IsEqual();
    component altEq = IsEqual();

    chrEq.in[0] <== variant_chr;
    chrEq.in[1] <== query_chr;

    posEq.in[0] <== variant_pos;
    posEq.in[1] <== query_pos;

    altEq.in[0] <== variant_alt;
    altEq.in[1] <== query_alt;

    // AND all comparisons (quadratic constraints only)
    signal chrPosMatch;
    chrPosMatch <== chrEq.out * posEq.out;

    found <== chrPosMatch * altEq.out;
}

component main = VariantPresence();
