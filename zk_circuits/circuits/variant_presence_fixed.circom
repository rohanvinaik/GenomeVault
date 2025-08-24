pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/comparators.circom";
include "./lib/poseidon.circom";

template VariantPresence(maxVariants) {
    signal input variants[maxVariants][3];
    signal input query[3];
    signal output found;
    signal output variant_hash;
    
    // Hash the query variant
    component hasher = Poseidon(3);
    hasher.inputs[0] <== query[0];
    hasher.inputs[1] <== query[1];
    hasher.inputs[2] <== query[2];
    variant_hash <== hasher.out;
    
    // Check each variant
    component isEqual[maxVariants][3];
    signal matches[maxVariants];
    
    for (var i = 0; i < maxVariants; i++) {
        var match = 1;
        for (var j = 0; j < 3; j++) {
            isEqual[i][j] = IsEqual();
            isEqual[i][j].in[0] <== variants[i][j];
            isEqual[i][j].in[1] <== query[j];
            match = match * isEqual[i][j].out;
        }
        matches[i] <== match;
    }
    
    // OR all matches
    signal accumulated[maxVariants];
    accumulated[0] <== matches[0];
    
    for (var i = 1; i < maxVariants; i++) {
        accumulated[i] <== accumulated[i-1] + matches[i] - accumulated[i-1] * matches[i];
    }
    
    found <== accumulated[maxVariants - 1];
}

component main = VariantPresence(100);
