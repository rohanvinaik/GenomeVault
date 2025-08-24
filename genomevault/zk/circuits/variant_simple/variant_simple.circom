pragma circom 2.1.6;

template VariantSimple() {
    // Public inputs
    signal input variant_hash;
    signal input threshold;

    // Private inputs
    signal input chr;
    signal input position;
    signal input score;

    // Output
    signal output valid;

    // Simple computation: check if score > threshold
    component gt = GreaterThan();
    gt.a <== score;
    gt.b <== threshold;

    // Hash constraint (simplified)
    signal computed_hash;
    computed_hash <== chr * 1000000 + position;

    // Output validity
    valid <== gt.out;
}

template GreaterThan() {
    signal input a;
    signal input b;
    signal output out;

    // Simple comparison (in real circuit would use bit decomposition)
    component isZero = IsZero();
    isZero.in <== b - a;
    out <== 1 - isZero.out;
}

template IsZero() {
    signal input in;
    signal output out;

    signal inv;

    // If in is zero, inv can be anything and out = 1
    // If in is non-zero, inv = 1/in and out = 0
    inv <-- in == 0 ? 0 : 1/in;

    out <== 1 - in * inv;

    // Constraint to ensure correctness
    in * (1 - out) === 0;
}

component main {public [variant_hash, threshold]} = VariantSimple();
