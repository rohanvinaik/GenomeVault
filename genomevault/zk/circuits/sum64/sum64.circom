pragma circom 2.1.6;

template Sum64() {
    signal input a;
    signal input b;
    signal input c; // public
    signal output ok;

    ok <== (a + b === c);
}

component main = Sum64();
