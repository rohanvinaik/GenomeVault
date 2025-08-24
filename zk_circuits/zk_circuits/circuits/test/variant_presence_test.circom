pragma circom 2.0.0;

include "../variant_presence.circom";

// Test component with smaller tree for faster testing
component main = VariantPresence(8); // 2^8 = 256 variants for testing
