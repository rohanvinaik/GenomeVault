// Simple Circom circuit test to verify integration
import path from "path";
import { dirname } from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe("GenomeVault Circuit Integration", function () {
    this.timeout(10000);

    it("Should have access to circomlib components", async () => {
        console.log("✅ Testing circomlib availability...");

        // Check that circomlib is available
        const circomlibPath = path.join(__dirname, "../node_modules/circomlib");

        if (!fs.existsSync(circomlibPath)) {
            throw new Error("❌ Circomlib not found in node_modules");
        }

        console.log("✅ Circomlib found at:", circomlibPath);

        // Check for key circomlib components
        const poseidonPath = path.join(circomlibPath, "circuits/poseidon.circom");
        const comparatorsPath = path.join(circomlibPath, "circuits/comparators.circom");

        if (!fs.existsSync(poseidonPath)) {
            throw new Error("❌ Poseidon circuit not found in circomlib");
        }

        if (!fs.existsSync(comparatorsPath)) {
            throw new Error("❌ Comparators circuit not found in circomlib");
        }

        console.log("✅ Key circomlib circuits found");
        console.log("   - Poseidon:", poseidonPath);
        console.log("   - Comparators:", comparatorsPath);

        return true;
    });

    it("Should compile a simple test circuit using circomlib", async () => {
        console.log("✅ Testing simple circuit compilation...");

        // Create a minimal test circuit that uses circomlib
        const testCircuitCode = `
pragma circom 2.0.0;

include "../node_modules/circomlib/circuits/comparators.circom";

template SimpleTest() {
    signal input a;
    signal input b;
    signal output equal;

    component eq = IsEqual();
    eq.in[0] <== a;
    eq.in[1] <== b;
    equal <== eq.out;
}

component main = SimpleTest();
        `;

        const testCircuitPath = path.join(__dirname, "simple_test.circom");
        fs.writeFileSync(testCircuitPath, testCircuitCode);

        try {
            // Note: Using dynamic import for circom_tester as it may not be ES module compatible
            const { wasm: wasm_tester } = await import("circom_tester");
            const circuit = await wasm_tester(testCircuitPath);

            // Test the circuit with some inputs
            const witness = await circuit.calculateWitness({ a: 5, b: 5 });
            await circuit.checkConstraints(witness);

            // Verify output (should be 1 since 5 == 5)
            const output = witness[1]; // Output is at index 1
            if (output !== 1n) {
                throw new Error(`❌ Expected output 1, got ${output}`);
            }

            console.log("✅ Simple circuit compiled and executed successfully");
            return true;
        } finally {
            // Clean up test file
            if (fs.existsSync(testCircuitPath)) {
                fs.unlinkSync(testCircuitPath);
            }
        }
    });
});

// If running directly (not via test framework)
if (import.meta.url === `file://${process.argv[1]}`) {
    console.log("🧪 GenomeVault Circom Integration Test");
    console.log("=====================================");

    const runTests = async () => {
        try {
            console.log("✅ Testing circomlib availability...");

            // Check that circomlib is available
            const circomlibPath = path.join(__dirname, "../node_modules/circomlib");

            if (!fs.existsSync(circomlibPath)) {
                throw new Error("❌ Circomlib not found in node_modules");
            }

            console.log("✅ Circomlib found at:", circomlibPath);

            // Check for key circomlib components
            const poseidonPath = path.join(circomlibPath, "circuits/poseidon.circom");
            const comparatorsPath = path.join(circomlibPath, "circuits/comparators.circom");

            console.log("🔍 Checking key circuits...");
            console.log(`   Poseidon: ${fs.existsSync(poseidonPath) ? "✅" : "❌"}`);
            console.log(`   Comparators: ${fs.existsSync(comparatorsPath) ? "✅" : "❌"}`);

            if (!fs.existsSync(poseidonPath) || !fs.existsSync(comparatorsPath)) {
                throw new Error("❌ Required circomlib circuits not found");
            }

            console.log("");
            console.log("🎯 INTEGRATION TEST RESULT: SUCCESS");
            console.log("   ✅ Circomlib is properly installed");
            console.log("   ✅ Key circuits are available");
            console.log("   ✅ GenomeVault ZK circuits can use circomlib");

        } catch (error) {
            console.error("");
            console.error("❌ INTEGRATION TEST RESULT: FAILED");
            console.error("   Error:", error.message);
            process.exit(1);
        }
    };

    runTests();
}
