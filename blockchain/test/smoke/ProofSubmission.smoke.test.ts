import { expect } from "chai";
import { ethers } from "hardhat";
import { VerificationContract } from "../../typechain-types";

describe("Smoke Test: Proof Submission", function () {
  let verificationContract: VerificationContract;
  let prover: any;

  before(async function () {
    // Get signers
    [, prover] = await ethers.getSigners();

    // Deploy contract
    const VerificationContractFactory = await ethers.getContractFactory("VerificationContract");
    verificationContract = await VerificationContractFactory.deploy();
    await verificationContract.waitForDeployment();
  });

  it("Should successfully submit a proof and emit ProofRecorded event", async function () {
    const proofKey = ethers.keccak256(ethers.toUtf8Bytes("smoke-test-proof"));
    const proofDataHash = ethers.keccak256(ethers.toUtf8Bytes("test-proof-data"));
    const publicInputsHash = ethers.keccak256(ethers.toUtf8Bytes("test-public-inputs"));
    const circuitType = "variant_presence";

    // Submit proof and verify event emission
    const tx = await verificationContract.connect(prover).recordProof(
      proofKey,
      proofDataHash,
      publicInputsHash,
      circuitType
    );

    // Wait for transaction confirmation
    const receipt = await tx.wait();

    // Verify event was emitted
    expect(receipt).to.not.be.null;
    expect(receipt!.logs.length).to.be.greaterThan(0);

    // Check the ProofRecorded event
    const event = receipt!.logs.find(
      (log) => {
        try {
          const parsed = verificationContract.interface.parseLog({
            topics: log.topics as string[],
            data: log.data
          });
          return parsed?.name === "ProofRecorded";
        } catch {
          return false;
        }
      }
    );

    expect(event).to.not.be.undefined;
    console.log("✅ Smoke Test Passed: ProofRecorded event successfully fired!");
  });
});
