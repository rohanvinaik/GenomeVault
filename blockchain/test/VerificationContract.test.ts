import { expect } from "chai";
import { ethers } from "hardhat";
import { VerificationContract } from "../typechain-types";

describe("VerificationContract", function () {
  let verificationContract: VerificationContract;
  let owner: any;
  let addr1: any;
  let addr2: any;

  beforeEach(async function () {
    // Get signers
    [owner, addr1, addr2] = await ethers.getSigners();

    // Deploy contract
    const VerificationContractFactory = await ethers.getContractFactory("VerificationContract");
    verificationContract = await VerificationContractFactory.deploy();
    await verificationContract.waitForDeployment();
  });

  describe("Deployment", function () {
    it("Should set the right owner", async function () {
      expect(await verificationContract.owner()).to.equal(owner.address);
    });

    it("Should initialize allowed circuits", async function () {
      expect(await verificationContract.allowedCircuits("variant_presence")).to.be.true;
      expect(await verificationContract.allowedCircuits("polygenic_risk_score")).to.be.true;
      expect(await verificationContract.allowedCircuits("ancestry_composition")).to.be.true;
    });
  });

  describe("Proof Recording", function () {
    it("Should record a proof and emit ProofRecorded event", async function () {
      const proofKey = ethers.keccak256(ethers.toUtf8Bytes("test-proof-1"));
      const proofDataHash = ethers.keccak256(ethers.toUtf8Bytes("proof-data"));
      const publicInputsHash = ethers.keccak256(ethers.toUtf8Bytes("public-inputs"));
      const circuitType = "variant_presence";

      // Record proof and check event
      await expect(
        verificationContract.connect(addr1).recordProof(
          proofKey,
          proofDataHash,
          publicInputsHash,
          circuitType
        )
      )
        .to.emit(verificationContract, "ProofRecorded")
        .withArgs(proofKey, addr1.address, circuitType, await ethers.provider.getBlock("latest").then(b => b!.timestamp));

      // Verify proof was recorded
      const proof = await verificationContract.getProof(proofKey);
      expect(proof.prover).to.equal(addr1.address);
      expect(proof.proofHash).to.equal(proofDataHash);
      expect(proof.publicInputsHash).to.equal(publicInputsHash);
      expect(proof.circuitType).to.equal(circuitType);
      expect(proof.verified).to.be.false;
    });

    it("Should not allow recording with disallowed circuit type", async function () {
      const proofKey = ethers.keccak256(ethers.toUtf8Bytes("test-proof-2"));
      const proofDataHash = ethers.keccak256(ethers.toUtf8Bytes("proof-data"));
      const publicInputsHash = ethers.keccak256(ethers.toUtf8Bytes("public-inputs"));
      const circuitType = "invalid_circuit";

      await expect(
        verificationContract.recordProof(
          proofKey,
          proofDataHash,
          publicInputsHash,
          circuitType
        )
      ).to.be.revertedWith("Circuit type not allowed");
    });

    it("Should not allow duplicate proof keys", async function () {
      const proofKey = ethers.keccak256(ethers.toUtf8Bytes("test-proof-3"));
      const proofDataHash = ethers.keccak256(ethers.toUtf8Bytes("proof-data"));
      const publicInputsHash = ethers.keccak256(ethers.toUtf8Bytes("public-inputs"));
      const circuitType = "variant_presence";

      // First recording should succeed
      await verificationContract.recordProof(
        proofKey,
        proofDataHash,
        publicInputsHash,
        circuitType
      );

      // Second recording with same key should fail
      await expect(
        verificationContract.recordProof(
          proofKey,
          proofDataHash,
          publicInputsHash,
          circuitType
        )
      ).to.be.revertedWith("Proof already recorded");
    });
  });

  describe("Proof Verification", function () {
    it("Should verify a proof and emit ProofVerified event", async function () {
      const proofKey = ethers.keccak256(ethers.toUtf8Bytes("test-proof-4"));
      const proofDataHash = ethers.keccak256(ethers.toUtf8Bytes("proof-data"));
      const publicInputsHash = ethers.keccak256(ethers.toUtf8Bytes("public-inputs"));
      const circuitType = "variant_presence";

      // Record proof first
      await verificationContract.recordProof(
        proofKey,
        proofDataHash,
        publicInputsHash,
        circuitType
      );

      // Verify proof
      await expect(
        verificationContract.connect(addr2).verifyProof(
          proofKey,
          true,
          "Valid proof"
        )
      )
        .to.emit(verificationContract, "ProofVerified")
        .withArgs(proofKey, true, addr2.address, await ethers.provider.getBlock("latest").then(b => b!.timestamp));

      // Check verification result
      const verification = await verificationContract.getVerification(proofKey);
      expect(verification.valid).to.be.true;
      expect(verification.verifier).to.equal(addr2.address);
      expect(verification.reason).to.equal("Valid proof");

      // Check proof status
      const [exists, verified, valid] = await verificationContract.checkProofStatus(proofKey);
      expect(exists).to.be.true;
      expect(verified).to.be.true;
      expect(valid).to.be.true;
    });
  });

  describe("Statistics", function () {
    it("Should track prover proof count", async function () {
      const proofKey1 = ethers.keccak256(ethers.toUtf8Bytes("test-proof-5"));
      const proofKey2 = ethers.keccak256(ethers.toUtf8Bytes("test-proof-6"));
      const proofDataHash = ethers.keccak256(ethers.toUtf8Bytes("proof-data"));
      const publicInputsHash = ethers.keccak256(ethers.toUtf8Bytes("public-inputs"));
      const circuitType = "variant_presence";

      // Initial count should be 0
      expect(await verificationContract.getProverStats(addr1.address)).to.equal(0);

      // Record two proofs
      await verificationContract.connect(addr1).recordProof(
        proofKey1,
        proofDataHash,
        publicInputsHash,
        circuitType
      );
      await verificationContract.connect(addr1).recordProof(
        proofKey2,
        proofDataHash,
        publicInputsHash,
        circuitType
      );

      // Count should be 2
      expect(await verificationContract.getProverStats(addr1.address)).to.equal(2);
    });

    it("Should track circuit usage count", async function () {
      const circuitType = "polygenic_risk_score";
      
      // Initial count
      const initialCount = await verificationContract.getCircuitStats(circuitType);

      // Record a proof
      const proofKey = ethers.keccak256(ethers.toUtf8Bytes("test-proof-7"));
      await verificationContract.recordProof(
        proofKey,
        ethers.keccak256(ethers.toUtf8Bytes("proof-data")),
        ethers.keccak256(ethers.toUtf8Bytes("public-inputs")),
        circuitType
      );

      // Count should increase by 1
      expect(await verificationContract.getCircuitStats(circuitType)).to.equal(initialCount + 1n);
    });
  });

  describe("Circuit Management", function () {
    it("Should allow owner to add new circuits", async function () {
      const newCircuit = "new_circuit_type";
      
      // Initially should not be allowed
      expect(await verificationContract.allowedCircuits(newCircuit)).to.be.false;

      // Add circuit
      await expect(verificationContract.addCircuit(newCircuit))
        .to.emit(verificationContract, "CircuitAdded")
        .withArgs(newCircuit);

      // Should now be allowed
      expect(await verificationContract.allowedCircuits(newCircuit)).to.be.true;
    });

    it("Should not allow non-owner to add circuits", async function () {
      await expect(
        verificationContract.connect(addr1).addCircuit("new_circuit")
      ).to.be.revertedWith("Only owner can add circuits");
    });
  });
});
