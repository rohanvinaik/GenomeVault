// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.19;

/**
 * @title GenomeVault Verification Contract
 * @notice Manages zero-knowledge proof verification and genomic data attestations
 */
contract VerificationContract {
    
    // Events
    event ProofSubmitted(
        address indexed submitter,
        bytes32 indexed proofHash,
        string circuitType,
        uint256 timestamp
    );
    
    event ProofVerified(
        bytes32 indexed proofHash,
        bool isValid,
        address indexed verifier
    );
    
    event AttestationIssued(
        address indexed subject,
        bytes32 indexed attestationHash,
        string attestationType,
        uint256 expiry
    );
    
    // Structs
    struct Proof {
        address submitter;
        bytes32 proofHash;
        string circuitType;
        uint256 timestamp;
        bool verified;
        mapping(address => bool) verifiers;
        uint256 verificationCount;
    }
    
    struct Attestation {
        address subject;
        bytes32 dataHash;
        string attestationType;
        uint256 issuedAt;
        uint256 expiresAt;
        bool revoked;
    }
    
    // State variables
    mapping(bytes32 => Proof) public proofs;
    mapping(bytes32 => Attestation) public attestations;
    mapping(address => bytes32[]) public userProofs;
    mapping(address => bytes32[]) public userAttestations;
    
    // Circuit type registry
    mapping(string => bool) public validCircuitTypes;
    
    // Verification threshold
    uint256 public constant VERIFICATION_THRESHOLD = 3;
    
    // Owner
    address public owner;
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier validCircuit(string memory circuitType) {
        require(validCircuitTypes[circuitType], "Invalid circuit type");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        
        // Initialize valid circuit types
        validCircuitTypes["variant_v1"] = true;
        validCircuitTypes["risk_score_v1"] = true;
        validCircuitTypes["pathway_v1"] = true;
        validCircuitTypes["phenotype_v1"] = true;
    }
    
    /**
     * @notice Submit a zero-knowledge proof
     * @param proofData The proof data (384 bytes)
     * @param circuitType The type of circuit used
     * @return proofHash The hash of the submitted proof
     */
    function submitProof(
        bytes calldata proofData,
        string calldata circuitType
    ) external validCircuit(circuitType) returns (bytes32) {
        require(proofData.length == 384, "Invalid proof size");
        
        bytes32 proofHash = keccak256(abi.encodePacked(
            msg.sender,
            proofData,
            circuitType,
            block.timestamp
        ));
        
        Proof storage proof = proofs[proofHash];
        proof.submitter = msg.sender;
        proof.proofHash = proofHash;
        proof.circuitType = circuitType;
        proof.timestamp = block.timestamp;
        proof.verified = false;
        proof.verificationCount = 0;
        
        userProofs[msg.sender].push(proofHash);
        
        emit ProofSubmitted(msg.sender, proofHash, circuitType, block.timestamp);
        
        return proofHash;
    }
    
    /**
     * @notice Verify a submitted proof (called by validator nodes)
     * @param proofHash The hash of the proof to verify
     * @param isValid Whether the proof is valid
     */
    function verifyProof(bytes32 proofHash, bool isValid) external {
        Proof storage proof = proofs[proofHash];
        require(proof.timestamp > 0, "Proof does not exist");
        require(!proof.verifiers[msg.sender], "Already verified by this address");
        
        proof.verifiers[msg.sender] = true;
        
        if (isValid) {
            proof.verificationCount++;
            
            if (proof.verificationCount >= VERIFICATION_THRESHOLD && !proof.verified) {
                proof.verified = true;
                emit ProofVerified(proofHash, true, msg.sender);
            }
        } else {
            emit ProofVerified(proofHash, false, msg.sender);
        }
    }
    
    /**
     * @notice Issue an attestation based on verified proofs
     * @param subject The address to receive the attestation
     * @param dataHash Hash of the attestation data
     * @param attestationType Type of attestation
     * @param validityPeriod How long the attestation is valid (seconds)
     * @return attestationHash The hash of the issued attestation
     */
    function issueAttestation(
        address subject,
        bytes32 dataHash,
        string calldata attestationType,
        uint256 validityPeriod
    ) external onlyOwner returns (bytes32) {
        bytes32 attestationHash = keccak256(abi.encodePacked(
            subject,
            dataHash,
            attestationType,
            block.timestamp
        ));
        
        Attestation storage attestation = attestations[attestationHash];
        attestation.subject = subject;
        attestation.dataHash = dataHash;
        attestation.attestationType = attestationType;
        attestation.issuedAt = block.timestamp;
        attestation.expiresAt = block.timestamp + validityPeriod;
        attestation.revoked = false;
        
        userAttestations[subject].push(attestationHash);
        
        emit AttestationIssued(
            subject,
            attestationHash,
            attestationType,
            attestation.expiresAt
        );
        
        return attestationHash;
    }
    
    /**
     * @notice Revoke an attestation
     * @param attestationHash The hash of the attestation to revoke
     */
    function revokeAttestation(bytes32 attestationHash) external onlyOwner {
        Attestation storage attestation = attestations[attestationHash];
        require(attestation.issuedAt > 0, "Attestation does not exist");
        require(!attestation.revoked, "Already revoked");
        
        attestation.revoked = true;
    }
    
    /**
     * @notice Check if an attestation is valid
     * @param attestationHash The hash of the attestation to check
     * @return isValid Whether the attestation is currently valid
     */
    function isAttestationValid(bytes32 attestationHash) external view returns (bool) {
        Attestation storage attestation = attestations[attestationHash];
        
        return attestation.issuedAt > 0 &&
               !attestation.revoked &&
               block.timestamp <= attestation.expiresAt;
    }
    
    /**
     * @notice Get user's proof history
     * @param user The user's address
     * @return Array of proof hashes
     */
    function getUserProofs(address user) external view returns (bytes32[] memory) {
        return userProofs[user];
    }
    
    /**
     * @notice Get user's attestations
     * @param user The user's address
     * @return Array of attestation hashes
     */
    function getUserAttestations(address user) external view returns (bytes32[] memory) {
        return userAttestations[user];
    }
    
    /**
     * @notice Add a new valid circuit type
     * @param circuitType The circuit type to add
     */
    function addCircuitType(string calldata circuitType) external onlyOwner {
        validCircuitTypes[circuitType] = true;
    }
    
    /**
     * @notice Transfer ownership
     * @param newOwner The new owner address
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}
