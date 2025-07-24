// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/**
 * @title VerificationContract
 * @dev Records and verifies zero-knowledge proofs on-chain
 * Part of GenomeVault 3.0 blockchain layer
 */
contract VerificationContract {
    
    // Proof record structure
    struct ProofRecord {
        address prover;           // Address that submitted the proof
        bytes32 proofHash;        // Hash of the proof data
        bytes32 publicInputsHash; // Hash of public inputs
        uint256 timestamp;        // Block timestamp when recorded
        string circuitType;       // Type of circuit used
        bool verified;            // Whether proof has been verified
        uint256 verificationTime; // When verification occurred
    }
    
    // Verification result
    struct VerificationResult {
        bool valid;
        address verifier;
        uint256 timestamp;
        string reason;
    }
    
    // Storage
    mapping(bytes32 => ProofRecord) public proofs;
    mapping(bytes32 => VerificationResult) public verifications;
    mapping(address => uint256) public proverProofCount;
    mapping(string => uint256) public circuitUsageCount;
    
    // Circuit registry
    mapping(string => bool) public allowedCircuits;
    
    // Events
    event ProofRecorded(
        bytes32 indexed proofKey,
        address indexed prover,
        string circuitType,
        uint256 timestamp
    );
    
    event ProofVerified(
        bytes32 indexed proofKey,
        bool valid,
        address indexed verifier,
        uint256 timestamp
    );
    
    event CircuitAdded(string circuitType);
    event CircuitRemoved(string circuitType);
    
    // Modifiers
    modifier onlyAllowedCircuit(string memory circuitType) {
        require(allowedCircuits[circuitType], "Circuit type not allowed");
        _;
    }
    
    modifier proofNotExists(bytes32 proofKey) {
        require(proofs[proofKey].timestamp == 0, "Proof already recorded");
        _;
    }
    
    modifier proofExists(bytes32 proofKey) {
        require(proofs[proofKey].timestamp > 0, "Proof does not exist");
        _;
    }
    
    // Owner (for circuit management)
    address public owner;
    
    constructor() {
        owner = msg.sender;
        
        // Initialize allowed circuits
        _addCircuit("variant_presence");
        _addCircuit("polygenic_risk_score");
        _addCircuit("ancestry_composition");
        _addCircuit("pharmacogenomic");
        _addCircuit("pathway_enrichment");
        _addCircuit("diabetes_risk_alert");
    }
    
    /**
     * @dev Record a new proof
     * @param proofKey Unique identifier for the proof
     * @param proofDataHash Hash of the actual proof data
     * @param publicInputsHash Hash of public inputs
     * @param circuitType Type of circuit used
     */
    function recordProof(
        bytes32 proofKey,
        bytes32 proofDataHash,
        bytes32 publicInputsHash,
        string memory circuitType
    ) external onlyAllowedCircuit(circuitType) proofNotExists(proofKey) {
        
        // Create proof record
        proofs[proofKey] = ProofRecord({
            prover: msg.sender,
            proofHash: proofDataHash,
            publicInputsHash: publicInputsHash,
            timestamp: block.timestamp,
            circuitType: circuitType,
            verified: false,
            verificationTime: 0
        });
        
        // Update counters
        proverProofCount[msg.sender]++;
        circuitUsageCount[circuitType]++;
        
        // Emit event
        emit ProofRecorded(proofKey, msg.sender, circuitType, block.timestamp);
    }
    
    /**
     * @dev Verify a recorded proof
     * @param proofKey Proof identifier
     * @param isValid Whether the proof is valid
     * @param reason Verification reason/notes
     */
    function verifyProof(
        bytes32 proofKey,
        bool isValid,
        string memory reason
    ) external proofExists(proofKey) {
        
        ProofRecord storage proof = proofs[proofKey];
        require(!proof.verified, "Proof already verified");
        
        // Record verification
        verifications[proofKey] = VerificationResult({
            valid: isValid,
            verifier: msg.sender,
            timestamp: block.timestamp,
            reason: reason
        });
        
        // Update proof record
        proof.verified = true;
        proof.verificationTime = block.timestamp;
        
        // Emit event
        emit ProofVerified(proofKey, isValid, msg.sender, block.timestamp);
    }
    
    /**
     * @dev Batch record multiple proofs
     * @param proofKeys Array of proof identifiers
     * @param proofDataHashes Array of proof data hashes
     * @param publicInputsHashes Array of public input hashes
     * @param circuitTypes Array of circuit types
     */
    function batchRecordProofs(
        bytes32[] memory proofKeys,
        bytes32[] memory proofDataHashes,
        bytes32[] memory publicInputsHashes,
        string[] memory circuitTypes
    ) external {
        require(
            proofKeys.length == proofDataHashes.length &&
            proofKeys.length == publicInputsHashes.length &&
            proofKeys.length == circuitTypes.length,
            "Array length mismatch"
        );
        
        for (uint i = 0; i < proofKeys.length; i++) {
            if (proofs[proofKeys[i]].timestamp == 0 && allowedCircuits[circuitTypes[i]]) {
                recordProof(
                    proofKeys[i],
                    proofDataHashes[i],
                    publicInputsHashes[i],
                    circuitTypes[i]
                );
            }
        }
    }
    
    /**
     * @dev Get proof details
     * @param proofKey Proof identifier
     * @return Proof record
     */
    function getProof(bytes32 proofKey) external view returns (ProofRecord memory) {
        return proofs[proofKey];
    }
    
    /**
     * @dev Get verification result
     * @param proofKey Proof identifier
     * @return Verification result
     */
    function getVerification(bytes32 proofKey) external view returns (VerificationResult memory) {
        return verifications[proofKey];
    }
    
    /**
     * @dev Check if proof exists and is verified
     * @param proofKey Proof identifier
     * @return exists Whether proof exists
     * @return verified Whether proof is verified
     * @return valid Whether proof is valid
     */
    function checkProofStatus(bytes32 proofKey) external view returns (
        bool exists,
        bool verified,
        bool valid
    ) {
        ProofRecord memory proof = proofs[proofKey];
        exists = proof.timestamp > 0;
        verified = proof.verified;
        valid = verified && verifications[proofKey].valid;
    }
    
    /**
     * @dev Add allowed circuit type (owner only)
     * @param circuitType Circuit type to add
     */
    function addCircuit(string memory circuitType) external {
        require(msg.sender == owner, "Only owner can add circuits");
        _addCircuit(circuitType);
    }
    
    /**
     * @dev Remove allowed circuit type (owner only)
     * @param circuitType Circuit type to remove
     */
    function removeCircuit(string memory circuitType) external {
        require(msg.sender == owner, "Only owner can remove circuits");
        allowedCircuits[circuitType] = false;
        emit CircuitRemoved(circuitType);
    }
    
    /**
     * @dev Internal function to add circuit
     * @param circuitType Circuit type to add
     */
    function _addCircuit(string memory circuitType) internal {
        allowedCircuits[circuitType] = true;
        emit CircuitAdded(circuitType);
    }
    
    /**
     * @dev Get statistics for a prover
     * @param prover Prover address
     * @return proofCount Number of proofs submitted
     */
    function getProverStats(address prover) external view returns (uint256 proofCount) {
        return proverProofCount[prover];
    }
    
    /**
     * @dev Get circuit usage statistics
     * @param circuitType Circuit type
     * @return usageCount Number of times used
     */
    function getCircuitStats(string memory circuitType) external view returns (uint256 usageCount) {
        return circuitUsageCount[circuitType];
    }
}
