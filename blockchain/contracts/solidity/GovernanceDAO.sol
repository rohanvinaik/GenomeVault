// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.19;

/**
 * @title GenomeVault DAO Governance Contract
 * @notice Implements dual-axis weighted voting for protocol governance
 */
contract GovernanceDAO {
    
    // Events
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        string title,
        uint256 startBlock,
        uint256 endBlock
    );
    
    event VoteCast(
        uint256 indexed proposalId,
        address indexed voter,
        bool support,
        uint256 weight,
        string reason
    );
    
    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCanceled(uint256 indexed proposalId);
    
    event NodeRegistered(
        address indexed nodeAddress,
        uint8 nodeClass,
        bool isSignatory,
        uint256 votingPower
    );
    
    // Enums
    enum ProposalState {
        Pending,
        Active,
        Canceled,
        Defeated,
        Succeeded,
        Queued,
        Expired,
        Executed
    }
    
    enum NodeClass {
        Light,      // c = 1
        Full,       // c = 4
        Archive     // c = 8
    }
    
    // Structs
    struct Node {
        NodeClass nodeClass;
        bool isSignatory;
        uint256 votingPower;    // w = c + s
        bool isActive;
        uint256 registeredBlock;
    }
    
    struct Proposal {
        uint256 id;
        address proposer;
        string title;
        string description;
        bytes callData;
        uint256 startBlock;
        uint256 endBlock;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 abstainVotes;
        bool executed;
        bool canceled;
        mapping(address => bool) hasVoted;
        mapping(address => Vote) votes;
    }
    
    struct Vote {
        bool support;
        uint256 weight;
        string reason;
    }
    
    // Constants
    uint256 public constant VOTING_DELAY = 100;        // ~10 minutes at 6s blocks
    uint256 public constant VOTING_PERIOD = 17280;     // ~3 days
    uint256 public constant PROPOSAL_THRESHOLD = 100;  // Min voting power to propose
    uint256 public constant QUORUM_PERCENTAGE = 30;    // 30% of total voting power
    uint256 public constant SIGNATORY_WEIGHT = 10;     // s = 10 for trusted signatories
    
    // State variables
    mapping(address => Node) public nodes;
    mapping(uint256 => Proposal) public proposals;
    uint256 public proposalCount;
    uint256 public totalVotingPower;
    address public guardian;    // Can cancel malicious proposals
    
    // Modifiers
    modifier onlyRegisteredNode() {
        require(nodes[msg.sender].isActive, "Not a registered node");
        _;
    }
    
    modifier onlyGuardian() {
        require(msg.sender == guardian, "Only guardian can call");
        _;
    }
    
    constructor() {
        guardian = msg.sender;
    }
    
    /**
     * @notice Register a node in the governance system
     * @param nodeClass The hardware class of the node
     * @param isSignatory Whether the node is a trusted signatory
     */
    function registerNode(NodeClass nodeClass, bool isSignatory) external {
        require(!nodes[msg.sender].isActive, "Node already registered");
        
        // Calculate voting power: w = c + s
        uint256 classWeight = getClassWeight(nodeClass);
        uint256 signatoryWeight = isSignatory ? SIGNATORY_WEIGHT : 0;
        uint256 votingPower = classWeight + signatoryWeight;
        
        nodes[msg.sender] = Node({
            nodeClass: nodeClass,
            isSignatory: isSignatory,
            votingPower: votingPower,
            isActive: true,
            registeredBlock: block.number
        });
        
        totalVotingPower += votingPower;
        
        emit NodeRegistered(msg.sender, uint8(nodeClass), isSignatory, votingPower);
    }
    
    /**
     * @notice Create a new governance proposal
     * @param title Short title of the proposal
     * @param description Full description
     * @param callData Encoded function call to execute
     */
    function createProposal(
        string memory title,
        string memory description,
        bytes memory callData
    ) external onlyRegisteredNode returns (uint256) {
        require(
            nodes[msg.sender].votingPower >= PROPOSAL_THRESHOLD,
            "Insufficient voting power to propose"
        );
        
        uint256 proposalId = proposalCount++;
        Proposal storage proposal = proposals[proposalId];
        
        proposal.id = proposalId;
        proposal.proposer = msg.sender;
        proposal.title = title;
        proposal.description = description;
        proposal.callData = callData;
        proposal.startBlock = block.number + VOTING_DELAY;
        proposal.endBlock = proposal.startBlock + VOTING_PERIOD;
        
        emit ProposalCreated(
            proposalId,
            msg.sender,
            title,
            proposal.startBlock,
            proposal.endBlock
        );
        
        return proposalId;
    }
    
    /**
     * @notice Cast a vote on a proposal
     * @param proposalId The ID of the proposal
     * @param support Whether to vote for (true) or against (false)
     * @param reason Optional reason for the vote
     */
    function castVote(
        uint256 proposalId,
        bool support,
        string memory reason
    ) external onlyRegisteredNode {
        Proposal storage proposal = proposals[proposalId];
        require(block.number >= proposal.startBlock, "Voting not started");
        require(block.number <= proposal.endBlock, "Voting ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");
        
        uint256 weight = nodes[msg.sender].votingPower;
        
        proposal.hasVoted[msg.sender] = true;
        proposal.votes[msg.sender] = Vote({
            support: support,
            weight: weight,
            reason: reason
        });
        
        if (support) {
            proposal.forVotes += weight;
        } else {
            proposal.againstVotes += weight;
        }
        
        emit VoteCast(proposalId, msg.sender, support, weight, reason);
    }
    
    /**
     * @notice Execute a successful proposal
     * @param proposalId The ID of the proposal to execute
     */
    function executeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(!proposal.executed, "Already executed");
        require(!proposal.canceled, "Proposal canceled");
        require(block.number > proposal.endBlock, "Voting not ended");
        
        // Check if proposal succeeded
        require(proposal.forVotes > proposal.againstVotes, "Proposal defeated");
        
        // Check quorum
        uint256 quorum = (totalVotingPower * QUORUM_PERCENTAGE) / 100;
        uint256 totalVotes = proposal.forVotes + proposal.againstVotes;
        require(totalVotes >= quorum, "Quorum not reached");
        
        proposal.executed = true;
        
        // Execute the proposal
        (bool success, ) = address(this).call(proposal.callData);
        require(success, "Execution failed");
        
        emit ProposalExecuted(proposalId);
    }
    
    /**
     * @notice Cancel a proposal (guardian only)
     * @param proposalId The ID of the proposal to cancel
     */
    function cancelProposal(uint256 proposalId) external onlyGuardian {
        Proposal storage proposal = proposals[proposalId];
        require(!proposal.executed, "Already executed");
        require(!proposal.canceled, "Already canceled");
        
        proposal.canceled = true;
        emit ProposalCanceled(proposalId);
    }
    
    /**
     * @notice Get the current state of a proposal
     * @param proposalId The ID of the proposal
     */
    function getProposalState(uint256 proposalId) external view returns (ProposalState) {
        Proposal storage proposal = proposals[proposalId];
        
        if (proposal.canceled) {
            return ProposalState.Canceled;
        }
        
        if (proposal.executed) {
            return ProposalState.Executed;
        }
        
        if (block.number < proposal.startBlock) {
            return ProposalState.Pending;
        }
        
        if (block.number <= proposal.endBlock) {
            return ProposalState.Active;
        }
        
        // Voting ended, check results
        if (proposal.forVotes <= proposal.againstVotes) {
            return ProposalState.Defeated;
        }
        
        // Check quorum
        uint256 quorum = (totalVotingPower * QUORUM_PERCENTAGE) / 100;
        uint256 totalVotes = proposal.forVotes + proposal.againstVotes;
        
        if (totalVotes < quorum) {
            return ProposalState.Defeated;
        }
        
        return ProposalState.Succeeded;
    }
    
    /**
     * @notice Get the voting power of a node
     * @param nodeAddress The address of the node
     */
    function getVotingPower(address nodeAddress) external view returns (uint256) {
        return nodes[nodeAddress].votingPower;
    }
    
    /**
     * @notice Get node information
     * @param nodeAddress The address of the node
     */
    function getNode(address nodeAddress) external view returns (
        NodeClass nodeClass,
        bool isSignatory,
        uint256 votingPower,
        bool isActive
    ) {
        Node storage node = nodes[nodeAddress];
        return (
            node.nodeClass,
            node.isSignatory,
            node.votingPower,
            node.isActive
        );
    }
    
    /**
     * @notice Update guardian address
     * @param newGuardian The new guardian address
     */
    function updateGuardian(address newGuardian) external onlyGuardian {
        require(newGuardian != address(0), "Invalid address");
        guardian = newGuardian;
    }
    
    /**
     * @notice Get the hardware class weight
     * @param nodeClass The node class
     */
    function getClassWeight(NodeClass nodeClass) private pure returns (uint256) {
        if (nodeClass == NodeClass.Light) return 1;
        if (nodeClass == NodeClass.Full) return 4;
        if (nodeClass == NodeClass.Archive) return 8;
        revert("Invalid node class");
    }
}
