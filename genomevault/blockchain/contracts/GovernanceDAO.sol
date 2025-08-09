// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title GenomeVault Governance DAO Contract
 * @notice Implements on-chain governance with quadratic voting and multi-stakeholder committees
 * @dev Follows the dual-axis voting model from the system specification
 */

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/math/Math.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract GovernanceDAO is AccessControl, ReentrancyGuard {
    using SafeMath for uint256;

    // Roles
    bytes32 public constant COMMITTEE_ROLE = keccak256("COMMITTEE_ROLE");
    bytes32 public constant PROPOSER_ROLE = keccak256("PROPOSER_ROLE");
    bytes32 public constant EXECUTOR_ROLE = keccak256("EXECUTOR_ROLE");

    // Proposal types
    enum ProposalType {
        ProtocolUpdate,
        ParameterChange,
        ReferenceUpdate,
        AlgorithmCertification,
        TreasuryAllocation,
        EmergencyAction,
        CommitteeElection
    }

    // Proposal status
    enum ProposalStatus {
        Draft,
        Active,
        Passed,
        Rejected,
        Executed,
        Cancelled
    }

    // Committee types
    enum CommitteeType {
        ScientificAdvisory,
        Ethics,
        Security,
        UserRepresentatives
    }

    // Vote choices
    enum VoteChoice {
        Yes,
        No,
        Abstain
    }

    struct Proposal {
        uint256 id;
        ProposalType proposalType;
        string title;
        string description;
        address proposer;
        uint256 startTime;
        uint256 endTime;
        uint256 executionDelay;
        ProposalStatus status;
        uint256 yesVotes;
        uint256 noVotes;
        uint256 abstainVotes;
        uint256 quorumRequired;
        uint256 approvalThreshold;
        bytes executionData;
        uint256 executionTime;
        mapping(address => VoteRecord) votes;
    }

    struct VoteRecord {
        bool hasVoted;
        VoteChoice choice;
        uint256 weight;
        address delegateFrom;
        uint256 timestamp;
    }

    struct Committee {
        CommitteeType committeeType;
        mapping(address => bool) members;
        address chair;
        uint256 termEnd;
        uint256 votingMultiplier; // Scaled by 100 (150 = 1.5x)
    }

    struct NodeInfo {
        uint8 nodeClass; // 1=Light, 4=Full, 8=Archive
        bool isTrustedSignatory;
        uint256 votingPower; // w = c + s
        uint256 stakeAmount;
        uint256 credits;
    }

    // State variables
    uint256 public proposalCounter;
    uint256 public totalVotingPower;
    uint256 public proposalThreshold = 100; // Min voting power to propose
    uint256 public votingPeriod = 7 days;
    uint256 public executionDelay = 2 days;

    mapping(uint256 => Proposal) public proposals;
    mapping(address => NodeInfo) public nodes;
    mapping(CommitteeType => Committee) public committees;
    mapping(address => address) public delegations;

    // Events
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        ProposalType proposalType,
        string title
    );

    event VoteCast(
        uint256 indexed proposalId,
        address indexed voter,
        VoteChoice choice,
        uint256 weight,
        address delegateFrom
    );

    event ProposalExecuted(uint256 indexed proposalId);
    event ProposalCancelled(uint256 indexed proposalId);

    event NodeRegistered(
        address indexed nodeAddress,
        uint8 nodeClass,
        bool isTrustedSignatory,
        uint256 votingPower
    );

    event CommitteeMemberAdded(
        CommitteeType indexed committeeType,
        address indexed member
    );

    event DelegationSet(address indexed delegator, address indexed delegate);
    event DelegationRevoked(address indexed delegator);

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(EXECUTOR_ROLE, msg.sender);

        // Initialize committees
        _initializeCommittees();
    }

    function _initializeCommittees() private {
        committees[CommitteeType.ScientificAdvisory].votingMultiplier = 150; // 1.5x
        committees[CommitteeType.Ethics].votingMultiplier = 130; // 1.3x
        committees[CommitteeType.Security].votingMultiplier = 200; // 2.0x
        committees[CommitteeType.UserRepresentatives].votingMultiplier = 100; // 1.0x
    }

    /**
     * @notice Register a node in the governance system
     * @param nodeAddress Address of the node
     * @param nodeClass Hardware class (1=Light, 4=Full, 8=Archive)
     * @param isTrustedSignatory Whether node is HIPAA-verified
     */
    function registerNode(
        address nodeAddress,
        uint8 nodeClass,
        bool isTrustedSignatory
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(nodeClass == 1 || nodeClass == 4 || nodeClass == 8, "Invalid node class");

        // Calculate voting power: w = c + s
        uint256 signatoryWeight = isTrustedSignatory ? 10 : 0;
        uint256 votingPower = uint256(nodeClass) + signatoryWeight;

        NodeInfo storage node = nodes[nodeAddress];
        node.nodeClass = nodeClass;
        node.isTrustedSignatory = isTrustedSignatory;
        node.votingPower = votingPower;

        totalVotingPower = totalVotingPower.add(votingPower);

        emit NodeRegistered(nodeAddress, nodeClass, isTrustedSignatory, votingPower);
    }

    /**
     * @notice Create a new governance proposal
     * @param proposalType Type of the proposal
     * @param title Proposal title
     * @param description Detailed description
     * @param executionData Encoded execution data
     */
    function createProposal(
        ProposalType proposalType,
        string memory title,
        string memory description,
        bytes memory executionData
    ) external returns (uint256) {
        require(
            nodes[msg.sender].votingPower >= proposalThreshold,
            "Insufficient voting power"
        );

        uint256 proposalId = proposalCounter++;
        Proposal storage proposal = proposals[proposalId];

        proposal.id = proposalId;
        proposal.proposalType = proposalType;
        proposal.title = title;
        proposal.description = description;
        proposal.proposer = msg.sender;
        proposal.startTime = block.timestamp + 1 days;
        proposal.endTime = block.timestamp + votingPeriod + 1 days;
        proposal.executionDelay = executionDelay;
        proposal.status = ProposalStatus.Draft;
        proposal.executionData = executionData;

        // Set requirements based on proposal type
        if (proposalType == ProposalType.ProtocolUpdate ||
            proposalType == ProposalType.EmergencyAction) {
            proposal.quorumRequired = totalVotingPower.mul(20).div(100); // 20%
            proposal.approvalThreshold = 67; // 67%
        } else {
            proposal.quorumRequired = totalVotingPower.mul(10).div(100); // 10%
            proposal.approvalThreshold = 51; // 51%
        }

        emit ProposalCreated(proposalId, msg.sender, proposalType, title);

        return proposalId;
    }

    /**
     * @notice Cast a vote on a proposal
     * @param proposalId ID of the proposal
     * @param choice Vote choice
     * @param votingPowerOverride Override voting power for quadratic voting
     */
    function vote(
        uint256 proposalId,
        VoteChoice choice,
        uint256 votingPowerOverride
    ) external nonReentrant {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.id == proposalId, "Proposal does not exist");
        require(
            block.timestamp >= proposal.startTime &&
            block.timestamp <= proposal.endTime,
            "Voting period not active"
        );

        // Update status if needed
        if (proposal.status == ProposalStatus.Draft) {
            proposal.status = ProposalStatus.Active;
        }
        require(proposal.status == ProposalStatus.Active, "Proposal not active");

        // Check delegation
        address voter = _getFinalDelegate(msg.sender);
        VoteRecord storage voteRecord = proposal.votes[voter];
        require(!voteRecord.hasVoted, "Already voted");

        // Get voting power
        uint256 votingPower = votingPowerOverride > 0 ?
            votingPowerOverride : nodes[voter].votingPower;
        require(votingPower > 0, "No voting power");

        // Apply quadratic voting
        uint256 voteWeight = _sqrt(votingPower);

        // Apply committee multiplier
        uint256 multiplier = _getCommitteeMultiplier(voter, proposal.proposalType);
        voteWeight = voteWeight.mul(multiplier).div(100);

        // Record vote
        voteRecord.hasVoted = true;
        voteRecord.choice = choice;
        voteRecord.weight = voteWeight;
        voteRecord.delegateFrom = voter != msg.sender ? msg.sender : address(0);
        voteRecord.timestamp = block.timestamp;

        // Update vote counts
        if (choice == VoteChoice.Yes) {
            proposal.yesVotes = proposal.yesVotes.add(voteWeight);
        } else if (choice == VoteChoice.No) {
            proposal.noVotes = proposal.noVotes.add(voteWeight);
        } else {
            proposal.abstainVotes = proposal.abstainVotes.add(voteWeight);
        }

        emit VoteCast(proposalId, voter, choice, voteWeight, voteRecord.delegateFrom);
    }

    /**
     * @notice Finalize a proposal after voting period
     * @param proposalId ID of the proposal
     */
    function finalizeProposal(uint256 proposalId) external {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.id == proposalId, "Proposal does not exist");
        require(block.timestamp > proposal.endTime, "Voting period not ended");
        require(
            proposal.status == ProposalStatus.Active,
            "Proposal not active"
        );

        uint256 totalVotes = proposal.yesVotes.add(proposal.noVotes).add(proposal.abstainVotes);

        // Check quorum
        if (totalVotes < proposal.quorumRequired) {
            proposal.status = ProposalStatus.Rejected;
            return;
        }

        // Check approval threshold
        uint256 approvalRate = proposal.yesVotes.mul(100).div(
            proposal.yesVotes.add(proposal.noVotes)
        );

        if (approvalRate >= proposal.approvalThreshold) {
            proposal.status = ProposalStatus.Passed;
            proposal.executionTime = block.timestamp + proposal.executionDelay;
        } else {
            proposal.status = ProposalStatus.Rejected;
        }
    }

    /**
     * @notice Execute a passed proposal
     * @param proposalId ID of the proposal
     */
    function executeProposal(uint256 proposalId) external onlyRole(EXECUTOR_ROLE) {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.id == proposalId, "Proposal does not exist");
        require(proposal.status == ProposalStatus.Passed, "Proposal not passed");
        require(block.timestamp >= proposal.executionTime, "Execution delay not met");

        proposal.status = ProposalStatus.Executed;

        // Execute based on proposal type
        if (proposal.proposalType == ProposalType.ParameterChange) {
            _executeParameterChange(proposal.executionData);
        } else if (proposal.proposalType == ProposalType.CommitteeElection) {
            _executeCommitteeElection(proposal.executionData);
        } else if (proposal.proposalType == ProposalType.TreasuryAllocation) {
            _executeTreasuryAllocation(proposal.executionData);
        }
        // Other types require manual execution

        emit ProposalExecuted(proposalId);
    }

    /**
     * @notice Add member to committee
     * @param committeeType Type of committee
     * @param member Address to add
     */
    function addCommitteeMember(
        CommitteeType committeeType,
        address member
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        committees[committeeType].members[member] = true;
        emit CommitteeMemberAdded(committeeType, member);
    }

    /**
     * @notice Set vote delegation
     * @param delegate Address to delegate to
     */
    function delegate(address delegate) external {
        require(delegate != msg.sender, "Cannot delegate to self");
        require(!_wouldCreateCycle(msg.sender, delegate), "Would create cycle");

        delegations[msg.sender] = delegate;
        emit DelegationSet(msg.sender, delegate);
    }

    /**
     * @notice Revoke vote delegation
     */
    function revokeDelegate() external {
        delete delegations[msg.sender];
        emit DelegationRevoked(msg.sender);
    }

    // Internal functions

    function _getFinalDelegate(address voter) private view returns (address) {
        address current = voter;
        uint256 iterations = 0;

        while (delegations[current] != address(0) && iterations < 10) {
            current = delegations[current];
            iterations++;
        }

        return current;
    }

    function _wouldCreateCycle(
        address delegator,
        address delegate
    ) private view returns (bool) {
        address current = delegate;
        uint256 iterations = 0;

        while (delegations[current] != address(0) && iterations < 10) {
            if (current == delegator) {
                return true;
            }
            current = delegations[current];
            iterations++;
        }

        return false;
    }

    function _getCommitteeMultiplier(
        address voter,
        ProposalType proposalType
    ) private view returns (uint256) {
        uint256 maxMultiplier = 100;

        // Check each committee
        for (uint i = 0; i < 4; i++) {
            CommitteeType cType = CommitteeType(i);
            if (committees[cType].members[voter]) {
                if (_isCommitteeRelevant(cType, proposalType)) {
                    uint256 multiplier = committees[cType].votingMultiplier;
                    if (multiplier > maxMultiplier) {
                        maxMultiplier = multiplier;
                    }
                }
            }
        }

        return maxMultiplier;
    }

    function _isCommitteeRelevant(
        CommitteeType committeeType,
        ProposalType proposalType
    ) private pure returns (bool) {
        if (committeeType == CommitteeType.ScientificAdvisory) {
            return proposalType == ProposalType.AlgorithmCertification ||
                   proposalType == ProposalType.ReferenceUpdate ||
                   proposalType == ProposalType.ProtocolUpdate;
        } else if (committeeType == CommitteeType.Ethics) {
            return proposalType == ProposalType.ProtocolUpdate ||
                   proposalType == ProposalType.ParameterChange;
        } else if (committeeType == CommitteeType.Security) {
            return proposalType == ProposalType.EmergencyAction ||
                   proposalType == ProposalType.ProtocolUpdate;
        } else {
            return proposalType == ProposalType.ParameterChange ||
                   proposalType == ProposalType.TreasuryAllocation;
        }
    }

    function _sqrt(uint256 x) private pure returns (uint256) {
        if (x == 0) return 0;
        uint256 z = (x + 1) / 2;
        uint256 y = x;
        while (z < y) {
            y = z;
            z = (x / z + z) / 2;
        }
        return y;
    }

    function _executeParameterChange(bytes memory data) private {
        // Decode and execute parameter change
        // Implementation depends on specific parameters
    }

    function _executeCommitteeElection(bytes memory data) private {
        // Decode and execute committee changes
        // Implementation depends on election format
    }

    function _executeTreasuryAllocation(bytes memory data) private {
        // Decode and execute treasury transfer
        (address recipient, uint256 amount) = abi.decode(data, (address, uint256));
        // Transfer logic here
    }

    // View functions

    function getProposal(uint256 proposalId) external view returns (
        ProposalType proposalType,
        string memory title,
        ProposalStatus status,
        uint256 yesVotes,
        uint256 noVotes,
        uint256 abstainVotes,
        uint256 startTime,
        uint256 endTime
    ) {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.proposalType,
            proposal.title,
            proposal.status,
            proposal.yesVotes,
            proposal.noVotes,
            proposal.abstainVotes,
            proposal.startTime,
            proposal.endTime
        );
    }

    function getNodeVotingPower(address nodeAddress) external view returns (uint256) {
        return nodes[nodeAddress].votingPower;
    }

    function isCommitteeMember(
        address member,
        CommitteeType committeeType
    ) external view returns (bool) {
        return committees[committeeType].members[member];
    }
}
