# GenomeVault Blockchain Layer

This directory contains the Ethereum smart contracts for GenomeVault's zero-knowledge proof verification system.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Compile contracts:
```bash
npm run compile
```

3. Start local blockchain:
```bash
npm run node
```

4. Deploy contracts (in a new terminal):
```bash
npm run deploy
```

5. Run tests:
```bash
npm test
```

## Contract Overview

### Consolidated Structure

All smart contracts are now consolidated in `genomevault/blockchain/contracts/`:
- **Source location**: `genomevault/blockchain/contracts/`
- **Build location**: `blockchain/contracts/` (via symlinks)

### VerificationContract

The main contract that handles:
- Recording zero-knowledge proofs on-chain
- Verifying submitted proofs
- Managing allowed circuit types
- Tracking statistics for provers and circuit usage

### GovernanceDAO

Governance contract that implements:
- On-chain governance with quadratic voting
- Multi-stakeholder committees
- Proposal management
- Emergency actions

### Supported Circuit Types

- `variant_presence`: Proves presence of genetic variants
- `polygenic_risk_score`: Polygenic risk score calculations
- `ancestry_composition`: Ancestry composition proofs
- `pharmacogenomic`: Drug response predictions
- `pathway_enrichment`: Pathway enrichment analysis
- `diabetes_risk_alert`: Diabetes risk assessment

## Events

The contract emits the following events:
- `ProofRecorded`: When a new proof is submitted
- `ProofVerified`: When a proof is verified
- `CircuitAdded`: When a new circuit type is allowed
- `CircuitRemoved`: When a circuit type is removed

## Deployment

The deployment script will:
1. Deploy the VerificationContract
2. Display the contract address and owner
3. Show all initially allowed circuit types
4. Save deployment information

## Testing

The test suite covers:
- Contract deployment
- Proof recording with events
- Proof verification
- Statistics tracking
- Circuit management
- Access control

Run tests with coverage:
```bash
npx hardhat coverage
```

## Integration

To integrate with the Python GenomeVault system, use the contract ABI from `artifacts/contracts/VerificationContract.sol/VerificationContract.json` with web3.py or ethers.py.
