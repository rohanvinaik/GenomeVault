# Blockchain Contracts

## Overview

This directory contains the consolidated Solidity smart contracts for GenomeVault's blockchain layer. All contracts have been unified here to provide a single source of truth for on-chain functionality.

## Contract Structure

### Core Contracts

#### VerificationContract.sol
- **Purpose**: Records and verifies zero-knowledge proofs on-chain
- **Version**: Solidity ^0.8.17
- **Key Features**:
  - Proof recording and verification
  - Circuit type registry
  - Verification result storage
  - Event emission for proof lifecycle

#### GovernanceDAO.sol
- **Purpose**: Implements on-chain governance with quadratic voting
- **Version**: Solidity ^0.8.19
- **Key Features**:
  - Multi-stakeholder committees
  - Proposal management (protocol updates, parameter changes, etc.)
  - Quadratic voting mechanism
  - Emergency action handling
  - Committee elections

### Python Integration

#### training_attestation.py
- Python interface for interacting with blockchain contracts
- Provides attestation creation and verification
- Handles contract deployment and interaction

## Deployment

Contracts are deployed using Hardhat from the `blockchain/` directory:

```bash
cd blockchain/
npm install
npx hardhat compile
npx hardhat run scripts/deploy.ts --network localhost
```

## Contract Addresses

Contracts use symlinks for build compatibility:
- `blockchain/contracts/` contains symlinks to the actual contracts
- Source contracts are maintained in `genomevault/blockchain/contracts/`

This structure ensures:
1. Single source of truth for contracts
2. Compatibility with Hardhat build system
3. Clean separation between Solidity and Python code

## Integration

To integrate with Python code:

```python
from genomevault.blockchain.contracts.training_attestation import (
    create_attestation,
    verify_attestation
)
```

## Testing

Contract tests are located in `blockchain/test/`:
- Unit tests: `test/VerificationContract.test.ts`
- Smoke tests: `test/smoke/ProofSubmission.smoke.test.ts`

Run tests:
```bash
cd blockchain/
npx hardhat test
```

## Security Considerations

1. All contracts use recent Solidity versions (0.8.17+) with built-in overflow protection
2. GovernanceDAO uses OpenZeppelin's battle-tested contracts
3. Access control implemented for sensitive operations
4. Reentrancy protection on state-changing functions

## Gas Optimization

Contracts are compiled with optimization enabled:
- Optimizer runs: 200
- Focus on deployment cost vs runtime cost balance

## Future Enhancements

- [ ] Add multi-signature wallet for treasury management
- [ ] Implement upgradeable proxy pattern for contracts
- [ ] Add more comprehensive event logging
- [ ] Implement cross-chain proof verification
- [ ] Add support for additional ZK proof systems

## Contract Verification

For mainnet/testnet deployments, verify contracts on Etherscan:

```bash
npx hardhat verify --network [network] [contract_address] [constructor_args]
```

## License

- VerificationContract: MIT
- GovernanceDAO: MIT with OpenZeppelin dependencies
