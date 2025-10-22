"""
Web3 Contract Interface for GenomeVault

Python wrapper for VerificationContract.sol interactions using web3.py.
Supports Ethereum, Polygon, and L2 networks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import web3, but gracefully handle if not installed
try:
    from web3 import Web3
    from web3.contract import Contract
    from web3.middleware import geth_poa_middleware

    WEB3_AVAILABLE = True
except ImportError:
    logger.warning("web3.py not installed, blockchain features will be disabled")
    WEB3_AVAILABLE = False
    Web3 = None
    Contract = None


class ContractInterface:
    """
    Python wrapper for VerificationContract.sol interactions.
    Handles connection to blockchain networks and contract calls.
    """

    # Network configurations
    NETWORKS = {
        "ethereum-mainnet": {
            "rpc_url": "https://eth-mainnet.g.alchemy.com/v2/",
            "chain_id": 1,
            "name": "Ethereum Mainnet",
        },
        "ethereum-goerli": {
            "rpc_url": "https://eth-goerli.g.alchemy.com/v2/",
            "chain_id": 5,
            "name": "Ethereum Goerli Testnet",
        },
        "polygon": {
            "rpc_url": "https://polygon-rpc.com",
            "chain_id": 137,
            "name": "Polygon Mainnet",
        },
        "polygon-mumbai": {
            "rpc_url": "https://rpc-mumbai.maticvigil.com",
            "chain_id": 80001,
            "name": "Polygon Mumbai Testnet",
        },
        "localhost": {
            "rpc_url": "http://localhost:8545",
            "chain_id": 1337,
            "name": "Local Development",
        },
    }

    def __init__(
        self,
        network: str,
        contract_address: str,
        private_key: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize contract interface.

        Args:
            network: Network name (ethereum-mainnet, polygon, polygon-mumbai, etc.)
            contract_address: Deployed contract address
            private_key: Private key for signing transactions (optional for read-only)
            api_key: API key for RPC provider (optional)
        """
        if not WEB3_AVAILABLE:
            raise ImportError(
                "web3.py is not installed. Install with: pip install web3>=6.0.0"
            )

        if network not in self.NETWORKS:
            raise ValueError(
                f"Unknown network: {network}. "
                f"Available: {', '.join(self.NETWORKS.keys())}"
            )

        self.network = network
        self.network_config = self.NETWORKS[network]
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.private_key = private_key

        # Initialize Web3 connection
        rpc_url = self.network_config["rpc_url"]
        if api_key and "alchemy" in rpc_url:
            rpc_url = rpc_url + api_key

        self.w3 = Web3(Web3.HTTPProvider(rpc_url))

        # Add PoA middleware for Polygon
        if "polygon" in network:
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError(
                f"Failed to connect to {self.network_config['name']} at {rpc_url}"
            )

        # Set account if private key provided
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            self.w3.eth.default_account = self.account.address
        else:
            self.account = None
            logger.warning("No private key provided, contract will be read-only")

        # Load contract ABI
        self.contract = self._load_contract()

        logger.info(
            f"Connected to {self.network_config['name']} "
            f"(chain_id={self.network_config['chain_id']}) "
            f"at contract {self.contract_address}"
        )

    def _load_contract(self) -> Contract:
        """
        Load contract from ABI file.

        Returns:
            Web3 contract instance
        """
        # Try to load ABI from file
        abi_path = Path(__file__).parent / "contracts" / "VerificationContract.json"

        if abi_path.exists():
            with open(abi_path) as f:
                contract_data = json.load(f)
                abi = contract_data.get("abi", contract_data)
        else:
            # Fallback: minimal ABI for basic functions
            logger.warning(f"ABI file not found at {abi_path}, using minimal ABI")
            abi = [
                {
                    "inputs": [
                        {"name": "proofKey", "type": "bytes32"},
                        {"name": "proofDataHash", "type": "bytes32"},
                        {"name": "publicInputsHash", "type": "bytes32"},
                        {"name": "circuitType", "type": "string"},
                    ],
                    "name": "recordProof",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function",
                },
                {
                    "inputs": [
                        {"name": "proofKey", "type": "bytes32"},
                        {"name": "isValid", "type": "bool"},
                        {"name": "reason", "type": "string"},
                    ],
                    "name": "verifyProof",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function",
                },
                {
                    "inputs": [
                        {"name": "proofKeys", "type": "bytes32[]"},
                        {"name": "proofDataHashes", "type": "bytes32[]"},
                        {"name": "publicInputsHashes", "type": "bytes32[]"},
                        {"name": "circuitTypes", "type": "string[]"},
                    ],
                    "name": "batchRecordProofs",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function",
                },
                {
                    "inputs": [{"name": "proofKey", "type": "bytes32"}],
                    "name": "getProof",
                    "outputs": [
                        {
                            "components": [
                                {"name": "prover", "type": "address"},
                                {"name": "proofHash", "type": "bytes32"},
                                {"name": "publicInputsHash", "type": "bytes32"},
                                {"name": "timestamp", "type": "uint256"},
                                {"name": "circuitType", "type": "string"},
                                {"name": "verified", "type": "bool"},
                                {"name": "verificationTime", "type": "uint256"},
                            ],
                            "name": "",
                            "type": "tuple",
                        }
                    ],
                    "stateMutability": "view",
                    "type": "function",
                },
            ]

        return self.w3.eth.contract(address=self.contract_address, abi=abi)

    def record_proof(
        self,
        proof_id: bytes,
        circuit_type: str,
        verification_result: bool,
        metadata_hash: bytes,
        gas_limit: int = 200000,
    ) -> str:
        """
        Call recordProof() on VerificationContract.sol

        Args:
            proof_id: Unique proof identifier (32 bytes)
            circuit_type: Circuit type string
            verification_result: Whether proof is valid
            metadata_hash: Hash of proof metadata (32 bytes)
            gas_limit: Gas limit for transaction

        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Cannot submit transaction without private key")

        # Ensure proof_id and metadata_hash are 32 bytes
        proof_key = self._to_bytes32(proof_id)
        proof_data_hash = self._to_bytes32(metadata_hash)
        public_inputs_hash = self._to_bytes32(metadata_hash)  # Same for now

        # Build transaction
        try:
            transaction = self.contract.functions.recordProof(
                proof_key, proof_data_hash, public_inputs_hash, circuit_type
            ).build_transaction(
                {
                    "from": self.account.address,
                    "gas": gas_limit,
                    "gasPrice": self.w3.eth.gas_price,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                }
            )

            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, private_key=self.private_key
            )

            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            logger.info(f"Proof {proof_id[:8].hex()} recorded, tx_hash: {tx_hash.hex()}")

            # Wait for confirmation (optional)
            # receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Failed to record proof: {e}")
            raise

    def batch_record_proofs(
        self,
        proof_ids: list[bytes],
        circuit_types: list[str],
        metadata_hashes: list[bytes],
        gas_limit: int = 500000,
    ) -> str:
        """
        Batch record multiple proofs to save gas.

        Args:
            proof_ids: List of proof identifiers
            circuit_types: List of circuit types
            metadata_hashes: List of metadata hashes
            gas_limit: Gas limit for transaction

        Returns:
            Transaction hash
        """
        if not self.account:
            raise ValueError("Cannot submit transaction without private key")

        if not (len(proof_ids) == len(circuit_types) == len(metadata_hashes)):
            raise ValueError("All arrays must have the same length")

        # Convert to bytes32
        proof_keys = [self._to_bytes32(pid) for pid in proof_ids]
        proof_data_hashes = [self._to_bytes32(mh) for mh in metadata_hashes]
        public_inputs_hashes = [self._to_bytes32(mh) for mh in metadata_hashes]

        # Build transaction
        try:
            transaction = self.contract.functions.batchRecordProofs(
                proof_keys, proof_data_hashes, public_inputs_hashes, circuit_types
            ).build_transaction(
                {
                    "from": self.account.address,
                    "gas": gas_limit,
                    "gasPrice": self.w3.eth.gas_price,
                    "nonce": self.w3.eth.get_transaction_count(self.account.address),
                }
            )

            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(
                transaction, private_key=self.private_key
            )
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            logger.info(f"Batch of {len(proof_ids)} proofs recorded, tx_hash: {tx_hash.hex()}")

            return tx_hash.hex()

        except Exception as e:
            logger.error(f"Failed to batch record proofs: {e}")
            raise

    def get_proof(self, proof_id: bytes) -> dict[str, Any]:
        """
        Retrieve proof details from blockchain.

        Args:
            proof_id: Proof identifier

        Returns:
            Proof details dictionary
        """
        proof_key = self._to_bytes32(proof_id)

        try:
            result = self.contract.functions.getProof(proof_key).call()

            return {
                "prover": result[0],
                "proofHash": result[1].hex(),
                "publicInputsHash": result[2].hex(),
                "timestamp": result[3],
                "circuitType": result[4],
                "verified": result[5],
                "verificationTime": result[6],
            }

        except Exception as e:
            logger.error(f"Failed to get proof: {e}")
            return {}

    def check_proof_status(self, proof_id: bytes) -> dict[str, bool]:
        """
        Check if proof exists and is verified.

        Args:
            proof_id: Proof identifier

        Returns:
            Status dictionary with exists, verified, valid
        """
        proof_key = self._to_bytes32(proof_id)

        try:
            result = self.contract.functions.checkProofStatus(proof_key).call()

            return {"exists": result[0], "verified": result[1], "valid": result[2]}

        except Exception as e:
            logger.error(f"Failed to check proof status: {e}")
            return {"exists": False, "verified": False, "valid": False}

    def get_network_info(self) -> dict[str, Any]:
        """
        Get current network information.

        Returns:
            Network info dictionary
        """
        return {
            "network": self.network,
            "network_name": self.network_config["name"],
            "chain_id": self.network_config["chain_id"],
            "connected": self.w3.is_connected(),
            "latest_block": self.w3.eth.block_number,
            "gas_price": self.w3.eth.gas_price,
            "contract_address": self.contract_address,
            "account": self.account.address if self.account else None,
        }

    def _to_bytes32(self, data: bytes | str) -> bytes:
        """
        Convert data to 32-byte array.

        Args:
            data: Bytes or hex string

        Returns:
            32-byte array
        """
        if isinstance(data, str):
            # Remove 0x prefix if present
            if data.startswith("0x"):
                data = data[2:]
            data = bytes.fromhex(data)

        # Pad to 32 bytes
        if len(data) < 32:
            data = data + b"\x00" * (32 - len(data))
        elif len(data) > 32:
            data = data[:32]

        return data


def create_contract_interface_from_config(
    blockchain_config: dict[str, Any],
) -> Optional[ContractInterface]:
    """
    Factory function to create contract interface from configuration.

    Args:
        blockchain_config: Configuration dictionary

    Returns:
        ContractInterface or None if disabled/not configured
    """
    if not blockchain_config.get("enabled", False):
        return None

    if not WEB3_AVAILABLE:
        logger.warning("web3.py not available, blockchain disabled")
        return None

    network = blockchain_config.get("network", "polygon-mumbai")
    contract_address = blockchain_config.get("contract_address")
    private_key = blockchain_config.get("private_key")
    api_key = blockchain_config.get("api_key")

    if not contract_address:
        logger.warning("No contract address configured")
        return None

    try:
        return ContractInterface(
            network=network,
            contract_address=contract_address,
            private_key=private_key,
            api_key=api_key,
        )
    except Exception as e:
        logger.error(f"Failed to create contract interface: {e}")
        return None
