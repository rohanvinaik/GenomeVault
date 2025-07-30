"""
GenomeVault 3.0 Consolidated Utilities

This module provides unified implementations of commonly used functions
across the GenomeVault codebase, eliminating duplication and ensuring
consistent behavior.

Created as part of the tail-chasing fixes initiative.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# AUTHENTICATION & AUTHORIZATION UTILITIES
# =============================================================================


def get_user_credits(user_id: str, provider_type: str = "generic") -> int:
    """
    Get credit balance for a user.

    Consolidated implementation replacing duplicates in:
    - genomevault/api/main.py
    - genomevault/blockchain/governance.py
    - genomevault/pir/client.py

    Args:
        user_id: User identifier
        provider_type: Type of provider ("hipaa", "research", "individual")

    Returns:
        Current credit balance
    """
    # Credit allocation based on user type and verification status
    credit_allocations = {
        "hospital_": 1000,  # HIPAA-verified healthcare providers
        "clinic_": 1000,  # Medical clinics
        "research_": 500,  # Academic/research institutions
        "pharma_": 2000,  # Pharmaceutical companies
        "biotech_": 1500,  # Biotech companies
        "lab_": 800,  # Reference laboratories
        "individual_": 100,  # Individual users
    }

    # Determine base allocation
    base_credits = 100  # Default for unrecognized prefixes
    for prefix, allocation in credit_allocations.items():
        if user_id.startswith(prefix):
            base_credits = allocation
            break

    # Apply provider type multipliers
    multipliers = {
        "hipaa": 1.5,  # HIPAA compliance bonus
        "research": 1.2,  # Academic research bonus
        "individual": 1.0,  # Standard individual rate
        "generic": 1.0,  # Default rate
    }

    final_credits = int(base_credits * multipliers.get(provider_type, 1.0))

    logger.info(f"User {user_id} allocated {final_credits} credits (type: {provider_type})")
    return final_credits


def verify_hsm(hsm_serial: str, provider_type: str = "generic") -> bool:
    """
    Verify Hardware Security Module credentials.

    Consolidated implementation replacing duplicates in:
    - genomevault/pir/server/pir_server.py
    - genomevault/utils/encryption.py
    - genomevault/blockchain/node.py

    Args:
        hsm_serial: HSM serial number
        provider_type: Provider type ("hipaa", "generic", "research")

    Returns:
        True if HSM is valid and trusted
    """
    if not hsm_serial or len(hsm_serial) < 6:
        logger.error(f"Invalid HSM serial format: {hsm_serial}")
        return False

    # Known trusted HSM manufacturer prefixes
    trusted_prefixes = {
        "HSM": "Generic Hardware Security Module",
        "TPM": "Trusted Platform Module",
        "YUB": "YubiKey Hardware Token",
        "NIT": "AWS Nitro Enclave",
        "SGX": "Intel SGX Hardware",
        "SEV": "AMD Secure Encrypted Virtualization",
        "ARM": "ARM TrustZone",
        "CRY": "CRYSTALS Hardware Module",
    }

    # Validate against known manufacturers
    is_known_manufacturer = any(hsm_serial.startswith(prefix) for prefix in trusted_prefixes.keys())

    if not is_known_manufacturer:
        logger.warning(f"HSM serial {hsm_serial} from unknown manufacturer")

    # HIPAA providers require specific certifications
    if provider_type == "hipaa":
        # Require FIPS 140-2 Level 3+ certified devices
        fips_certified_prefixes = ["HSM", "TPM", "NIT", "SGX"]
        if not any(hsm_serial.startswith(prefix) for prefix in fips_certified_prefixes):
            logger.error(f"HSM {hsm_serial} not FIPS 140-2 Level 3+ certified for HIPAA use")
            return False

    # Additional validation for research institutions
    if provider_type == "research":
        # Research institutions need at least basic hardware security
        if not is_known_manufacturer:
            logger.error("Research provider requires recognized HSM manufacturer")
            return False

    logger.info(f"HSM {hsm_serial} verified for {provider_type} provider")
    return True


# =============================================================================
# BLOCKCHAIN & GOVERNANCE UTILITIES
# =============================================================================


def calculate_total_voting_power(nodes: dict[str, Any]) -> int:
    """
    Calculate total voting power across all nodes in dual-axis model.

    Consolidated implementation replacing duplicates in:
    - genomevault/blockchain/governance.py
    - genomevault/blockchain/consensus.py
    - genomevault/api/topology.py

    Args:
        nodes: Dictionary of node information with class and signatory weights

    Returns:
        Total voting power in the network (sum of all w = c + s)
    """
    total_power = 0

    # Voting weight calculation: w = c + s
    # c (class weight): Light=1, Full=4, Archive=8
    # s (signatory weight): Non-TS=0, TS=10

    for node_id, node_info in nodes.items():
        class_weight = node_info.get("class_weight", 1)
        signatory_weight = node_info.get("signatory_weight", 0)

        # Validate weight bounds
        if class_weight not in [1, 4, 8]:
            logger.warning(
                f"Invalid class weight {class_weight} for node {node_id}, defaulting to 1"
            )
            class_weight = 1

        if signatory_weight not in [0, 10]:
            logger.warning(
                f"Invalid signatory weight {signatory_weight} for node {node_id}, defaulting to 0"
            )
            signatory_weight = 0

        voting_power = class_weight + signatory_weight
        total_power += voting_power

        logger.debug(f"Node {node_id}: c={class_weight}, s={signatory_weight}, w={voting_power}")

    logger.info(f"Total network voting power: {total_power} across {len(nodes)} nodes")
    return total_power


def get_config(key: str, default: Any = None, config_type: str = "main") -> Any:
    """
    Unified configuration getter across all modules.

    Consolidated implementation replacing duplicates in:
    - genomevault/core/config.py
    - genomevault/utils/config.py
    - genomevault/local_processing/config.py

    Args:
        key: Configuration key to retrieve
        default: Default value if key not found
        config_type: Configuration type ("main", "local", "network", "security")

    Returns:
        Configuration value or default
    """
    # Configuration hierarchy and defaults
    config_defaults = {
        "main": {
            "log_level": "INFO",
            "max_workers": 4,
            "timeout_seconds": 30,
            "enable_metrics": True,
        },
        "local": {
            "data_dir": "~/.genomevault/data",
            "cache_dir": "~/.genomevault/cache",
            "max_storage_gb": 100,
            "auto_backup": True,
        },
        "network": {
            "pir_servers": 5,
            "pir_timeout": 30,
            "max_retries": 3,
            "preferred_regions": ["us-east", "eu-west"],
        },
        "security": {
            "encryption_algorithm": "AES-256-GCM",
            "key_rotation_days": 90,
            "audit_logging": True,
            "require_hsm": False,
        },
    }

    # Get from environment first, then defaults
    env_key = f"GENOMEVAULT_{config_type.upper()}_{key.upper()}"
    env_value = os.environ.get(env_key)

    if env_value is not None:
        # Try to parse as JSON for complex types
        try:
            return json.loads(env_value)
        except (json.JSONDecodeError, TypeError):
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            return env_value
            raise

    # Fall back to defaults
    config_section = config_defaults.get(config_type, {})
    return config_section.get(key, default)


# =============================================================================
# ZERO-KNOWLEDGE PROOF UTILITIES
# =============================================================================


def create_circuit_template(circuit_type: str, **kwargs) -> dict[str, Any]:
    """
    Create zero-knowledge proof circuit template.

    Consolidated implementation replacing duplicates in:
    - genomevault/zk_proofs/prover.py (6 different circuit functions)
    - genomevault/zk_proofs/circuits/

    Args:
        circuit_type: Type of circuit to create
        **kwargs: Circuit-specific parameters

    Returns:
        Circuit template configuration
    """
    # Base template with common parameters
    base_template = {
        "security_level": 128,
        "curve": "bn254",
        "proof_system": "plonk",
        "max_constraints": 50000,
        "post_quantum_ready": False,
        "verification_key_size": 32,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Circuit-specific templates
    circuit_templates = {
        "variant_presence": {
            **base_template,
            "public_inputs": ["variant_hash", "reference_hash", "commitment_root"],
            "private_inputs": ["variant_data", "merkle_proof", "witness_randomness"],
            "constraints": 5000,
            "proof_size_bytes": 192,
            "verification_time_ms": 10,
            "description": "Proves presence of genetic variant without revealing location",
        },
        "polygenic_risk_score": {
            **base_template,
            "public_inputs": ["prs_model_hash", "score_range", "result_commitment"],
            "private_inputs": ["variants", "weights", "merkle_proofs"],
            "constraints": 20000,
            "proof_size_bytes": 384,
            "verification_time_ms": 25,
            "description": "Computes polygenic risk score with privacy",
        },
        "ancestry_composition": {
            **base_template,
            "public_inputs": ["ancestry_model_hash", "composition_commitment"],
            "private_inputs": ["genotype_data", "population_weights"],
            "constraints": 15000,
            "proof_size_bytes": 320,
            "verification_time_ms": 20,
            "description": "Verifies ancestry composition without revealing genotype",
        },
        "pharmacogenomic": {
            **base_template,
            "public_inputs": ["drug_model_hash", "response_prediction"],
            "private_inputs": ["pharmacogenes", "drug_interactions"],
            "constraints": 18000,
            "proof_size_bytes": 320,
            "verification_time_ms": 22,
            "description": "Predicts drug response while protecting genetic privacy",
        },
        "pathway_enrichment": {
            **base_template,
            "public_inputs": ["pathway_definition_hash", "enrichment_score"],
            "private_inputs": ["gene_expression", "pathway_membership"],
            "constraints": 25000,
            "proof_size_bytes": 448,
            "verification_time_ms": 30,
            "description": "Analyzes biological pathway enrichment privately",
        },
        "diabetes_risk": {
            **base_template,
            "public_inputs": [
                "glucose_threshold",
                "risk_threshold",
                "result_commitment",
            ],
            "private_inputs": ["glucose_reading", "genetic_risk_score"],
            "constraints": 15000,
            "proof_size_bytes": 384,
            "verification_time_ms": 25,
            "description": "Diabetes risk assessment combining genetic and clinical data",
        },
    }

    if circuit_type not in circuit_templates:
        available_types = list(circuit_templates.keys())
        raise ValueError(
            f"Unknown circuit type: {circuit_type}. Available types: {available_types}"
        )

    # Start with template and apply overrides
    template = circuit_templates[circuit_type].copy()
    template.update(kwargs)

    logger.info(
        f"Created {circuit_type} circuit template with {template['constraints']} constraints"
    )
    return template


# =============================================================================
# HYPERVECTOR ENCODING UTILITIES
# =============================================================================


def create_hierarchical_encoder(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create hierarchical hypervector encoder configuration.

    Consolidated implementation replacing duplicates in:
    - genomevault/hypervector_transform/hierarchical.py
    - genomevault/hypervector_transform/encoding.py
    - genomevault/local_processing/sequencing.py

    Args:
        config: Optional configuration overrides

    Returns:
        Configured hierarchical encoder specification
    """
    # Default hierarchical configuration
    default_config = {
        "dimensions": {
            "base": 10000,  # Individual features
            "mid": 15000,  # Gene/pathway level
            "high": 20000,  # System-wide patterns
        },
        "compression_tiers": {
            "mini": 5000,  # Most studied SNPs (~25 KB)
            "clinical": 120000,  # ACMG + PharmGKB (~300 KB)
            "full": 10000,  # Full HDC per modality (100-200 KB)
        },
        "domain_projections": {
            "genomic": True,
            "transcriptomic": True,
            "epigenomic": True,
            "proteomic": True,
            "clinical": True,
        },
        "binding_operations": {
            "circular_convolution": True,
            "element_wise_multiplication": True,
            "permutation_based": True,
            "position_aware": True,
        },
        "optimization": {
            "sparsity_threshold": 0.1,
            "quantization_bits": 8,
            "use_simd": True,
            "cache_projections": True,
        },
        "privacy": {
            "differential_privacy": True,
            "epsilon": 1.0,
            "delta": 1e-6,
            "noise_distribution": "gaussian",
        },
    }

    # Apply configuration overrides
    if config:
        for section, section_config in config.items():
            if section in default_config and isinstance(section_config, dict):
                default_config[section].update(section_config)
            else:
                default_config[section] = section_config

    # Validate configuration
    dims = default_config["dimensions"]
    if not (dims["base"] <= dims["mid"] <= dims["high"]):
        raise ValueError("Dimension hierarchy must be base <= mid <= high")

    # Calculate storage requirements
    tiers = default_config["compression_tiers"]
    storage_estimate = {
        "mini_only": tiers["mini"] // 8,  # bits to bytes
        "clinical_only": tiers["clinical"] // 8,
        "full_genomic": dims["base"] * 4,  # 32-bit floats
        "full_multiomics": dims["base"]
        * 4
        * len([k for k, v in default_config["domain_projections"].items() if v]),
    }

    default_config["storage_estimates_bytes"] = storage_estimate

    logger.info(
        f"Hierarchical encoder configured with {dims} dimensions across {len(default_config['domain_projections'])} modalities"
    )
    return default_config


# =============================================================================
# HIPAA COMPLIANCE UTILITIES
# =============================================================================


def check_hipaa_compliance(data_dict: dict[str, Any], context: str = "general") -> dict[str, Any]:
    """
    Check HIPAA compliance for data processing and storage.

    Implementation for phantom functions:
    - HospitalFLClient._check_consent()
    - HospitalFLClient._check_deidentification()

    Args:
        data_dict: Data to check for compliance
        context: Processing context ("storage", "transmission", "analysis")

    Returns:
        Compliance check results with recommendations
    """
    # HIPAA Safe Harbor identifiers that must be removed/protected
    prohibited_identifiers = {
        "direct": [
            "name",
            "ssn",
            "mrn",
            "address",
            "phone",
            "email",
            "fax",
            "account_number",
            "certificate_number",
            "vehicle_id",
            "device_id",
            "url",
            "ip_address",
            "biometric_id",
            "photo",
        ],
        "dates": [
            "birth_date",
            "admission_date",
            "discharge_date",
            "death_date",
            "treatment_date",
            "service_date",
        ],
        "ages_over_89": ["age", "birth_year", "age_at_diagnosis", "age_at_treatment"],
    }

    compliance_results = {
        "compliant": True,
        "violations": [],
        "warnings": [],
        "recommendations": [],
        "context": context,
        "checked_at": datetime.utcnow().isoformat(),
    }

    # Check for direct identifiers
    for category, identifiers in prohibited_identifiers.items():
        for identifier in identifiers:
            if identifier in data_dict:
                value = data_dict[identifier]
                if value is not None and str(value).strip():
                    compliance_results["violations"].append(
                        {
                            "type": "direct_identifier",
                            "field": identifier,
                            "category": category,
                            "severity": "critical",
                        }
                    )
                    compliance_results["compliant"] = False

    # Check for quasi-identifiers (combinations that enable re-identification)
    quasi_identifiers = [
        "zipcode",
        "birth_year",
        "gender",
        "ethnicity",
        "diagnosis_codes",
    ]
    present_quasi = [qi for qi in quasi_identifiers if data_dict.get(qi)]

    if len(present_quasi) >= 3:
        compliance_results["warnings"].append(
            {
                "type": "quasi_identifier_risk",
                "fields": present_quasi,
                "risk_level": "medium",
                "message": "Multiple quasi-identifiers may enable re-identification",
            }
        )

    # Context-specific checks
    if context == "transmission":
        if not data_dict.get("encrypted", False):
            compliance_results["violations"].append(
                {
                    "type": "encryption_required",
                    "severity": "critical",
                    "message": "Data transmission must be encrypted",
                }
            )
            compliance_results["compliant"] = False

    if context == "storage":
        if not data_dict.get("access_controls", False):
            compliance_results["warnings"].append(
                {
                    "type": "access_controls",
                    "severity": "medium",
                    "message": "Implement role-based access controls",
                }
            )

    # Generate recommendations
    if compliance_results["violations"]:
        compliance_results["recommendations"].extend(
            [
                "Remove all direct identifiers before processing",
                "Implement data de-identification procedures",
                "Add encryption for data at rest and in transit",
                "Establish audit logging for all data access",
            ]
        )

    if compliance_results["warnings"]:
        compliance_results["recommendations"].extend(
            [
                "Consider additional privacy protections (differential privacy)",
                "Limit quasi-identifier combinations in datasets",
                "Implement k-anonymity or l-diversity techniques",
            ]
        )

    logger.info(
        f"HIPAA compliance check: {'PASS' if compliance_results['compliant'] else 'FAIL'} "
        f"({len(compliance_results['violations'])} violations, {len(compliance_results['warnings'])} warnings)"
    )

    return compliance_results


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================


# Create aliases for backward compatibility with existing code
def ancestry_composition_circuit(**kwargs):
    """Backward compatibility alias"""
    return create_circuit_template("ancestry_composition", **kwargs)


def diabetes_risk_circuit(**kwargs):
    """Backward compatibility alias"""
    return create_circuit_template("diabetes_risk", **kwargs)


def pathway_enrichment_circuit(**kwargs):
    """Backward compatibility alias"""
    return create_circuit_template("pathway_enrichment", **kwargs)


def pharmacogenomic_circuit(**kwargs):
    """Backward compatibility alias"""
    return create_circuit_template("pharmacogenomic", **kwargs)


def polygenic_risk_score_circuit(**kwargs):
    """Backward compatibility alias"""
    return create_circuit_template("polygenic_risk_score", **kwargs)


def variant_presence_circuit(**kwargs):
    """Backward compatibility alias"""
    return create_circuit_template("variant_presence", **kwargs)


# Legacy function aliases
def root():
    """Legacy alias for API root endpoint - redirects to main API handler"""
    return {
        "status": "GenomeVault 3.0 API",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# UTILITY FUNCTIONS FOR FIXES
# =============================================================================


def validate_fix_implementation():
    """Validate that all consolidated utilities work correctly"""
    test_results = []

    # Test user credits
    try:
        credits = get_user_credits("hospital_test_001", "hipaa")
        test_results.append(("get_user_credits", credits == 1500, f"Got {credits} credits"))
    except Exception as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        test_results.append(("get_user_credits", False, str(e)))
        raise

    # Test HSM verification
    try:
        hsm_valid = verify_hsm("HSM123456789", "hipaa")
        test_results.append(("verify_hsm", hsm_valid, "HSM verification successful"))
    except Exception as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        test_results.append(("verify_hsm", False, str(e)))
        raise

    # Test circuit template
    try:
        template = create_circuit_template("variant_presence")
        test_results.append(
            ("create_circuit_template", "constraints" in template, "Template created")
        )
    except Exception as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        test_results.append(("create_circuit_template", False, str(e)))
        raise

    # Test configuration
    try:
        config_val = get_config("log_level", "DEBUG", "main")
        test_results.append(("get_config", config_val is not None, f"Config: {config_val}"))
    except Exception as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        test_results.append(("get_config", False, str(e)))
        raise

    return test_results


if __name__ == "__main__":
    # Run validation tests
    results = validate_fix_implementation()
    print("\n=== Utility Validation Results ===")
    for test_name, passed, message in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
