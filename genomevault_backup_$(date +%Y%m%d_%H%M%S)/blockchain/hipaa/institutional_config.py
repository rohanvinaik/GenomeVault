"""
Institutional Node Configuration for GenomeVault Phase 2

Defines hardware requirements and resource classes for institutional nodes.
Supports different deployment tiers (LIGHT, FULL, ARCHIVE) based on
institutional capacity.

Features:
- Hardware specification validation
- Resource class assignment
- Institutional deployment templates
- Cost estimation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class NodeResourceClass(Enum):
    """Resource class for institutional nodes"""

    LIGHT = auto()  # Minimal resources, query only
    FULL = auto()  # Full node, 1U server
    ARCHIVE = auto()  # Archive node, 4U+ server, high storage


class DeploymentMode(Enum):
    """Deployment mode for institutional infrastructure"""

    CLOUD = auto()  # Cloud deployment (AWS, Azure, GCP)
    ON_PREM = auto()  # On-premises deployment
    HYBRID = auto()  # Hybrid cloud + on-prem


@dataclass
class HardwareRequirements:
    """Hardware requirements for node resource class"""

    # Compute
    cpu_cores: int
    ram_gb: int

    # Storage
    storage_tb: int
    storage_type: str  # "SSD", "NVMe", "HDD"

    # Network
    bandwidth_mbps: int
    uptime_requirement: float  # 0.0 - 1.0 (e.g., 0.99 = 99% uptime)

    # Security
    requires_hsm: bool = False
    requires_tpm: bool = False
    requires_secure_enclave: bool = False

    # Optional GPU
    gpu_count: int = 0
    gpu_memory_gb: int = 0

    def meets_requirements(self, actual: HardwareRequirements) -> bool:
        """Check if actual hardware meets requirements"""
        return (
            actual.cpu_cores >= self.cpu_cores
            and actual.ram_gb >= self.ram_gb
            and actual.storage_tb >= self.storage_tb
            and actual.bandwidth_mbps >= self.bandwidth_mbps
            and actual.uptime_requirement >= self.uptime_requirement
            and (not self.requires_hsm or actual.requires_hsm)
            and (not self.requires_tpm or actual.requires_tpm)
        )


@dataclass
class InstitutionalNodeConfig:
    """Configuration for institutional blockchain node"""

    # Institution info
    npi: str
    institution_name: str
    resource_class: NodeResourceClass
    deployment_mode: DeploymentMode

    # Hardware
    hardware: HardwareRequirements

    # Network
    public_ip: Optional[str] = None
    domain_name: Optional[str] = None
    port: int = 8545  # Default Ethereum RPC port

    # Security
    hsm_serial: Optional[str] = None
    firewall_rules: list[str] = None
    ssl_cert_hash: Optional[str] = None

    # Data capabilities
    stores_reference_genomes: bool = True
    stores_variant_database: bool = True
    provides_pir_service: bool = False
    provides_zk_verification: bool = False

    # Performance
    max_concurrent_queries: int = 100
    max_data_contribution_gb_per_day: int = 100

    # Compliance
    hipaa_compliant: bool = True
    gdpr_compliant: bool = True
    data_residency_region: Optional[str] = None  # "US", "EU", "APAC", etc.

    def __post_init__(self):
        if self.firewall_rules is None:
            self.firewall_rules = []

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate node configuration.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate NPI
        if not self.npi or len(self.npi) != 10 or not self.npi.isdigit():
            errors.append("Invalid NPI format")

        # Validate hardware meets resource class requirements
        required_hw = get_hardware_requirements(self.resource_class)
        if not required_hw.meets_requirements(self.hardware):
            errors.append(f"Hardware does not meet {self.resource_class.name} requirements")

        # Validate security requirements
        if self.resource_class in [NodeResourceClass.FULL, NodeResourceClass.ARCHIVE]:
            if not self.hardware.requires_hsm:
                errors.append(f"{self.resource_class.name} nodes require HSM")

            if not self.hsm_serial:
                errors.append(f"{self.resource_class.name} nodes require HSM serial number")

        # Validate network
        if self.provides_pir_service and not self.public_ip:
            errors.append("PIR service nodes require public IP")

        # Validate compliance
        if self.hipaa_compliant and not self.hardware.requires_hsm:
            errors.append("HIPAA-compliant nodes require HSM")

        # Validate data residency
        if self.gdpr_compliant and not self.data_residency_region:
            errors.append("GDPR-compliant nodes require data residency region")

        is_valid = len(errors) == 0
        return is_valid, errors

    def estimate_monthly_cost(self) -> dict[str, float]:
        """
        Estimate monthly operational cost.

        Returns:
            Dictionary with cost breakdown
        """
        costs = {
            "compute": 0.0,
            "storage": 0.0,
            "network": 0.0,
            "security": 0.0,
            "total": 0.0,
        }

        # Compute cost ($/month, based on cloud pricing)
        if self.deployment_mode == DeploymentMode.CLOUD:
            # AWS c5.2xlarge = 8 vCPU, 16 GB RAM ≈ $250/mo
            costs["compute"] = (self.hardware.cpu_cores / 8.0) * (self.hardware.ram_gb / 16.0) * 250.0

            # Storage cost ($/GB/month)
            if self.hardware.storage_type == "NVMe":
                storage_cost_per_gb = 0.30  # Premium NVMe
            elif self.hardware.storage_type == "SSD":
                storage_cost_per_gb = 0.10  # Standard SSD
            else:
                storage_cost_per_gb = 0.023  # HDD
            costs["storage"] = self.hardware.storage_tb * 1000 * storage_cost_per_gb

            # Network cost ($/GB)
            # Assume 10TB/month outbound = $900
            costs["network"] = 900.0 if self.provides_pir_service else 300.0

            # Security (HSM rental, firewall, etc.)
            if self.hardware.requires_hsm:
                costs["security"] = 500.0  # CloudHSM ≈ $1.60/hour ≈ $1200/mo, shared
            else:
                costs["security"] = 50.0  # Basic firewall

        elif self.deployment_mode == DeploymentMode.ON_PREM:
            # On-prem: amortize hardware over 3 years
            # 1U server: ~$5,000 / 36 months = $139/mo
            # 4U server: ~$20,000 / 36 months = $556/mo
            if self.resource_class == NodeResourceClass.ARCHIVE:
                costs["compute"] = 556.0
            elif self.resource_class == NodeResourceClass.FULL:
                costs["compute"] = 139.0
            else:
                costs["compute"] = 50.0  # VM on existing hardware

            # Storage (amortized)
            costs["storage"] = self.hardware.storage_tb * 10.0  # $10/TB/month amortized

            # Network (bandwidth cost)
            costs["network"] = 200.0  # Corporate internet

            # Security (HSM, physical security)
            costs["security"] = 200.0 if self.hardware.requires_hsm else 50.0

        else:  # Hybrid
            # Average of cloud and on-prem
            cloud_costs = self.estimate_monthly_cost_cloud()
            on_prem_costs = self.estimate_monthly_cost_on_prem()
            for key in costs.keys():
                if key != "total":
                    costs[key] = (cloud_costs[key] + on_prem_costs[key]) / 2.0

        # Total
        costs["total"] = sum(v for k, v in costs.items() if k != "total")

        return costs

    def estimate_monthly_cost_cloud(self) -> dict[str, float]:
        """Helper for cloud cost estimation"""
        old_mode = self.deployment_mode
        self.deployment_mode = DeploymentMode.CLOUD
        costs = self.estimate_monthly_cost()
        self.deployment_mode = old_mode
        return costs

    def estimate_monthly_cost_on_prem(self) -> dict[str, float]:
        """Helper for on-prem cost estimation"""
        old_mode = self.deployment_mode
        self.deployment_mode = DeploymentMode.ON_PREM
        costs = self.estimate_monthly_cost()
        self.deployment_mode = old_mode
        return costs


def get_hardware_requirements(resource_class: NodeResourceClass) -> HardwareRequirements:
    """
    Get minimum hardware requirements for resource class.

    Args:
        resource_class: Node resource class

    Returns:
        Minimum hardware requirements
    """
    requirements = {
        NodeResourceClass.LIGHT: HardwareRequirements(
            cpu_cores=4,
            ram_gb=8,
            storage_tb=1,
            storage_type="SSD",
            bandwidth_mbps=100,
            uptime_requirement=0.95,  # 95% uptime
            requires_hsm=False,
            requires_tpm=False,
        ),
        NodeResourceClass.FULL: HardwareRequirements(
            cpu_cores=16,
            ram_gb=64,
            storage_tb=10,
            storage_type="NVMe",
            bandwidth_mbps=1000,
            uptime_requirement=0.99,  # 99% uptime
            requires_hsm=True,
            requires_tpm=True,
            requires_secure_enclave=True,
        ),
        NodeResourceClass.ARCHIVE: HardwareRequirements(
            cpu_cores=32,
            ram_gb=256,
            storage_tb=100,
            storage_type="NVMe",
            bandwidth_mbps=10000,
            uptime_requirement=0.999,  # 99.9% uptime
            requires_hsm=True,
            requires_tpm=True,
            requires_secure_enclave=True,
            gpu_count=2,  # Optional for ZK acceleration
            gpu_memory_gb=24,
        ),
    }

    return requirements[resource_class]


def create_deployment_template(
    resource_class: NodeResourceClass,
    deployment_mode: DeploymentMode,
) -> dict[str, Any]:
    """
    Create deployment template for institutional node.

    Args:
        resource_class: Node resource class
        deployment_mode: Deployment mode

    Returns:
        Deployment template configuration
    """
    hw_reqs = get_hardware_requirements(resource_class)

    template = {
        "resource_class": resource_class.name,
        "deployment_mode": deployment_mode.name,
        "hardware_requirements": {
            "cpu_cores": hw_reqs.cpu_cores,
            "ram_gb": hw_reqs.ram_gb,
            "storage_tb": hw_reqs.storage_tb,
            "storage_type": hw_reqs.storage_type,
            "bandwidth_mbps": hw_reqs.bandwidth_mbps,
            "uptime_requirement": hw_reqs.uptime_requirement,
        },
        "security_requirements": {
            "hsm": hw_reqs.requires_hsm,
            "tpm": hw_reqs.requires_tpm,
            "secure_enclave": hw_reqs.requires_secure_enclave,
        },
    }

    # Cloud-specific configuration
    if deployment_mode == DeploymentMode.CLOUD:
        if resource_class == NodeResourceClass.LIGHT:
            template["cloud"] = {
                "aws_instance_type": "t3.xlarge",
                "azure_vm_size": "Standard_D4s_v3",
                "gcp_machine_type": "n1-standard-4",
            }
        elif resource_class == NodeResourceClass.FULL:
            template["cloud"] = {
                "aws_instance_type": "c5.4xlarge",
                "azure_vm_size": "Standard_F16s_v2",
                "gcp_machine_type": "n1-highcpu-16",
            }
        else:  # ARCHIVE
            template["cloud"] = {
                "aws_instance_type": "r5.8xlarge",
                "azure_vm_size": "Standard_E32s_v3",
                "gcp_machine_type": "n1-highmem-32",
            }

    # On-prem hardware recommendations
    elif deployment_mode == DeploymentMode.ON_PREM:
        if resource_class == NodeResourceClass.LIGHT:
            template["hardware"] = {
                "server_type": "1U rack server",
                "cpu": "Intel Xeon Silver 4208 (8 cores)",
                "ram": "32GB DDR4 ECC",
                "storage": "2x 1TB NVMe SSD (RAID 1)",
                "network": "2x 1GbE",
            }
        elif resource_class == NodeResourceClass.FULL:
            template["hardware"] = {
                "server_type": "1U rack server",
                "cpu": "2x Intel Xeon Gold 6230 (40 cores total)",
                "ram": "128GB DDR4 ECC",
                "storage": "4x 4TB NVMe SSD (RAID 10)",
                "network": "2x 10GbE",
                "hsm": "Thales Luna SA 7 or AWS CloudHSM",
            }
        else:  # ARCHIVE
            template["hardware"] = {
                "server_type": "4U rack server",
                "cpu": "2x Intel Xeon Platinum 8280 (56 cores total)",
                "ram": "512GB DDR4 ECC",
                "storage": "12x 16TB NVMe SSD (RAID 6) + 24x 20TB HDD (RAID 6)",
                "network": "2x 25GbE",
                "hsm": "Thales Luna SA 7 + backup HSM",
                "gpu": "2x NVIDIA A100 (optional, for ZK acceleration)",
            }

    return template


def validate_institutional_deployment(config: InstitutionalNodeConfig) -> dict[str, Any]:
    """
    Validate institutional node deployment.

    Args:
        config: Institutional node configuration

    Returns:
        Validation result with recommendations
    """
    is_valid, errors = config.validate()

    result = {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": [],
        "recommendations": [],
        "estimated_cost": config.estimate_monthly_cost(),
    }

    # Add warnings
    if config.resource_class == NodeResourceClass.LIGHT and config.provides_pir_service:
        result["warnings"].append("LIGHT nodes not recommended for PIR service (high bandwidth)")

    if config.deployment_mode == DeploymentMode.CLOUD and not config.data_residency_region:
        result["warnings"].append("Cloud deployment should specify data residency region")

    # Add recommendations
    if config.resource_class == NodeResourceClass.FULL:
        result["recommendations"].append("Consider ARCHIVE class for long-term data storage")

    if config.deployment_mode == DeploymentMode.ON_PREM and config.hardware.storage_tb > 50:
        result["recommendations"].append("Large storage deployments may benefit from hybrid cloud")

    if not config.hardware.requires_hsm and config.hipaa_compliant:
        result["recommendations"].append("HIPAA compliance strongly recommends HSM for key storage")

    return result


# Example deployment templates
DEPLOYMENT_TEMPLATES = {
    "small_hospital": {
        "resource_class": NodeResourceClass.LIGHT,
        "deployment_mode": DeploymentMode.CLOUD,
        "description": "Small hospital or clinic, cloud-based, minimal resources",
        "estimated_cost_usd_per_month": 500,
    },
    "medium_hospital": {
        "resource_class": NodeResourceClass.FULL,
        "deployment_mode": DeploymentMode.HYBRID,
        "description": "Medium hospital, hybrid deployment, full node capabilities",
        "estimated_cost_usd_per_month": 1500,
    },
    "large_academic_center": {
        "resource_class": NodeResourceClass.ARCHIVE,
        "deployment_mode": DeploymentMode.ON_PREM,
        "description": "Large academic medical center, on-prem, archive capabilities",
        "estimated_cost_usd_per_month": 3000,
    },
    "research_consortium": {
        "resource_class": NodeResourceClass.ARCHIVE,
        "deployment_mode": DeploymentMode.HYBRID,
        "description": "Multi-institutional research consortium, hybrid, full capabilities",
        "estimated_cost_usd_per_month": 4000,
    },
}
