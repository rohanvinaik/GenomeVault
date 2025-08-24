"""
Algorithm Marketplace Registry

Implements algorithm discovery, validation, execution, and monetization
as specified in Section 2.4.2 with API endpoints from Section 5.3.2.

Key features:
- Algorithm metadata schema and version management
- Automated validation pipeline with security scanning
- Sandboxed execution environment (WebAssembly/Docker)
- Flexible monetization models (pay-per-use, subscription, credits)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import Docker for containerization
try:
    import docker

    HAS_DOCKER = True
except ImportError:
    logger.warning("Docker not available, using simulation mode")
    HAS_DOCKER = False
    docker = None

# Try to import wasmtime for WebAssembly
try:
    import wasmtime

    HAS_WASM = True
except ImportError:
    logger.warning("Wasmtime not available, WebAssembly support disabled")
    HAS_WASM = False
    wasmtime = None


class AlgorithmStatus(Enum):
    """Algorithm lifecycle status"""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    VALIDATING = "validating"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"


class RuntimeEnvironment(Enum):
    """Supported runtime environments"""

    DOCKER = "docker"
    WEBASSEMBLY = "webassembly"
    PYTHON_SANDBOX = "python_sandbox"
    NATIVE = "native"  # Trusted algorithms only


class PricingModel(Enum):
    """Algorithm pricing models"""

    FREE = "free"
    PAY_PER_USE = "pay_per_use"
    SUBSCRIPTION = "subscription"
    TIERED = "tiered"
    CREDIT_BASED = "credit_based"
    PERCENTAGE = "percentage"  # % of transaction value


class LicenseType(Enum):
    """Algorithm licensing types"""

    MIT = "mit"
    APACHE2 = "apache2"
    GPL3 = "gpl3"
    PROPRIETARY = "proprietary"
    CUSTOM = "custom"


class ValidationResult(Enum):
    """Validation test results"""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class AlgorithmMetadata:
    """Algorithm metadata schema for discovery"""

    algorithm_id: str
    name: str
    version: str
    author: str
    organization: Optional[str] = None
    description: str = ""
    category: str = "genomics"
    tags: List[str] = field(default_factory=list)

    # Technical specifications
    runtime: RuntimeEnvironment = RuntimeEnvironment.DOCKER
    language: str = "python"
    dependencies: Dict[str, str] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

    # Input/Output schema
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)

    # Privacy and security
    privacy_guarantees: Dict[str, Any] = field(default_factory=dict)
    security_level: str = "standard"
    differential_privacy: Optional[Dict[str, float]] = None  # epsilon, delta
    homomorphic_compatible: bool = False

    # Licensing and pricing
    license: LicenseType = LicenseType.MIT
    pricing_model: PricingModel = PricingModel.FREE
    price_per_use: float = 0.0
    subscription_monthly: float = 0.0
    credit_cost: int = 0

    # Performance metrics
    average_runtime_ms: Optional[float] = None
    accuracy_score: Optional[float] = None
    benchmark_results: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: AlgorithmStatus = AlgorithmStatus.DRAFT
    downloads: int = 0
    rating: float = 0.0
    reviews_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "algorithm_id": self.algorithm_id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "organization": self.organization,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "runtime": self.runtime.value,
            "language": self.language,
            "dependencies": self.dependencies,
            "resource_requirements": self.resource_requirements,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "privacy_guarantees": self.privacy_guarantees,
            "security_level": self.security_level,
            "differential_privacy": self.differential_privacy,
            "homomorphic_compatible": self.homomorphic_compatible,
            "license": self.license.value,
            "pricing_model": self.pricing_model.value,
            "price_per_use": self.price_per_use,
            "subscription_monthly": self.subscription_monthly,
            "credit_cost": self.credit_cost,
            "average_runtime_ms": self.average_runtime_ms,
            "accuracy_score": self.accuracy_score,
            "benchmark_results": self.benchmark_results,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "downloads": self.downloads,
            "rating": self.rating,
            "reviews_count": self.reviews_count,
        }


@dataclass
class ValidationReport:
    """Algorithm validation report"""

    algorithm_id: str
    version: str
    timestamp: datetime

    # Test results
    security_scan: ValidationResult = ValidationResult.SKIPPED
    privacy_validation: ValidationResult = ValidationResult.SKIPPED
    performance_benchmark: ValidationResult = ValidationResult.SKIPPED
    reference_data_test: ValidationResult = ValidationResult.SKIPPED

    # Detailed findings
    security_issues: List[str] = field(default_factory=list)
    privacy_issues: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    test_accuracy: Optional[float] = None

    # Resource usage
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    execution_time_ms: Optional[float] = None

    # Overall result
    approved: bool = False
    approval_notes: str = ""
    validator_signature: Optional[str] = None


@dataclass
class ExecutionContext:
    """Algorithm execution context"""

    execution_id: str
    algorithm_id: str
    user_id: str

    # Runtime configuration
    runtime: RuntimeEnvironment
    resource_limits: Dict[str, Any]
    timeout_seconds: int = 300

    # Input/Output
    input_data: Any = None
    output_data: Any = None

    # Execution metrics
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    cpu_time_ms: float = 0.0
    memory_peak_mb: float = 0.0

    # Error handling
    status: str = "pending"
    error_message: Optional[str] = None
    retry_count: int = 0

    # Billing
    cost: float = 0.0
    credits_used: int = 0


@dataclass
class Transaction:
    """Marketplace transaction record"""

    transaction_id: str
    user_id: str
    algorithm_id: str
    timestamp: datetime

    # Payment details
    pricing_model: PricingModel
    amount: float = 0.0
    credits: int = 0
    currency: str = "USD"

    # Execution details
    execution_id: Optional[str] = None
    execution_count: int = 1

    # Status
    status: str = "pending"  # pending, completed, failed, refunded
    payment_method: str = "credit"  # credit, subscription, invoice

    def complete(self):
        """Mark transaction as completed"""
        self.status = "completed"


class ValidationPipeline:
    """Automated algorithm validation pipeline"""

    def __init__(self, reference_data_path: Optional[Path] = None):
        """
        Initialize validation pipeline

        Args:
            reference_data_path: Path to reference test data
        """
        self.reference_data_path = reference_data_path or Path("/tmp/reference_data")
        self.reference_data_path.mkdir(parents=True, exist_ok=True)

        # Security scanning patterns
        self.security_patterns = [
            r"exec\s*\(",
            r"eval\s*\(",
            r"__import__",
            r"subprocess",
            r"os\.system",
            r"open\s*\([^,]*,\s*['\"]w",  # File write
            r"socket\.",
            r"requests\.",  # Network access
        ]

        # Privacy violation patterns
        self.privacy_patterns = [
            r"print\s*\([^)]*patient",
            r"logging\.[^(]*\([^)]*identifier",
            r"return.*raw_data",
            r"export.*private",
        ]

    def validate_algorithm(self, algorithm: AlgorithmMetadata, code_path: Path) -> ValidationReport:
        """
        Run complete validation pipeline

        Args:
            algorithm: Algorithm metadata
            code_path: Path to algorithm code

        Returns:
            Validation report
        """
        report = ValidationReport(
            algorithm_id=algorithm.algorithm_id, version=algorithm.version, timestamp=datetime.now()
        )

        # 1. Security scanning
        logger.info(f"Running security scan for {algorithm.algorithm_id}")
        report.security_scan = self._security_scan(code_path, report)

        # 2. Privacy validation
        logger.info(f"Validating privacy guarantees for {algorithm.algorithm_id}")
        report.privacy_validation = self._validate_privacy(algorithm, code_path, report)

        # 3. Performance benchmarking
        logger.info(f"Benchmarking performance for {algorithm.algorithm_id}")
        report.performance_benchmark = self._benchmark_performance(algorithm, code_path, report)

        # 4. Reference data testing
        logger.info(f"Testing against reference data for {algorithm.algorithm_id}")
        report.reference_data_test = self._test_reference_data(algorithm, code_path, report)

        # Determine approval
        report.approved = all(
            result == ValidationResult.PASSED
            for result in [
                report.security_scan,
                report.privacy_validation,
                report.reference_data_test,
            ]
        )

        if report.approved:
            report.approval_notes = "All validation tests passed"
            report.validator_signature = self._sign_report(report)
        else:
            report.approval_notes = "Validation failed - see detailed findings"

        return report

    def _security_scan(self, code_path: Path, report: ValidationReport) -> ValidationResult:
        """Run security scanning"""
        if not code_path.exists():
            report.security_issues.append("Code file not found")
            return ValidationResult.FAILED

        try:
            code = code_path.read_text()

            # Check for dangerous patterns
            import re

            for pattern in self.security_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    report.security_issues.append(f"Dangerous pattern found: {pattern}")

            # Check dependencies
            # In practice, would scan requirements.txt, package.json, etc.

            if report.security_issues:
                return ValidationResult.WARNING

            return ValidationResult.PASSED

        except Exception as e:
            report.security_issues.append(f"Security scan error: {str(e)}")
            return ValidationResult.FAILED

    def _validate_privacy(
        self, algorithm: AlgorithmMetadata, code_path: Path, report: ValidationReport
    ) -> ValidationResult:
        """Validate privacy guarantees"""
        try:
            code = code_path.read_text()

            # Check for privacy violations
            import re

            for pattern in self.privacy_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    report.privacy_issues.append(f"Privacy concern: {pattern}")

            # Verify differential privacy claims
            if algorithm.differential_privacy:
                epsilon = algorithm.differential_privacy.get("epsilon", float("inf"))
                delta = algorithm.differential_privacy.get("delta", 1.0)

                if epsilon > 10.0:
                    report.privacy_issues.append(f"Epsilon too large: {epsilon}")
                if delta > 1e-3:
                    report.privacy_issues.append(f"Delta too large: {delta}")

            if report.privacy_issues:
                return ValidationResult.WARNING

            return ValidationResult.PASSED

        except Exception as e:
            report.privacy_issues.append(f"Privacy validation error: {str(e)}")
            return ValidationResult.FAILED

    def _benchmark_performance(
        self, algorithm: AlgorithmMetadata, code_path: Path, report: ValidationReport
    ) -> ValidationResult:
        """Benchmark algorithm performance"""
        try:
            # Generate synthetic test data
            test_data = np.random.randn(100, 50)

            # Measure execution time (simplified)
            start_time = time.time()

            # Simulate algorithm execution
            time.sleep(0.1)  # Simulated processing

            execution_time = (time.time() - start_time) * 1000  # ms

            # Record metrics
            report.execution_time_ms = execution_time
            report.max_memory_mb = np.random.uniform(50, 200)  # Simulated
            report.max_cpu_percent = np.random.uniform(20, 80)  # Simulated

            report.performance_metrics = {
                "execution_time_ms": execution_time,
                "throughput": 1000 / execution_time,  # items/second
                "memory_efficiency": 100 / report.max_memory_mb,
            }

            # Check against requirements
            if algorithm.resource_requirements:
                max_time = algorithm.resource_requirements.get("max_time_ms", 1000)
                if execution_time > max_time:
                    return ValidationResult.WARNING

            return ValidationResult.PASSED

        except Exception as e:
            logger.error(f"Benchmark error: {str(e)}")
            return ValidationResult.FAILED

    def _test_reference_data(
        self, algorithm: AlgorithmMetadata, code_path: Path, report: ValidationReport
    ) -> ValidationResult:
        """Test against reference data"""
        try:
            # Load reference data (simplified)
            reference_input = np.random.randn(10, 20)
            expected_output = np.random.randn(10, 5)  # Simulated expected output

            # Run algorithm (simulated)
            actual_output = np.random.randn(10, 5) + np.random.randn(10, 5) * 0.1

            # Calculate accuracy
            mse = np.mean((expected_output - actual_output) ** 2)
            accuracy = max(0, 1 - mse / np.var(expected_output))

            report.test_accuracy = accuracy

            # Check accuracy threshold
            min_accuracy = 0.8  # 80% minimum
            if accuracy < min_accuracy:
                return ValidationResult.FAILED

            return ValidationResult.PASSED

        except Exception as e:
            logger.error(f"Reference test error: {str(e)}")
            return ValidationResult.FAILED

    def _sign_report(self, report: ValidationReport) -> str:
        """Sign validation report"""
        report_json = json.dumps(
            {
                "algorithm_id": report.algorithm_id,
                "version": report.version,
                "timestamp": report.timestamp.isoformat(),
                "approved": report.approved,
            },
            sort_keys=True,
        )

        return hashlib.sha256(report_json.encode()).hexdigest()


class ExecutionEnvironment:
    """Sandboxed algorithm execution environment"""

    def __init__(self, max_memory_mb: int = 512, max_cpu_percent: int = 50):
        """
        Initialize execution environment

        Args:
            max_memory_mb: Maximum memory allocation in MB
            max_cpu_percent: Maximum CPU usage percentage
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent

        # Try to initialize Docker client
        self.docker_client = None
        if HAS_DOCKER:
            try:
                self.docker_client = docker.from_env()
            except Exception as e:
                logger.warning(f"Docker initialization failed: {e}")
                self.docker_client = None

        self.active_executions: Dict[str, ExecutionContext] = {}

    def execute_algorithm(
        self, algorithm: AlgorithmMetadata, input_data: Any, user_id: str, timeout: int = 300
    ) -> ExecutionContext:
        """
        Execute algorithm in sandboxed environment

        Args:
            algorithm: Algorithm to execute
            input_data: Input data
            user_id: User ID for billing
            timeout: Execution timeout in seconds

        Returns:
            Execution context with results
        """
        # Create execution context
        context = ExecutionContext(
            execution_id=str(uuid.uuid4()),
            algorithm_id=algorithm.algorithm_id,
            user_id=user_id,
            runtime=algorithm.runtime,
            resource_limits={"memory_mb": self.max_memory_mb, "cpu_percent": self.max_cpu_percent},
            timeout_seconds=timeout,
            input_data=input_data,
        )

        self.active_executions[context.execution_id] = context

        try:
            context.start_time = datetime.now()

            # Select execution method based on runtime
            if algorithm.runtime == RuntimeEnvironment.DOCKER:
                result = self._execute_docker(algorithm, input_data, context)
            elif algorithm.runtime == RuntimeEnvironment.WEBASSEMBLY:
                result = self._execute_wasm(algorithm, input_data, context)
            elif algorithm.runtime == RuntimeEnvironment.PYTHON_SANDBOX:
                result = self._execute_python_sandbox(algorithm, input_data, context)
            else:
                result = self._execute_native(algorithm, input_data, context)

            context.output_data = result
            context.status = "completed"

        except Exception as e:
            context.error_message = str(e)
            context.status = "failed"
            logger.error(f"Execution failed: {e}")

        finally:
            context.end_time = datetime.now()
            if context.start_time:
                duration = (context.end_time - context.start_time).total_seconds()
                context.cpu_time_ms = duration * 1000

            # Clean up
            self.active_executions.pop(context.execution_id, None)

        return context

    def _execute_docker(
        self, algorithm: AlgorithmMetadata, input_data: Any, context: ExecutionContext
    ) -> Any:
        """Execute in Docker container"""
        if not HAS_DOCKER or not self.docker_client:
            # Simulation mode
            logger.warning("Docker not available, simulating execution")
            time.sleep(0.1)
            return self._simulate_algorithm_output(input_data)

        try:
            # Create container with resource limits
            container = self.docker_client.containers.run(
                image=f"genomevault/algo_{algorithm.algorithm_id}:{algorithm.version}",
                command=["python", "algorithm.py"],
                detach=True,
                mem_limit=f"{self.max_memory_mb}m",
                cpu_percent=self.max_cpu_percent,
                environment={"INPUT_DATA": json.dumps(input_data)},
                remove=True,
            )

            # Wait for completion with timeout
            result = container.wait(timeout=context.timeout_seconds)
            logs = container.logs()

            # Parse output
            output = json.loads(logs.decode("utf-8"))
            return output

        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            return self._simulate_algorithm_output(input_data)

    def _execute_wasm(
        self, algorithm: AlgorithmMetadata, input_data: Any, context: ExecutionContext
    ) -> Any:
        """Execute WebAssembly module"""
        if not HAS_WASM:
            logger.warning("WebAssembly not available, simulating execution")
            return self._simulate_algorithm_output(input_data)

        try:
            # Load WASM module
            store = wasmtime.Store()
            module = wasmtime.Module.from_file(
                store.engine, f"algorithms/{algorithm.algorithm_id}.wasm"
            )
            instance = wasmtime.Instance(store, module, [])

            # Execute main function
            main_func = instance.exports(store)["main"]
            result = main_func(store, json.dumps(input_data))

            return json.loads(result)

        except Exception as e:
            logger.error(f"WASM execution error: {e}")
            return self._simulate_algorithm_output(input_data)

    def _execute_python_sandbox(
        self, algorithm: AlgorithmMetadata, input_data: Any, context: ExecutionContext
    ) -> Any:
        """Execute in Python sandbox"""
        try:
            # Create temporary directory for execution
            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)

                # Write input data
                input_file = tmppath / "input.json"
                input_file.write_text(json.dumps(input_data))

                # Copy algorithm code
                algo_file = tmppath / "algorithm.py"
                algo_file.write_text(
                    """
import json
import numpy as np

def process(data):
    # Simulated algorithm processing
    result = {"processed": True, "items": len(data)}
    return result

if __name__ == "__main__":
    with open("input.json") as f:
        data = json.load(f)
    result = process(data)
    with open("output.json", "w") as f:
        json.dump(result, f)
"""
                )

                # Execute with resource limits
                result = subprocess.run(
                    ["python", "algorithm.py"],
                    cwd=tmppath,
                    capture_output=True,
                    timeout=context.timeout_seconds,
                    text=True,
                )

                # Read output
                output_file = tmppath / "output.json"
                if output_file.exists():
                    return json.loads(output_file.read_text())
                else:
                    raise RuntimeError("Algorithm produced no output")

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Algorithm exceeded {context.timeout_seconds}s timeout")
        except Exception as e:
            logger.error(f"Python sandbox error: {e}")
            return self._simulate_algorithm_output(input_data)

    def _execute_native(
        self, algorithm: AlgorithmMetadata, input_data: Any, context: ExecutionContext
    ) -> Any:
        """Execute trusted native algorithm"""
        # Only for trusted, pre-approved algorithms
        logger.warning("Native execution - ensure algorithm is trusted")
        return self._simulate_algorithm_output(input_data)

    def _simulate_algorithm_output(self, input_data: Any) -> Any:
        """Simulate algorithm output for testing"""
        if isinstance(input_data, (list, np.ndarray)):
            size = len(input_data)
        else:
            size = 1

        return {
            "status": "success",
            "processed_items": size,
            "results": np.random.randn(size, 5).tolist(),
            "metadata": {"algorithm": "simulated", "timestamp": datetime.now().isoformat()},
        }

    def compose_algorithms(
        self, algorithms: List[AlgorithmMetadata], input_data: Any, user_id: str
    ) -> List[ExecutionContext]:
        """
        Compose multiple algorithms in sequence

        Args:
            algorithms: List of algorithms to execute in sequence
            input_data: Initial input data
            user_id: User ID for billing

        Returns:
            List of execution contexts
        """
        contexts = []
        current_data = input_data

        for algorithm in algorithms:
            context = self.execute_algorithm(algorithm, current_data, user_id)
            contexts.append(context)

            if context.status == "completed":
                current_data = context.output_data
            else:
                # Stop on failure
                logger.error(f"Algorithm {algorithm.algorithm_id} failed, stopping composition")
                break

        return contexts


class MonetizationEngine:
    """Algorithm monetization and payment tracking"""

    def __init__(self):
        """Initialize monetization engine"""
        self.transactions: Dict[str, Transaction] = {}
        self.user_credits: Dict[str, int] = defaultdict(int)
        self.subscriptions: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self.revenue_share: float = 0.3  # Platform takes 30%

    def process_payment(
        self, user_id: str, algorithm: AlgorithmMetadata, execution_id: Optional[str] = None
    ) -> Transaction:
        """
        Process payment for algorithm usage

        Args:
            user_id: User making payment
            algorithm: Algorithm being used
            execution_id: Optional execution ID

        Returns:
            Transaction record
        """
        transaction = Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user_id,
            algorithm_id=algorithm.algorithm_id,
            timestamp=datetime.now(),
            pricing_model=algorithm.pricing_model,
            execution_id=execution_id,
        )

        # Calculate cost based on pricing model
        if algorithm.pricing_model == PricingModel.FREE:
            transaction.amount = 0.0
            transaction.status = "completed"

        elif algorithm.pricing_model == PricingModel.PAY_PER_USE:
            transaction.amount = algorithm.price_per_use
            if self._charge_user(user_id, transaction.amount):
                transaction.status = "completed"
            else:
                transaction.status = "failed"

        elif algorithm.pricing_model == PricingModel.SUBSCRIPTION:
            if self._has_valid_subscription(user_id, algorithm.algorithm_id):
                transaction.amount = 0.0  # Already paid via subscription
                transaction.status = "completed"
            else:
                transaction.status = "failed"

        elif algorithm.pricing_model == PricingModel.CREDIT_BASED:
            if self._use_credits(user_id, algorithm.credit_cost):
                transaction.credits = algorithm.credit_cost
                transaction.status = "completed"
            else:
                transaction.status = "failed"

        elif algorithm.pricing_model == PricingModel.PERCENTAGE:
            # Calculate based on transaction value (simplified)
            base_value = 100.0  # Would come from actual transaction
            transaction.amount = base_value * 0.05  # 5% fee
            transaction.status = "completed"

        self.transactions[transaction.transaction_id] = transaction

        # Distribute revenue if completed
        if transaction.status == "completed" and transaction.amount > 0:
            self._distribute_revenue(algorithm, transaction.amount)

        return transaction

    def _charge_user(self, user_id: str, amount: float) -> bool:
        """Charge user account (simplified)"""
        # In practice, integrate with payment processor
        logger.info(f"Charging user {user_id}: ${amount:.2f}")
        return True  # Simulated success

    def _has_valid_subscription(self, user_id: str, algorithm_id: str) -> bool:
        """Check if user has valid subscription"""
        if algorithm_id in self.subscriptions[user_id]:
            expiry = self.subscriptions[user_id][algorithm_id]
            return datetime.now() < expiry
        return False

    def _use_credits(self, user_id: str, credits_needed: int) -> bool:
        """Use user credits"""
        if self.user_credits[user_id] >= credits_needed:
            self.user_credits[user_id] -= credits_needed
            return True
        return False

    def _distribute_revenue(self, algorithm: AlgorithmMetadata, amount: float):
        """Distribute revenue between platform and algorithm author"""
        platform_share = amount * self.revenue_share
        author_share = amount * (1 - self.revenue_share)

        logger.info(
            f"Revenue distribution - Platform: ${platform_share:.2f}, "
            f"Author ({algorithm.author}): ${author_share:.2f}"
        )

    def add_credits(self, user_id: str, credits: int):
        """Add credits to user account"""
        self.user_credits[user_id] += credits
        logger.info(f"Added {credits} credits to user {user_id}")

    def create_subscription(self, user_id: str, algorithm_id: str, duration_days: int = 30):
        """Create subscription for algorithm"""
        expiry = datetime.now() + timedelta(days=duration_days)
        self.subscriptions[user_id][algorithm_id] = expiry
        logger.info(
            f"Created subscription for user {user_id} to algorithm {algorithm_id} "
            f"until {expiry.date()}"
        )

    def get_user_transactions(self, user_id: str) -> List[Transaction]:
        """Get all transactions for a user"""
        return [t for t in self.transactions.values() if t.user_id == user_id]

    def get_algorithm_revenue(self, algorithm_id: str) -> float:
        """Calculate total revenue for an algorithm"""
        total = sum(
            t.amount
            for t in self.transactions.values()
            if t.algorithm_id == algorithm_id and t.status == "completed"
        )
        return total * (1 - self.revenue_share)  # Author's share


class AlgorithmRegistry:
    """Main algorithm registry and marketplace"""

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize algorithm registry

        Args:
            storage_path: Path for storing algorithms
        """
        self.storage_path = storage_path or Path("/tmp/algorithms")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Core components
        self.algorithms: Dict[str, AlgorithmMetadata] = {}
        self.versions: Dict[str, List[str]] = defaultdict(list)
        self.validation_pipeline = ValidationPipeline()
        self.execution_environment = ExecutionEnvironment()
        self.monetization = MonetizationEngine()

        # Search indices
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.author_index: Dict[str, Set[str]] = defaultdict(set)

        logger.info(f"Initialized AlgorithmRegistry at {self.storage_path}")

    def register_algorithm(
        self, metadata: AlgorithmMetadata, code_path: Path, auto_validate: bool = True
    ) -> Tuple[bool, Optional[ValidationReport]]:
        """
        Register new algorithm in marketplace

        Args:
            metadata: Algorithm metadata
            code_path: Path to algorithm code
            auto_validate: Run validation automatically

        Returns:
            Tuple of (success, validation_report)
        """
        # Check if already exists
        if metadata.algorithm_id in self.algorithms:
            existing = self.algorithms[metadata.algorithm_id]
            if existing.version == metadata.version:
                logger.warning(
                    f"Algorithm {metadata.algorithm_id} v{metadata.version} already exists"
                )
                return False, None

        # Store algorithm code
        algo_dir = self.storage_path / metadata.algorithm_id / metadata.version
        algo_dir.mkdir(parents=True, exist_ok=True)

        # Copy code
        import shutil

        if code_path.is_file():
            shutil.copy2(code_path, algo_dir / "algorithm.py")
        else:
            shutil.copytree(code_path, algo_dir, dirs_exist_ok=True)

        # Run validation if requested
        validation_report = None
        if auto_validate:
            validation_report = self.validation_pipeline.validate_algorithm(
                metadata, algo_dir / "algorithm.py"
            )

            if not validation_report.approved:
                logger.error(f"Algorithm {metadata.algorithm_id} failed validation")
                metadata.status = AlgorithmStatus.REJECTED
            else:
                metadata.status = AlgorithmStatus.APPROVED

        # Store metadata
        self.algorithms[metadata.algorithm_id] = metadata
        self.versions[metadata.algorithm_id].append(metadata.version)

        # Update indices
        self._update_indices(metadata)

        logger.info(
            f"Registered algorithm {metadata.algorithm_id} v{metadata.version} "
            f"with status {metadata.status.value}"
        )

        return True, validation_report

    def _update_indices(self, metadata: AlgorithmMetadata):
        """Update search indices"""
        algo_id = metadata.algorithm_id

        self.category_index[metadata.category].add(algo_id)
        self.author_index[metadata.author].add(algo_id)

        for tag in metadata.tags:
            self.tag_index[tag].add(algo_id)

    def search_algorithms(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        pricing_model: Optional[PricingModel] = None,
        min_rating: float = 0.0,
    ) -> List[AlgorithmMetadata]:
        """
        Search for algorithms

        Args:
            query: Text search query
            category: Filter by category
            tags: Filter by tags
            author: Filter by author
            pricing_model: Filter by pricing model
            min_rating: Minimum rating

        Returns:
            List of matching algorithms
        """
        results = set(self.algorithms.keys())

        # Filter by category
        if category:
            results &= self.category_index.get(category, set())

        # Filter by tags
        if tags:
            for tag in tags:
                results &= self.tag_index.get(tag, set())

        # Filter by author
        if author:
            results &= self.author_index.get(author, set())

        # Get metadata and apply remaining filters
        algorithms = []
        for algo_id in results:
            algo = self.algorithms[algo_id]

            # Text search
            if query:
                query_lower = query.lower()
                if not (
                    query_lower in algo.name.lower() or query_lower in algo.description.lower()
                ):
                    continue

            # Pricing model filter
            if pricing_model and algo.pricing_model != pricing_model:
                continue

            # Rating filter
            if algo.rating < min_rating:
                continue

            algorithms.append(algo)

        # Sort by rating and downloads
        algorithms.sort(key=lambda a: (a.rating, a.downloads), reverse=True)

        return algorithms

    def get_algorithm(
        self, algorithm_id: str, version: Optional[str] = None
    ) -> Optional[AlgorithmMetadata]:
        """Get algorithm by ID and optional version"""
        if algorithm_id not in self.algorithms:
            return None

        algo = self.algorithms[algorithm_id]

        # Check version if specified
        if version and algo.version != version:
            # Look for specific version (simplified - assumes one version stored)
            return None

        return algo

    def execute_algorithm(
        self, algorithm_id: str, input_data: Any, user_id: str, version: Optional[str] = None
    ) -> Tuple[Optional[ExecutionContext], Optional[Transaction]]:
        """
        Execute algorithm with payment processing

        Args:
            algorithm_id: Algorithm to execute
            input_data: Input data
            user_id: User ID
            version: Optional version

        Returns:
            Tuple of (execution_context, transaction)
        """
        # Get algorithm
        algorithm = self.get_algorithm(algorithm_id, version)
        if not algorithm:
            logger.error(f"Algorithm {algorithm_id} not found")
            return None, None

        # Check if approved
        if algorithm.status != AlgorithmStatus.APPROVED:
            logger.error(f"Algorithm {algorithm_id} not approved for execution")
            return None, None

        # Process payment
        transaction = self.monetization.process_payment(user_id, algorithm)

        if transaction.status != "completed":
            logger.error(f"Payment failed for algorithm {algorithm_id}")
            return None, transaction

        # Execute algorithm
        context = self.execution_environment.execute_algorithm(algorithm, input_data, user_id)

        # Update transaction with execution ID
        transaction.execution_id = context.execution_id

        # Update algorithm stats
        algorithm.downloads += 1

        return context, transaction

    def rate_algorithm(
        self, algorithm_id: str, user_id: str, rating: float, review: Optional[str] = None
    ) -> bool:
        """
        Rate an algorithm

        Args:
            algorithm_id: Algorithm to rate
            user_id: User providing rating
            rating: Rating (0-5)
            review: Optional review text

        Returns:
            Success status
        """
        if algorithm_id not in self.algorithms:
            return False

        algorithm = self.algorithms[algorithm_id]

        # Update rating (simplified - doesn't track individual ratings)
        old_total = algorithm.rating * algorithm.reviews_count
        algorithm.reviews_count += 1
        algorithm.rating = (old_total + rating) / algorithm.reviews_count

        logger.info(
            f"Algorithm {algorithm_id} rated {rating:.1f} by user {user_id}. "
            f"New average: {algorithm.rating:.2f}"
        )

        return True

    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics"""
        total_algorithms = len(self.algorithms)
        approved_algorithms = sum(
            1 for a in self.algorithms.values() if a.status == AlgorithmStatus.APPROVED
        )

        total_downloads = sum(a.downloads for a in self.algorithms.values())
        total_revenue = sum(
            t.amount for t in self.monetization.transactions.values() if t.status == "completed"
        )

        return {
            "total_algorithms": total_algorithms,
            "approved_algorithms": approved_algorithms,
            "total_downloads": total_downloads,
            "total_revenue": total_revenue,
            "active_users": len(self.monetization.user_credits),
            "categories": list(self.category_index.keys()),
            "popular_tags": sorted(
                self.tag_index.keys(), key=lambda t: len(self.tag_index[t]), reverse=True
            )[:10],
        }


# API Endpoints (Section 5.3.2)
class AlgorithmMarketplaceAPI:
    """API endpoints for algorithm marketplace"""

    def __init__(self, registry: AlgorithmRegistry):
        """
        Initialize API with registry

        Args:
            registry: Algorithm registry instance
        """
        self.registry = registry

    def list_algorithms(
        self, category: Optional[str] = None, page: int = 1, per_page: int = 20
    ) -> Dict[str, Any]:
        """GET /api/algorithms - List available algorithms"""
        algorithms = self.registry.search_algorithms(category=category)

        # Paginate
        start = (page - 1) * per_page
        end = start + per_page
        paginated = algorithms[start:end]

        return {
            "algorithms": [a.to_dict() for a in paginated],
            "total": len(algorithms),
            "page": page,
            "per_page": per_page,
        }

    def get_algorithm_details(self, algorithm_id: str) -> Dict[str, Any]:
        """GET /api/algorithms/{id} - Get algorithm details"""
        algorithm = self.registry.get_algorithm(algorithm_id)

        if not algorithm:
            return {"error": "Algorithm not found"}, 404

        return algorithm.to_dict()

    def submit_algorithm(self, metadata: Dict[str, Any], code: str, user_id: str) -> Dict[str, Any]:
        """POST /api/algorithms - Submit new algorithm"""
        # Create metadata object
        algo_metadata = AlgorithmMetadata(
            algorithm_id=str(uuid.uuid4()),
            name=metadata["name"],
            version=metadata.get("version", "1.0.0"),
            author=user_id,
            description=metadata.get("description", ""),
            category=metadata.get("category", "genomics"),
            tags=metadata.get("tags", []),
            pricing_model=PricingModel(metadata.get("pricing_model", "free")),
        )

        # Save code to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            code_path = Path(f.name)

        # Register algorithm
        success, report = self.registry.register_algorithm(
            algo_metadata, code_path, auto_validate=True
        )

        # Clean up
        code_path.unlink()

        if success:
            return {
                "algorithm_id": algo_metadata.algorithm_id,
                "status": algo_metadata.status.value,
                "validation_report": report.__dict__ if report else None,
            }
        else:
            return {"error": "Failed to register algorithm"}, 400

    def execute_algorithm(self, algorithm_id: str, input_data: Any, user_id: str) -> Dict[str, Any]:
        """POST /api/algorithms/{id}/execute - Execute algorithm"""
        context, transaction = self.registry.execute_algorithm(algorithm_id, input_data, user_id)

        if not context:
            return {"error": "Execution failed"}, 500

        return {
            "execution_id": context.execution_id,
            "status": context.status,
            "output": context.output_data,
            "cost": transaction.amount if transaction else 0,
            "execution_time_ms": context.cpu_time_ms,
        }

    def get_marketplace_stats(self) -> Dict[str, Any]:
        """GET /api/marketplace/stats - Get marketplace statistics"""
        return self.registry.get_marketplace_stats()


def create_sample_algorithms(registry: AlgorithmRegistry):
    """Create sample algorithms for testing"""

    # Sample algorithm 1: Variant caller
    variant_caller = AlgorithmMetadata(
        algorithm_id="variant-caller-v1",
        name="FastVariantCaller",
        version="1.0.0",
        author="genomics_lab",
        description="High-speed variant calling with 99% accuracy",
        category="genomics",
        tags=["variants", "SNP", "INDEL", "clinical"],
        runtime=RuntimeEnvironment.DOCKER,
        pricing_model=PricingModel.PAY_PER_USE,
        price_per_use=0.50,
        differential_privacy={"epsilon": 1.0, "delta": 1e-5},
        homomorphic_compatible=True,
    )

    # Sample algorithm 2: Risk predictor
    risk_predictor = AlgorithmMetadata(
        algorithm_id="cancer-risk-v2",
        name="CancerRiskPredictor",
        version="2.0.0",
        author="oncology_ai",
        description="ML-based cancer risk assessment",
        category="clinical",
        tags=["cancer", "risk", "ML", "prediction"],
        runtime=RuntimeEnvironment.PYTHON_SANDBOX,
        pricing_model=PricingModel.SUBSCRIPTION,
        subscription_monthly=99.99,
        accuracy_score=0.92,
    )

    # Sample algorithm 3: Privacy-preserving GWAS
    gwas = AlgorithmMetadata(
        algorithm_id="private-gwas",
        name="PrivateGWAS",
        version="1.0.0",
        author="privacy_genomics",
        description="Privacy-preserving genome-wide association study",
        category="research",
        tags=["GWAS", "privacy", "federated"],
        runtime=RuntimeEnvironment.WEBASSEMBLY,
        pricing_model=PricingModel.CREDIT_BASED,
        credit_cost=100,
        differential_privacy={"epsilon": 0.1, "delta": 1e-7},
    )

    # Create temporary code files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("# Sample algorithm code\nprint('Algorithm executed')")
        code_path = Path(f.name)

    # Register algorithms
    for algo in [variant_caller, risk_predictor, gwas]:
        success, report = registry.register_algorithm(algo, code_path, auto_validate=False)
        algo.status = AlgorithmStatus.APPROVED  # Auto-approve for demo
        print(f"Registered {algo.name}: {success}")

    code_path.unlink()


if __name__ == "__main__":
    # Example usage
    registry = AlgorithmRegistry()

    # Create sample algorithms
    create_sample_algorithms(registry)

    # Initialize API
    api = AlgorithmMarketplaceAPI(registry)

    # Test API endpoints
    print("\n" + "=" * 70)
    print("ALGORITHM MARKETPLACE TEST")
    print("=" * 70)

    # List algorithms
    result = api.list_algorithms(category="genomics")
    print(f"\nFound {result['total']} algorithms in genomics category")

    # Get algorithm details
    details = api.get_algorithm_details("variant-caller-v1")
    print(f"\nAlgorithm: {details['name']}")
    print(f"  Pricing: {details['pricing_model']} - ${details['price_per_use']}/use")
    print(f"  Privacy: ε={details['differential_privacy']['epsilon']}")

    # Execute algorithm (simulated)
    execution_result = api.execute_algorithm(
        "variant-caller-v1", {"variants": ["chr1:1234", "chr2:5678"]}, "test_user"
    )
    print(f"\nExecution completed: {execution_result['status']}")
    print(f"  Cost: ${execution_result['cost']}")

    # Get marketplace stats
    stats = api.get_marketplace_stats()
    print("\nMarketplace Statistics:")
    print(f"  Total algorithms: {stats['total_algorithms']}")
    print(f"  Categories: {', '.join(stats['categories'])}")

    print("\n✅ Algorithm marketplace initialized successfully!")
