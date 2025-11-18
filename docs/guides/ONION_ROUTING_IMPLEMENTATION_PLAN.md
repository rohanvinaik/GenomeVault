# Onion Routing Enhancements - Implementation Plan

**Document Version:** 1.0.0
**Date:** October 24, 2025
**Status:** Planning Phase
**Target Completion:** Phase 1 (4 weeks), Phase 2 (3 months), Phase 3 (6 months)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Module Structure](#module-structure)
3. [Configuration System](#configuration-system)
4. [CLI Interface](#cli-interface)
5. [API Integration](#api-integration)
6. [Implementation Phases](#implementation-phases)
7. [Testing Strategy](#testing-strategy)
8. [Performance Benchmarking](#performance-benchmarking)
9. [Documentation Plan](#documentation-plan)
10. [Migration Path](#migration-path)

---

## Executive Summary

### Objectives

Implement 8 onion routing-inspired enhancements to GenomeVault's privacy architecture:

| Phase | Enhancements | Timeline | Privacy Gain | Complexity |
|-------|-------------|----------|--------------|------------|
| **Phase 1** | Metadata DP, Cover Traffic, Garlic Routing | 4 weeks | +3 bits | Low |
| **Phase 2** | Mix Networks, Rendezvous, PIR-Tor | 3 months | +6.6 bits | Medium |
| **Phase 3** | Threshold Crypto, Layered ZK | 6 months | Distributed trust | High |

### Key Principles

1. **Backwards Compatibility**: All enhancements are opt-in via configuration
2. **Composability**: Each enhancement works independently or in combination
3. **Performance Tuning**: User-configurable trade-offs between privacy and latency
4. **Production Ready**: Full testing, benchmarking, and documentation

---

## Module Structure

### Directory Organization

```
genomevault/
├── onion_routing/                    # New module for all enhancements
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   │
│   ├── mix_networks/                 # Enhancement 1
│   │   ├── __init__.py
│   │   ├── query_batcher.py          # Batching logic
│   │   ├── shuffler.py               # Shuffle algorithm
│   │   └── token_manager.py          # Unlinkable tokens
│   │
│   ├── threshold_crypto/             # Enhancement 2
│   │   ├── __init__.py
│   │   ├── secret_sharing.py         # Shamir secret sharing
│   │   ├── shard_manager.py          # Distribute/reconstruct
│   │   └── institutional_node.py     # Institution interface
│   │
│   ├── layered_zk/                   # Enhancement 3
│   │   ├── __init__.py
│   │   ├── layer_verifier.py         # Per-layer verification
│   │   ├── circuits/                 # ZK circuits for each layer
│   │   │   ├── consensus_correctness.circom
│   │   │   ├── pooling_correctness.circom
│   │   │   ├── query_correctness.circom
│   │   │   └── encoding_correctness.circom
│   │   └── commitment_chain.py       # Link layer commitments
│   │
│   ├── cover_traffic/                # Enhancement 4
│   │   ├── __init__.py
│   │   ├── dummy_generator.py        # Dummy query generation
│   │   ├── background_scheduler.py   # Continuous background queries
│   │   └── statistics_tracker.py     # Privacy metrics
│   │
│   ├── rendezvous/                   # Enhancement 5
│   │   ├── __init__.py
│   │   ├── client_protocol.py        # Client-side logic
│   │   ├── server_protocol.py        # Server-side logic
│   │   ├── rendezvous_node.py        # Routing layer
│   │   └── token_channel.py          # Result delivery
│   │
│   ├── metadata_dp/                  # Enhancement 6
│   │   ├── __init__.py
│   │   ├── timing_obfuscation.py     # Timing noise
│   │   ├── size_padding.py           # Size obfuscation
│   │   └── dp_guarantees.py          # Formal DP proofs
│   │
│   ├── pir_tor/                      # Enhancement 7
│   │   ├── __init__.py
│   │   ├── tor_client.py             # Tor integration
│   │   ├── circuit_manager.py        # Tor circuit management
│   │   └── onion_pir_wrapper.py      # PIR through Tor
│   │
│   ├── garlic_routing/               # Enhancement 8
│   │   ├── __init__.py
│   │   ├── query_bundler.py          # Bundle multiple queries
│   │   ├── unbundler.py              # Extract queries
│   │   └── padding_strategy.py       # Anti-traffic-analysis
│   │
│   └── integration/                  # Cross-enhancement coordination
│       ├── __init__.py
│       ├── enhancement_manager.py    # Enable/disable enhancements
│       ├── metrics_collector.py      # Performance tracking
│       └── security_validator.py     # Verify security properties
│
├── api/
│   └── routers/
│       └── onion_routing.py          # New API endpoints
│
├── cli/
│   └── onion_routing_cli.py          # New CLI commands
│
└── config/
    └── onion_routing.yaml            # Configuration file
```

---

## Configuration System

### Configuration File Format

**File:** `genomevault/config/onion_routing.yaml`

```yaml
# Onion Routing Enhancements Configuration
# Version: 1.0.0

onion_routing:
  enabled: false  # Master switch (opt-in)

  # Enhancement 1: Mix Networks
  mix_networks:
    enabled: false
    batch_size: 100           # Queries per batch
    delay_window: 60          # Seconds to wait
    mode: adaptive            # fixed | adaptive
    adaptive_thresholds:
      high_traffic: 10        # queries/sec → small batches
      low_traffic: 1          # queries/sec → large batches

  # Enhancement 2: Threshold Cryptography
  threshold_crypto:
    enabled: false
    threshold: 3              # k (minimum shares to reconstruct)
    total_shares: 5           # n (total shares created)
    institutions:             # List of participating institutions
      - name: "Institution A"
        endpoint: "https://inst-a.genomevault.org/api"
        public_key: "inst_a_pubkey.pem"
      - name: "Institution B"
        endpoint: "https://inst-b.genomevault.org/api"
        public_key: "inst_b_pubkey.pem"
      # ... (3 more institutions)

  # Enhancement 3: Layered ZK Verification
  layered_zk:
    enabled: false
    verify_layer1: true       # Consensus correctness
    verify_layer2: true       # Pooling correctness
    verify_layer3: true       # Query correctness
    verify_layer4: true       # Encoding correctness
    parallel_verification: true
    trusted_setup_path: "data/zk_setups/layered/"

  # Enhancement 4: Cover Traffic
  cover_traffic:
    enabled: false
    dummy_ratio: 3.0          # Dummy:real query ratio
    real_query_rate: 10       # Queries per hour
    distribution: uniform     # uniform | realistic
    background_enabled: true  # Run continuous background task

  # Enhancement 5: Rendezvous Protocol
  rendezvous:
    enabled: false
    rendezvous_nodes:         # List of rendezvous points
      - "https://rdv1.genomevault.org"
      - "https://rdv2.genomevault.org"
    selection: random         # random | round_robin | load_balanced
    timeout: 60               # Seconds

  # Enhancement 6: Metadata Differential Privacy
  metadata_dp:
    enabled: false
    epsilon: 1.0              # Privacy parameter (lower = more private)
    delta: 0.00001            # Failure probability
    timing_noise: true        # Obfuscate query timing
    size_padding: true        # Pad result sizes
    frequency_jitter: true    # Randomize query frequency

  # Enhancement 7: PIR-Tor Integration
  pir_tor:
    enabled: false
    tor_control_port: 9051
    circuit_timeout: 300      # Seconds
    use_fresh_circuits: true  # New circuit per query
    entry_nodes: []           # Optional: preferred entry nodes

  # Enhancement 8: Garlic Routing
  garlic_routing:
    enabled: false
    bundle_size: 5            # Queries per bundle
    padding_size: 1024        # Bytes of random padding
    bundling_strategy: opportunistic  # opportunistic | forced

# Global privacy mode presets
presets:
  standard:
    onion_routing.enabled: false

  enhanced_privacy:
    onion_routing.enabled: true
    mix_networks.enabled: true
    cover_traffic.enabled: true
    metadata_dp.enabled: true
    garlic_routing.enabled: true

  maximum_privacy:
    onion_routing.enabled: true
    mix_networks.enabled: true
    mix_networks.batch_size: 1000
    cover_traffic.enabled: true
    cover_traffic.dummy_ratio: 10.0
    threshold_crypto.enabled: true
    layered_zk.enabled: true
    rendezvous.enabled: true
    metadata_dp.enabled: true
    metadata_dp.epsilon: 0.5
    pir_tor.enabled: true
    garlic_routing.enabled: true

# Performance tuning
performance:
  # Acceptable latency increase (used for adaptive tuning)
  max_latency_increase: 1000  # milliseconds

  # Bandwidth budget (used for cover traffic tuning)
  max_bandwidth_overhead: 5.0  # multiplier (5.0 = 5× original)

  # CPU budget (used for ZK proof generation tuning)
  max_cpu_overhead: 3.0        # multiplier
```

### Configuration Manager

**File:** `genomevault/onion_routing/config.py`

```python
from dataclasses import dataclass
from typing import List, Optional
import yaml
from pathlib import Path

@dataclass
class MixNetworksConfig:
    enabled: bool = False
    batch_size: int = 100
    delay_window: int = 60
    mode: str = "adaptive"

@dataclass
class ThresholdCryptoConfig:
    enabled: bool = False
    threshold: int = 3
    total_shares: int = 5
    institutions: List[dict] = None

@dataclass
class LayeredZKConfig:
    enabled: bool = False
    verify_layer1: bool = True
    verify_layer2: bool = True
    verify_layer3: bool = True
    verify_layer4: bool = True
    parallel_verification: bool = True
    trusted_setup_path: str = "data/zk_setups/layered/"

@dataclass
class CoverTrafficConfig:
    enabled: bool = False
    dummy_ratio: float = 3.0
    real_query_rate: int = 10
    distribution: str = "uniform"
    background_enabled: bool = True

@dataclass
class RendezvousConfig:
    enabled: bool = False
    rendezvous_nodes: List[str] = None
    selection: str = "random"
    timeout: int = 60

@dataclass
class MetadataDPConfig:
    enabled: bool = False
    epsilon: float = 1.0
    delta: float = 0.00001
    timing_noise: bool = True
    size_padding: bool = True
    frequency_jitter: bool = True

@dataclass
class PIRTorConfig:
    enabled: bool = False
    tor_control_port: int = 9051
    circuit_timeout: int = 300
    use_fresh_circuits: bool = True
    entry_nodes: List[str] = None

@dataclass
class GarlicRoutingConfig:
    enabled: bool = False
    bundle_size: int = 5
    padding_size: int = 1024
    bundling_strategy: str = "opportunistic"

@dataclass
class OnionRoutingConfig:
    """Complete configuration for all onion routing enhancements."""
    enabled: bool = False
    mix_networks: MixNetworksConfig = None
    threshold_crypto: ThresholdCryptoConfig = None
    layered_zk: LayeredZKConfig = None
    cover_traffic: CoverTrafficConfig = None
    rendezvous: RendezvousConfig = None
    metadata_dp: MetadataDPConfig = None
    pir_tor: PIRTorConfig = None
    garlic_routing: GarlicRoutingConfig = None

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'OnionRoutingConfig':
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "onion_routing.yaml"

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Parse configuration
        onion_config = config_data.get('onion_routing', {})

        return cls(
            enabled=onion_config.get('enabled', False),
            mix_networks=MixNetworksConfig(**onion_config.get('mix_networks', {})),
            threshold_crypto=ThresholdCryptoConfig(**onion_config.get('threshold_crypto', {})),
            layered_zk=LayeredZKConfig(**onion_config.get('layered_zk', {})),
            cover_traffic=CoverTrafficConfig(**onion_config.get('cover_traffic', {})),
            rendezvous=RendezvousConfig(**onion_config.get('rendezvous', {})),
            metadata_dp=MetadataDPConfig(**onion_config.get('metadata_dp', {})),
            pir_tor=PIRTorConfig(**onion_config.get('pir_tor', {})),
            garlic_routing=GarlicRoutingConfig(**onion_config.get('garlic_routing', {}))
        )

    def apply_preset(self, preset_name: str):
        """Apply a privacy preset (standard, enhanced_privacy, maximum_privacy)."""
        presets = {
            'standard': {
                'enabled': False
            },
            'enhanced_privacy': {
                'enabled': True,
                'mix_networks.enabled': True,
                'cover_traffic.enabled': True,
                'metadata_dp.enabled': True,
                'garlic_routing.enabled': True
            },
            'maximum_privacy': {
                'enabled': True,
                'mix_networks.enabled': True,
                'mix_networks.batch_size': 1000,
                'cover_traffic.enabled': True,
                'cover_traffic.dummy_ratio': 10.0,
                'threshold_crypto.enabled': True,
                'layered_zk.enabled': True,
                'rendezvous.enabled': True,
                'metadata_dp.enabled': True,
                'metadata_dp.epsilon': 0.5,
                'pir_tor.enabled': True,
                'garlic_routing.enabled': True
            }
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        # Apply preset values
        preset = presets[preset_name]
        for key, value in preset.items():
            # Handle nested attributes (e.g., "mix_networks.enabled")
            if '.' in key:
                module, attr = key.split('.')
                setattr(getattr(self, module), attr, value)
            else:
                setattr(self, key, value)

    def get_active_enhancements(self) -> List[str]:
        """Return list of enabled enhancements."""
        if not self.enabled:
            return []

        active = []
        if self.mix_networks.enabled:
            active.append("mix_networks")
        if self.threshold_crypto.enabled:
            active.append("threshold_crypto")
        if self.layered_zk.enabled:
            active.append("layered_zk")
        if self.cover_traffic.enabled:
            active.append("cover_traffic")
        if self.rendezvous.enabled:
            active.append("rendezvous")
        if self.metadata_dp.enabled:
            active.append("metadata_dp")
        if self.pir_tor.enabled:
            active.append("pir_tor")
        if self.garlic_routing.enabled:
            active.append("garlic_routing")

        return active

    def estimate_overhead(self) -> dict:
        """Estimate performance overhead from enabled enhancements."""
        latency_ms = 0
        bandwidth_mult = 1.0
        cpu_mult = 1.0

        if self.mix_networks.enabled:
            latency_ms += self.mix_networks.delay_window * 500  # Average delay

        if self.cover_traffic.enabled:
            bandwidth_mult *= (1 + self.cover_traffic.dummy_ratio)
            cpu_mult *= (1 + self.cover_traffic.dummy_ratio)

        if self.threshold_crypto.enabled:
            latency_ms += 15  # Parallel retrieval overhead

        if self.layered_zk.enabled:
            latency_ms += 2500  # ZK proof generation
            cpu_mult *= 1.5

        if self.rendezvous.enabled:
            latency_ms += 15  # Extra hop

        if self.metadata_dp.enabled:
            latency_ms += 5  # Noise generation
            bandwidth_mult *= 1.1  # Padding

        if self.pir_tor.enabled:
            latency_ms += 500  # Tor routing

        if self.garlic_routing.enabled:
            bandwidth_mult *= 0.8  # Bundling reduces bandwidth

        return {
            'latency_increase_ms': latency_ms,
            'bandwidth_multiplier': bandwidth_mult,
            'cpu_multiplier': cpu_mult,
            'acceptable': (
                latency_ms <= 1000 and
                bandwidth_mult <= 5.0 and
                cpu_mult <= 3.0
            )
        }
```

---

## CLI Interface

### New CLI Commands

**File:** `genomevault/cli/onion_routing_cli.py`

```python
import click
from genomevault.onion_routing.config import OnionRoutingConfig
from genomevault.onion_routing.integration.enhancement_manager import EnhancementManager

@click.group(name='onion')
def onion_routing_cli():
    """Onion routing enhancements for GenomeVault."""
    pass

@onion_routing_cli.command(name='status')
def status():
    """Show status of onion routing enhancements."""
    config = OnionRoutingConfig.load()

    if not config.enabled:
        click.echo("❌ Onion routing enhancements: DISABLED")
        return

    click.echo("✅ Onion routing enhancements: ENABLED\n")

    active = config.get_active_enhancements()
    click.echo(f"Active enhancements ({len(active)}/8):")

    enhancements = {
        'mix_networks': 'Mix Networks (batching + shuffling)',
        'threshold_crypto': 'Threshold Cryptography (distributed storage)',
        'layered_zk': 'Layered ZK Verification (all 4 layers)',
        'cover_traffic': 'Cover Traffic (dummy queries)',
        'rendezvous': 'Rendezvous Protocol (indirect addressing)',
        'metadata_dp': 'Metadata Differential Privacy',
        'pir_tor': 'PIR-Tor Integration',
        'garlic_routing': 'Garlic Routing (query bundling)'
    }

    for key, name in enhancements.items():
        status_icon = "✅" if key in active else "❌"
        click.echo(f"  {status_icon} {name}")

    # Show performance overhead
    click.echo("\nEstimated performance overhead:")
    overhead = config.estimate_overhead()
    click.echo(f"  Latency: +{overhead['latency_increase_ms']}ms")
    click.echo(f"  Bandwidth: {overhead['bandwidth_multiplier']:.1f}×")
    click.echo(f"  CPU: {overhead['cpu_multiplier']:.1f}×")

    if overhead['acceptable']:
        click.echo("  ✅ Within acceptable limits")
    else:
        click.echo("  ⚠️  Exceeds recommended limits")

@onion_routing_cli.command(name='enable')
@click.option('--preset', type=click.Choice(['standard', 'enhanced_privacy', 'maximum_privacy']),
              help='Apply a privacy preset')
@click.option('--enhancement', multiple=True,
              type=click.Choice(['mix_networks', 'threshold_crypto', 'layered_zk',
                                 'cover_traffic', 'rendezvous', 'metadata_dp',
                                 'pir_tor', 'garlic_routing']),
              help='Enable specific enhancement(s)')
def enable(preset, enhancement):
    """Enable onion routing enhancements."""
    config = OnionRoutingConfig.load()
    config.enabled = True

    if preset:
        config.apply_preset(preset)
        click.echo(f"✅ Applied preset: {preset}")

    if enhancement:
        for enh in enhancement:
            setattr(getattr(config, enh), 'enabled', True)
            click.echo(f"✅ Enabled: {enh}")

    # Save configuration
    # config.save()  # TODO: Implement save method

    click.echo("\nOnion routing enhancements enabled!")
    click.echo("Run 'genomevault onion status' to verify.")

@onion_routing_cli.command(name='disable')
@click.option('--enhancement', multiple=True,
              type=click.Choice(['mix_networks', 'threshold_crypto', 'layered_zk',
                                 'cover_traffic', 'rendezvous', 'metadata_dp',
                                 'pir_tor', 'garlic_routing', 'all']),
              help='Disable specific enhancement(s)')
def disable(enhancement):
    """Disable onion routing enhancements."""
    config = OnionRoutingConfig.load()

    if 'all' in enhancement:
        config.enabled = False
        click.echo("❌ Disabled all onion routing enhancements")
    else:
        for enh in enhancement:
            setattr(getattr(config, enh), 'enabled', False)
            click.echo(f"❌ Disabled: {enh}")

    # Save configuration
    # config.save()

    click.echo("\nRun 'genomevault onion status' to verify.")

@onion_routing_cli.command(name='benchmark')
@click.option('--enhancement',
              type=click.Choice(['mix_networks', 'threshold_crypto', 'layered_zk',
                                 'cover_traffic', 'rendezvous', 'metadata_dp',
                                 'pir_tor', 'garlic_routing', 'all']),
              default='all',
              help='Benchmark specific enhancement')
@click.option('--iterations', default=10, help='Number of iterations')
def benchmark(enhancement, iterations):
    """Benchmark onion routing enhancements."""
    from genomevault.onion_routing.integration.metrics_collector import MetricsCollector

    click.echo(f"Benchmarking {enhancement} ({iterations} iterations)...")

    collector = MetricsCollector()
    results = collector.benchmark_enhancement(enhancement, iterations)

    # Display results
    click.echo(f"\n📊 Benchmark Results for {enhancement}:")
    click.echo(f"  Avg latency: {results['avg_latency_ms']:.1f}ms")
    click.echo(f"  P50 latency: {results['p50_latency_ms']:.1f}ms")
    click.echo(f"  P95 latency: {results['p95_latency_ms']:.1f}ms")
    click.echo(f"  P99 latency: {results['p99_latency_ms']:.1f}ms")
    click.echo(f"  Throughput: {results['throughput_qps']:.2f} queries/sec")
    click.echo(f"  Success rate: {results['success_rate']:.1%}")

@onion_routing_cli.command(name='test')
@click.option('--enhancement',
              type=click.Choice(['mix_networks', 'threshold_crypto', 'layered_zk',
                                 'cover_traffic', 'rendezvous', 'metadata_dp',
                                 'pir_tor', 'garlic_routing', 'all']),
              default='all',
              help='Test specific enhancement')
def test(enhancement):
    """Run integration tests for onion routing enhancements."""
    click.echo(f"Testing {enhancement}...")

    # Run pytest for specific enhancement
    import subprocess

    if enhancement == 'all':
        cmd = "pytest tests/onion_routing/ -v"
    else:
        cmd = f"pytest tests/onion_routing/test_{enhancement}.py -v"

    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        click.echo(f"\n✅ All tests passed for {enhancement}")
    else:
        click.echo(f"\n❌ Tests failed for {enhancement}")
        click.echo("See test output above for details.")

@onion_routing_cli.command(name='metrics')
@click.option('--live', is_flag=True, help='Show live metrics (updates every 5 seconds)')
@click.option('--duration', default=60, help='How long to show live metrics (seconds)')
def metrics(live, duration):
    """Show privacy and performance metrics."""
    from genomevault.onion_routing.integration.metrics_collector import MetricsCollector
    import time

    collector = MetricsCollector()

    if live:
        click.echo(f"Showing live metrics for {duration} seconds (Ctrl+C to stop)...\n")

        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                metrics = collector.get_current_metrics()

                # Clear screen and display
                click.clear()
                click.echo("🔒 Onion Routing Live Metrics")
                click.echo("=" * 50)
                click.echo(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

                click.echo("Privacy Metrics:")
                click.echo(f"  Total entropy: {metrics['total_entropy_bits']:.1f} bits")
                click.echo(f"  Anonymity set size: {metrics['anonymity_set_size']}")
                click.echo(f"  Query unlinkability: {metrics['query_unlinkability']:.1%}\n")

                click.echo("Performance Metrics:")
                click.echo(f"  Queries/sec: {metrics['queries_per_second']:.2f}")
                click.echo(f"  Avg latency: {metrics['avg_latency_ms']:.1f}ms")
                click.echo(f"  Bandwidth usage: {metrics['bandwidth_mbps']:.2f} Mbps\n")

                click.echo("Active Enhancements:")
                for enh in metrics['active_enhancements']:
                    click.echo(f"  ✅ {enh}")

                time.sleep(5)
        except KeyboardInterrupt:
            click.echo("\n\nStopped live metrics.")
    else:
        # Show snapshot
        metrics = collector.get_current_metrics()
        click.echo("📊 Current Metrics Snapshot")
        click.echo("=" * 50)
        # ... (similar display)
```

### Integration with Main CLI

**Update:** `genomevault/cli/__init__.py`

```python
from genomevault.cli.onion_routing_cli import onion_routing_cli

# Register onion routing commands
cli.add_command(onion_routing_cli)
```

### Usage Examples

```bash
# Check status
genomevault onion status

# Enable with preset
genomevault onion enable --preset enhanced_privacy

# Enable specific enhancements
genomevault onion enable --enhancement mix_networks --enhancement cover_traffic

# Disable all
genomevault onion disable --enhancement all

# Benchmark
genomevault onion benchmark --enhancement mix_networks --iterations 100

# Run tests
genomevault onion test --enhancement layered_zk

# Show live metrics
genomevault onion metrics --live --duration 300
```

---

## API Integration

### New REST API Endpoints

**File:** `genomevault/api/routers/onion_routing.py`

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from genomevault.onion_routing.config import OnionRoutingConfig
from genomevault.onion_routing.integration.enhancement_manager import EnhancementManager

router = APIRouter(prefix="/onion-routing", tags=["onion-routing"])

class EnhancementStatus(BaseModel):
    enhancement: str
    enabled: bool
    description: str
    overhead_ms: float

class OnionRoutingStatusResponse(BaseModel):
    enabled: bool
    active_enhancements: List[str]
    enhancement_details: List[EnhancementStatus]
    estimated_overhead: dict

class EnableEnhancementRequest(BaseModel):
    preset: Optional[str] = None
    enhancements: Optional[List[str]] = None

@router.get("/status", response_model=OnionRoutingStatusResponse)
async def get_status():
    """Get status of onion routing enhancements."""
    config = OnionRoutingConfig.load()

    if not config.enabled:
        return OnionRoutingStatusResponse(
            enabled=False,
            active_enhancements=[],
            enhancement_details=[],
            estimated_overhead={}
        )

    active = config.get_active_enhancements()

    # Build enhancement details
    enhancement_info = {
        'mix_networks': ('Mix Networks', 'Batching and shuffling for timing unlinkability'),
        'threshold_crypto': ('Threshold Cryptography', 'Distributed storage across institutions'),
        'layered_zk': ('Layered ZK', 'Verify all 4 pipeline layers independently'),
        'cover_traffic': ('Cover Traffic', 'Dummy queries to hide patterns'),
        'rendezvous': ('Rendezvous Protocol', 'Indirect addressing via relay'),
        'metadata_dp': ('Metadata DP', 'Differential privacy on query metadata'),
        'pir_tor': ('PIR-Tor', 'Route PIR queries through Tor network'),
        'garlic_routing': ('Garlic Routing', 'Bundle multiple queries per packet')
    }

    details = []
    for key, (name, desc) in enhancement_info.items():
        details.append(EnhancementStatus(
            enhancement=key,
            enabled=(key in active),
            description=desc,
            overhead_ms=0  # TODO: Calculate per-enhancement overhead
        ))

    return OnionRoutingStatusResponse(
        enabled=True,
        active_enhancements=active,
        enhancement_details=details,
        estimated_overhead=config.estimate_overhead()
    )

@router.post("/enable")
async def enable_enhancements(request: EnableEnhancementRequest):
    """Enable onion routing enhancements."""
    config = OnionRoutingConfig.load()
    config.enabled = True

    if request.preset:
        config.apply_preset(request.preset)

    if request.enhancements:
        for enh in request.enhancements:
            if hasattr(config, enh):
                setattr(getattr(config, enh), 'enabled', True)

    # Save configuration (TODO: Implement)
    # config.save()

    return {
        "message": "Onion routing enhancements enabled",
        "active_enhancements": config.get_active_enhancements()
    }

@router.post("/disable")
async def disable_enhancements(enhancements: Optional[List[str]] = None):
    """Disable onion routing enhancements."""
    config = OnionRoutingConfig.load()

    if enhancements is None or 'all' in enhancements:
        config.enabled = False
    else:
        for enh in enhancements:
            if hasattr(config, enh):
                setattr(getattr(config, enh), 'enabled', False)

    # Save configuration
    # config.save()

    return {
        "message": "Onion routing enhancements disabled",
        "active_enhancements": config.get_active_enhancements()
    }

@router.get("/metrics")
async def get_metrics():
    """Get current privacy and performance metrics."""
    from genomevault.onion_routing.integration.metrics_collector import MetricsCollector

    collector = MetricsCollector()
    metrics = collector.get_current_metrics()

    return metrics

@router.post("/benchmark/{enhancement}")
async def benchmark_enhancement(
    enhancement: str,
    background_tasks: BackgroundTasks,
    iterations: int = 10
):
    """Run benchmark for specific enhancement."""
    from genomevault.onion_routing.integration.metrics_collector import MetricsCollector

    collector = MetricsCollector()

    # Run benchmark in background
    background_tasks.add_task(
        collector.benchmark_enhancement,
        enhancement,
        iterations
    )

    return {
        "message": f"Benchmark started for {enhancement}",
        "iterations": iterations,
        "status": "running"
    }
```

### API Documentation

Automatic OpenAPI docs will be available at:
- `/api/docs` - Swagger UI
- `/api/redoc` - ReDoc

Example API calls:

```bash
# Get status
curl http://localhost:8000/api/onion-routing/status

# Enable with preset
curl -X POST http://localhost:8000/api/onion-routing/enable \
  -H "Content-Type: application/json" \
  -d '{"preset": "enhanced_privacy"}'

# Enable specific enhancements
curl -X POST http://localhost:8000/api/onion-routing/enable \
  -H "Content-Type: application/json" \
  -d '{"enhancements": ["mix_networks", "cover_traffic"]}'

# Get metrics
curl http://localhost:8000/api/onion-routing/metrics

# Benchmark
curl -X POST http://localhost:8000/api/onion-routing/benchmark/mix_networks?iterations=100
```

---

## Implementation Phases

### Phase 1: Low-Hanging Fruit (4 weeks)

**Target Date:** Week of November 18, 2025

**Enhancements:**
1. ✅ Metadata Differential Privacy
2. ✅ Cover Traffic
3. ✅ Garlic Routing

**Week 1-2: Metadata DP + Cover Traffic**
```
Tasks:
- [ ] Implement MetadataPrivacyLayer class
- [ ] Add timing obfuscation (Laplace noise)
- [ ] Add size padding (fixed + random)
- [ ] Add frequency jitter
- [ ] Implement CoverTrafficGenerator
- [ ] Add background dummy query scheduler
- [ ] Create unit tests (target: 90% coverage)
- [ ] Benchmark overhead (< 10ms expected)
- [ ] Document configuration options
```

**Week 3-4: Garlic Routing**
```
Tasks:
- [ ] Implement GarlicQueryBundler
- [ ] Add bundling strategies (opportunistic, forced)
- [ ] Implement unbundler on server side
- [ ] Add anti-traffic-analysis padding
- [ ] Create integration tests
- [ ] Benchmark bandwidth savings (expect ~20% reduction)
- [ ] Update API with bundling endpoint
- [ ] Add CLI commands for configuration
```

**Deliverables:**
- Working implementations of 3 enhancements
- Test suite with 90% coverage
- Performance benchmarks
- User documentation
- API endpoints
- CLI commands

**Success Metrics:**
- +3 bits entropy gain
- < 10ms latency increase
- Bandwidth neutral (cover traffic overhead offset by garlic bundling)
- All tests passing

---

### Phase 2: Core Infrastructure (3 months)

**Target Date:** Week of January 20, 2026

**Enhancements:**
4. ✅ Mix Networks
5. ✅ Rendezvous Protocol
6. ✅ PIR-Tor Integration

**Month 1 (Weeks 5-8): Mix Networks**
```
Tasks:
- [ ] Implement MixNetworkQueryProcessor
- [ ] Add batching logic with adaptive sizing
- [ ] Implement shuffling algorithm (Fisher-Yates)
- [ ] Create TokenManager for unlinkable returns
- [ ] Add result channel infrastructure
- [ ] Implement timeout handling
- [ ] Create stress tests (1000+ concurrent queries)
- [ ] Benchmark at scale (batch sizes 10-1000)
- [ ] Document trade-offs (latency vs privacy)
```

**Month 2 (Weeks 9-12): Rendezvous Protocol**
```
Tasks:
- [ ] Implement RendezvousProtocol class
- [ ] Add client-side protocol (token-based)
- [ ] Implement rendezvous node (routing layer)
- [ ] Add server-side protocol (blind processing)
- [ ] Create multi-party crypto (nested encryption)
- [ ] Implement result delivery channel
- [ ] Security audit (verify unlinkability)
- [ ] Deploy test rendezvous nodes
- [ ] Load testing (1000 qps)
```

**Month 3 (Weeks 13-16): PIR-Tor Integration**
```
Tasks:
- [ ] Integrate Tor control library (stem)
- [ ] Implement OnionPIRClient
- [ ] Add circuit management (fresh per query)
- [ ] Handle Tor failures gracefully
- [ ] Create fallback mechanism (non-Tor PIR)
- [ ] Benchmark Tor overhead (~500ms expected)
- [ ] Document Tor setup for users
- [ ] Add optional entry node selection
```

**Deliverables:**
- Working implementations of 3 additional enhancements
- Comprehensive test suite
- Performance benchmarks at scale
- Security audit report
- User documentation
- Deployment guide

**Success Metrics:**
- +6.6 bits additional entropy (total +9.6 bits)
- < 600ms latency increase (including Tor)
- 100% test coverage for critical paths
- Security audit passed

---

### Phase 3: Advanced Security (6 months)

**Target Date:** Week of July 20, 2026

**Enhancements:**
7. ✅ Threshold Cryptography
8. ✅ Layered ZK Verification

**Months 4-5 (Weeks 17-24): Threshold Cryptography**
```
Tasks:
- [ ] Implement Shamir secret sharing
- [ ] Create ShardManager for distribution
- [ ] Design InstitutionalNode interface
- [ ] Build multi-institutional API
- [ ] Implement parallel reconstruction
- [ ] Add fault tolerance (n-k failures)
- [ ] Security proof (information-theoretic)
- [ ] Partner with 5 institutions (academic/clinical)
- [ ] Deploy distributed infrastructure
- [ ] Real-world testing with partners
```

**Month 6 (Weeks 25-28): Layered ZK Verification**
```
Tasks:
- [ ] Design 4 ZK circuits (one per layer)
  - consensus_correctness.circom
  - pooling_correctness.circom
  - query_correctness.circom
  - encoding_correctness.circom
- [ ] Implement LayerVerifier class
- [ ] Create commitment chain
- [ ] Generate trusted setups (universal SRS)
- [ ] Implement parallel proof generation
- [ ] Benchmark proof generation time (<3s total)
- [ ] Create audit API for external verification
- [ ] Document layered verification protocol
```

**Deliverables:**
- Complete implementation of all 8 enhancements
- Production-ready distributed system
- Formal security proofs
- Multi-institutional deployment
- External audit capability
- Complete documentation suite

**Success Metrics:**
- Distributed trust across 5 institutions
- Complete audit trail (4 ZK proofs per query)
- 12× harder to attack (threshold security)
- 99.9% availability (fault tolerance)
- All 8 enhancements working together

---

## Testing Strategy

### Unit Tests

**Directory:** `tests/onion_routing/`

```
tests/onion_routing/
├── test_mix_networks.py
├── test_threshold_crypto.py
├── test_layered_zk.py
├── test_cover_traffic.py
├── test_rendezvous.py
├── test_metadata_dp.py
├── test_pir_tor.py
├── test_garlic_routing.py
└── test_integration.py
```

**Example Test:** `tests/onion_routing/test_mix_networks.py`

```python
import pytest
import asyncio
from genomevault.onion_routing.mix_networks import MixNetworkQueryProcessor

@pytest.mark.asyncio
async def test_batching():
    """Test that queries are batched correctly."""
    processor = MixNetworkQueryProcessor(batch_size=10, delay_window=5)

    # Submit 10 queries
    tasks = []
    for i in range(10):
        task = asyncio.create_task(processor.submit_query(f"query_{i}", f"user_{i}"))
        tasks.append(task)

    # All should complete when batch is full
    results = await asyncio.gather(*tasks, timeout=10)

    assert len(results) == 10
    assert all(r is not None for r in results)

@pytest.mark.asyncio
async def test_shuffling():
    """Test that queries are shuffled (not in submission order)."""
    processor = MixNetworkQueryProcessor(batch_size=100, delay_window=1)

    query_order = []
    async def submit_and_record(query_id):
        await processor.submit_query(f"query_{query_id}", f"user_{query_id}")
        query_order.append(query_id)

    # Submit 100 queries
    tasks = [asyncio.create_task(submit_and_record(i)) for i in range(100)]
    await asyncio.gather(*tasks)

    # Order should not match submission order (with high probability)
    assert query_order != list(range(100))

@pytest.mark.asyncio
async def test_anonymity_set():
    """Test that anonymity set size matches batch size."""
    processor = MixNetworkQueryProcessor(batch_size=50, delay_window=2)

    # Submit 50 queries from different users
    tasks = [
        asyncio.create_task(processor.submit_query(f"query_{i}", f"user_{i}"))
        for i in range(50)
    ]

    await asyncio.gather(*tasks)

    # Anonymity set should be 50
    stats = processor.get_statistics()
    assert stats['anonymity_set_size'] == 50
    assert stats['entropy_bits'] == pytest.approx(5.64, abs=0.1)  # log2(50)

def test_entropy_calculation():
    """Test entropy gain calculation."""
    processor = MixNetworkQueryProcessor(batch_size=100, delay_window=60)

    entropy = processor.calculate_entropy()

    # log2(100) ≈ 6.64 bits
    assert entropy == pytest.approx(6.64, abs=0.01)
```

### Integration Tests

**File:** `tests/onion_routing/test_integration.py`

```python
import pytest
from genomevault.onion_routing.config import OnionRoutingConfig
from genomevault.onion_routing.integration.enhancement_manager import EnhancementManager

def test_full_pipeline_with_enhancements():
    """Test complete GenomeVault pipeline with all enhancements enabled."""
    # Enable all enhancements
    config = OnionRoutingConfig.load()
    config.apply_preset('maximum_privacy')

    manager = EnhancementManager(config)

    # Run query through enhanced pipeline
    query = "test_query"
    result = manager.execute_query(query)

    # Verify result correctness
    assert result is not None

    # Verify all enhancements were applied
    applied = manager.get_applied_enhancements()
    assert len(applied) == 8

    # Verify privacy properties
    privacy_metrics = manager.get_privacy_metrics()
    assert privacy_metrics['total_entropy_bits'] >= 270  # 261 + 9.6

def test_enhancement_composition():
    """Test that enhancements compose correctly."""
    config = OnionRoutingConfig.load()

    # Enable subset of enhancements
    config.mix_networks.enabled = True
    config.cover_traffic.enabled = True
    config.metadata_dp.enabled = True

    manager = EnhancementManager(config)

    # Verify they work together
    for i in range(10):
        result = manager.execute_query(f"query_{i}")
        assert result is not None

@pytest.mark.slow
def test_performance_overhead():
    """Benchmark performance overhead of all enhancements."""
    import time

    # Baseline: no enhancements
    config_baseline = OnionRoutingConfig.load()
    config_baseline.enabled = False
    manager_baseline = EnhancementManager(config_baseline)

    start = time.time()
    for i in range(100):
        manager_baseline.execute_query(f"query_{i}")
    baseline_time = time.time() - start

    # With enhancements
    config_enhanced = OnionRoutingConfig.load()
    config_enhanced.apply_preset('enhanced_privacy')
    manager_enhanced = EnhancementManager(config_enhanced)

    start = time.time()
    for i in range(100):
        manager_enhanced.execute_query(f"query_{i}")
    enhanced_time = time.time() - start

    overhead = (enhanced_time - baseline_time) / baseline_time

    # Overhead should be < 2× for enhanced_privacy preset
    assert overhead < 2.0
```

### Performance Tests

```python
@pytest.mark.benchmark
def test_mix_networks_throughput(benchmark):
    """Benchmark mix networks throughput."""
    processor = MixNetworkQueryProcessor(batch_size=100, delay_window=1)

    async def run_batch():
        tasks = [
            asyncio.create_task(processor.submit_query(f"q_{i}", f"u_{i}"))
            for i in range(100)
        ]
        await asyncio.gather(*tasks)

    # Benchmark
    result = benchmark(asyncio.run, run_batch())

    # Should process 100 queries in < 5 seconds
    assert result < 5.0

@pytest.mark.benchmark
def test_layered_zk_proof_generation(benchmark):
    """Benchmark layered ZK proof generation time."""
    from genomevault.onion_routing.layered_zk import LayeredVerificationProtocol

    protocol = LayeredVerificationProtocol()

    def generate_all_proofs():
        proofs = {}
        proofs['consensus'] = protocol.generate_layer1_proof()
        proofs['pooling'] = protocol.generate_layer2_proof()
        proofs['query'] = protocol.generate_layer3_proof()
        proofs['encoding'] = protocol.generate_layer4_proof()
        return proofs

    # Benchmark
    result = benchmark(generate_all_proofs)

    # Should generate 4 proofs in < 3 seconds
    assert benchmark.stats.mean < 3.0
```

---

## Performance Benchmarking

### Benchmarking Plan

**File:** `benchmarks/onion_routing_benchmark.py`

```python
#!/usr/bin/env python3
"""
Comprehensive benchmark for onion routing enhancements.
"""

import time
import asyncio
import numpy as np
from typing import Dict, List
from genomevault.onion_routing.config import OnionRoutingConfig
from genomevault.onion_routing.integration.enhancement_manager import EnhancementManager

class OnionRoutingBenchmark:
    def __init__(self):
        self.results = {}

    def benchmark_enhancement(
        self,
        enhancement_name: str,
        iterations: int = 100
    ) -> Dict:
        """Benchmark a single enhancement."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {enhancement_name}")
        print(f"Iterations: {iterations}")
        print(f"{'='*60}")

        # Setup configuration
        config = OnionRoutingConfig.load()
        config.enabled = True
        setattr(getattr(config, enhancement_name), 'enabled', True)

        manager = EnhancementManager(config)

        # Warm-up
        for i in range(10):
            manager.execute_query(f"warmup_{i}")

        # Benchmark
        latencies = []
        bandwidths = []
        cpu_times = []

        for i in range(iterations):
            start = time.time()
            start_cpu = time.process_time()

            result = manager.execute_query(f"query_{i}")

            latency = (time.time() - start) * 1000  # ms
            cpu_time = (time.process_time() - start_cpu) * 1000  # ms

            latencies.append(latency)
            cpu_times.append(cpu_time)

            # Measure bandwidth (if applicable)
            if hasattr(result, 'bytes_transferred'):
                bandwidths.append(result.bytes_transferred)

        # Calculate statistics
        results = {
            'enhancement': enhancement_name,
            'iterations': iterations,
            'latency': {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies)
            },
            'cpu_time': {
                'mean': np.mean(cpu_times),
                'std': np.std(cpu_times)
            },
            'throughput': {
                'qps': iterations / (sum(latencies) / 1000)
            }
        }

        if bandwidths:
            results['bandwidth'] = {
                'mean_bytes': np.mean(bandwidths),
                'total_bytes': sum(bandwidths)
            }

        # Display results
        print(f"\n📊 Results:")
        print(f"  Latency (ms):")
        print(f"    Mean:   {results['latency']['mean']:.2f}")
        print(f"    Median: {results['latency']['median']:.2f}")
        print(f"    P50:    {results['latency']['p50']:.2f}")
        print(f"    P95:    {results['latency']['p95']:.2f}")
        print(f"    P99:    {results['latency']['p99']:.2f}")
        print(f"    Std:    {results['latency']['std']:.2f}")
        print(f"    Range:  [{results['latency']['min']:.2f}, {results['latency']['max']:.2f}]")
        print(f"\n  CPU Time (ms):")
        print(f"    Mean:   {results['cpu_time']['mean']:.2f}")
        print(f"    Std:    {results['cpu_time']['std']:.2f}")
        print(f"\n  Throughput:")
        print(f"    QPS:    {results['throughput']['qps']:.2f} queries/sec")

        self.results[enhancement_name] = results
        return results

    def benchmark_all(self, iterations: int = 100):
        """Benchmark all enhancements."""
        enhancements = [
            'mix_networks',
            'threshold_crypto',
            'layered_zk',
            'cover_traffic',
            'rendezvous',
            'metadata_dp',
            'pir_tor',
            'garlic_routing'
        ]

        for enhancement in enhancements:
            self.benchmark_enhancement(enhancement, iterations)

        # Comparison table
        self.print_comparison_table()

    def print_comparison_table(self):
        """Print comparison table of all enhancements."""
        print(f"\n\n{'='*80}")
        print("COMPARISON TABLE: All Enhancements")
        print(f"{'='*80}\n")

        print(f"{'Enhancement':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'QPS':<12} {'Overhead':<12}")
        print(f"{'-'*80}")

        # Baseline (no enhancements)
        baseline_latency = 20.0  # Assume baseline latency

        for name, results in self.results.items():
            mean = results['latency']['mean']
            p95 = results['latency']['p95']
            qps = results['throughput']['qps']
            overhead = (mean - baseline_latency) / baseline_latency * 100

            print(f"{name:<20} {mean:<12.2f} {p95:<12.2f} {qps:<12.2f} {overhead:<12.1f}%")

    def export_results(self, filename: str = "onion_routing_benchmark_results.json"):
        """Export results to JSON file."""
        import json

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✅ Results exported to: {filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark onion routing enhancements")
    parser.add_argument('--enhancement', type=str, default='all',
                        help='Enhancement to benchmark (default: all)')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations (default: 100)')
    parser.add_argument('--export', type=str, default=None,
                        help='Export results to JSON file')

    args = parser.parse_args()

    benchmark = OnionRoutingBenchmark()

    if args.enhancement == 'all':
        benchmark.benchmark_all(args.iterations)
    else:
        benchmark.benchmark_enhancement(args.enhancement, args.iterations)

    if args.export:
        benchmark.export_results(args.export)
```

### Usage

```bash
# Benchmark all enhancements
python benchmarks/onion_routing_benchmark.py --enhancement all --iterations 1000

# Benchmark specific enhancement
python benchmarks/onion_routing_benchmark.py --enhancement mix_networks --iterations 500

# Export results
python benchmarks/onion_routing_benchmark.py --enhancement all --iterations 1000 --export results.json
```

---

## Documentation Plan

### Documentation Structure

```
docs/
├── guides/
│   ├── onion_routing_enhancements.md        # Original research doc
│   ├── ONION_ROUTING_IMPLEMENTATION_PLAN.md # This document
│   ├── onion_routing_user_guide.md          # User-facing guide
│   └── onion_routing_developer_guide.md     # Developer reference
│
├── api-docs/
│   └── onion_routing_api.md                 # API documentation
│
└── reports/
    ├── onion_routing_benchmark_results.md   # Benchmark report
    └── onion_routing_security_audit.md      # Security audit report
```

### User Guide Outline

**File:** `docs/guides/onion_routing_user_guide.md`

```markdown
# Onion Routing Enhancements - User Guide

## Introduction
- What are onion routing enhancements?
- Why should I use them?
- Trade-offs: Privacy vs Performance

## Quick Start
- Enabling enhancements (presets)
- Checking status
- Basic usage

## Configuration
- Configuration file format
- Available presets
- Tuning parameters

## Enhancements Explained
- Mix Networks (timing unlinkability)
- Threshold Cryptography (distributed trust)
- Layered ZK (complete audit trail)
- Cover Traffic (pattern hiding)
- Rendezvous Protocol (network anonymity)
- Metadata DP (metadata protection)
- PIR-Tor Integration (ISP-level anonymity)
- Garlic Routing (query bundling)

## Performance Tuning
- Latency vs privacy trade-offs
- Bandwidth considerations
- CPU requirements

## Troubleshooting
- Common issues
- Debugging tips
- Support resources

## FAQ
- Q&A on common questions
```

### Developer Guide Outline

**File:** `docs/guides/onion_routing_developer_guide.md`

```markdown
# Onion Routing Enhancements - Developer Guide

## Architecture
- Module structure
- Component interactions
- Data flow diagrams

## Implementation Details
- Mix Networks implementation
- Threshold Cryptography implementation
- ... (all 8 enhancements)

## API Reference
- Configuration API
- Enhancement Manager API
- Metrics Collector API

## Testing
- Unit testing strategy
- Integration testing
- Performance testing

## Contributing
- Code style
- Pull request process
- Testing requirements

## Appendices
- Security proofs
- Performance benchmarks
- Research references
```

---

## Migration Path

### For Existing Users

**Goal:** Seamless transition with zero disruption

**Strategy:**
1. All enhancements are **opt-in** (disabled by default)
2. Existing pipelines work without any changes
3. Users can enable enhancements incrementally
4. Rollback mechanism available

**Migration Steps:**

```bash
# Step 1: Update GenomeVault
pip install --upgrade genomevault

# Step 2: Check new features
genomevault onion status
# Output: ❌ Onion routing enhancements: DISABLED

# Step 3: Test with a single enhancement
genomevault onion enable --enhancement metadata_dp
genomevault onion test --enhancement metadata_dp

# Step 4: Run existing pipeline (verify backward compatibility)
genomevault run --config existing_config.yaml

# Step 5: Enable more enhancements gradually
genomevault onion enable --preset enhanced_privacy

# Step 6: Monitor performance
genomevault onion metrics --live --duration 300

# Step 7: Full deployment
genomevault onion enable --preset maximum_privacy
```

### Rollback Procedure

```bash
# Disable all enhancements
genomevault onion disable --enhancement all

# Verify rollback
genomevault onion status
# Output: ❌ Onion routing enhancements: DISABLED

# System reverts to standard behavior
```

---

## Summary

### Deliverables by Phase

| Phase | Duration | Enhancements | Deliverables |
|-------|----------|--------------|--------------|
| **Phase 1** | 4 weeks | Metadata DP, Cover Traffic, Garlic | Code, tests, docs, benchmarks |
| **Phase 2** | 3 months | Mix Networks, Rendezvous, PIR-Tor | Code, tests, docs, security audit |
| **Phase 3** | 6 months | Threshold Crypto, Layered ZK | Complete system, multi-institutional deployment |

### Success Metrics

| Metric | Target | Verification Method |
|--------|--------|---------------------|
| Privacy gain | +9.6 bits entropy | Entropy calculation |
| Latency increase | < 600ms | Benchmarking |
| Test coverage | > 90% | pytest-cov |
| Documentation | Complete | User feedback |
| Security audit | Passed | External audit |
| Multi-institutional | 5 partners | Deployment verified |

### Next Steps

1. **Immediate (This Week):**
   - Review this implementation plan
   - Get stakeholder approval
   - Set up development environment

2. **Week 1 (Starting Oct 28):**
   - Create module structure
   - Implement configuration system
   - Start Phase 1 development (Metadata DP)

3. **Ongoing:**
   - Weekly progress reviews
   - Continuous integration testing
   - Documentation updates

---

**End of Implementation Plan**
