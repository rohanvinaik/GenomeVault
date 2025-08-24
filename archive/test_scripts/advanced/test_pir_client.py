#!/usr/bin/env python3
"""
Demo script for testing PIR client with index 42.

This script:
1. Starts PIR servers in the background
2. Creates a PIR client
3. Queries index 42 privately
4. Verifies the retrieved data
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import directly to avoid circular import issues
import sys

sys.path.insert(0, str(project_root / "genomevault" / "pir" / "client"))
from pir_client import PIRClient, ServerConfig, QueryProtocol
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


async def wait_for_servers(urls, timeout=30):
    """Wait for servers to be ready."""
    import aiohttp

    start_time = time.time()
    ready = False

    while time.time() - start_time < timeout:
        all_ready = True

        async with aiohttp.ClientSession() as session:
            for url in urls:
                try:
                    async with session.get(
                        f"{url}/health", timeout=aiohttp.ClientTimeout(total=2.0)
                    ) as response:
                        if response.status != 200:
                            all_ready = False
                            break
                except:
                    all_ready = False
                    break

        if all_ready:
            ready = True
            break

        await asyncio.sleep(1)

    return ready


async def main():
    """Main demo function."""
    print("\n" + "=" * 60)
    print("PIR CLIENT DEMO - Private Retrieval of Index 42")
    print("=" * 60)

    # Configuration
    num_servers = 3
    dataset_size = 100
    target_index = 42

    print("\nConfiguration:")
    print(f"  Servers: {num_servers}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Target index: {target_index}")
    print()

    # Start PIR servers
    print("Starting PIR servers...")
    server_process = subprocess.Popen(
        [sys.executable, "scripts/run_pir_servers.py", str(num_servers), str(dataset_size)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for servers to start
    server_urls = [f"http://127.0.0.1:{9001 + i}" for i in range(num_servers)]

    print("Waiting for servers to be ready...")
    servers_ready = await wait_for_servers(server_urls, timeout=15)

    if not servers_ready:
        print("ERROR: Servers failed to start")
        server_process.terminate()
        return 1

    print("✓ Servers are ready")

    try:
        # Configure PIR client
        servers = [
            ServerConfig(url=url, server_id=i, timeout_seconds=10.0)
            for i, url in enumerate(server_urls)
        ]

        # Create PIR client
        async with PIRClient(
            servers=servers,
            database_size=dataset_size,
            element_size=1024,
            min_servers=2,
            protocol=QueryProtocol.IT_PIR,
        ) as client:
            print("\n" + "-" * 60)
            print("PRIVACY-PRESERVING QUERY")
            print("-" * 60)

            # Check server status
            print("\n1. Checking server status...")
            statuses = await client.get_server_status()
            for server_id, status in statuses.items():
                if status["online"]:
                    print(f"   Server {server_id}: ✓ Online")
                else:
                    print(f"   Server {server_id}: ✗ Offline - {status.get('error', 'Unknown')}")

            # Perform privacy-preserving query
            print(f"\n2. Querying index {target_index} privately...")
            print("   Generating IT-PIR query vectors...")
            print("   Sending queries to all servers...")

            start_time = time.time()
            retrieved_data = await client.retrieve(target_index)
            query_time = (time.time() - start_time) * 1000

            print(f"   ✓ Query completed in {query_time:.2f}ms")

            # Parse retrieved data
            print("\n3. Verifying retrieved data...")

            try:
                # Try to parse as JSON (genomic record)
                decoded = json.loads(retrieved_data.decode("utf-8"))

                print(f"   Retrieved record at index {target_index}:")
                print(f"     Variant ID: {decoded.get('variant_id', 'unknown')}")
                print(f"     Chromosome: {decoded.get('chrom', 'unknown')}")
                print(f"     Position: {decoded.get('pos', 'unknown')}")
                print(f"     Reference: {decoded.get('ref', 'unknown')}")
                print(f"     Alternate: {decoded.get('alt', 'unknown')}")
                print(f"     Index: {decoded.get('index', 'unknown')}")

                # Verify index matches
                if decoded.get("index") == target_index:
                    print("\n   ✓ Index verification: PASSED")
                else:
                    print("\n   ✗ Index verification: FAILED")
                    print(f"     Expected: {target_index}, Got: {decoded.get('index')}")

            except json.JSONDecodeError:
                print(f"   Raw data (first 100 bytes): {retrieved_data[:100]}")

            # Get client statistics
            print("\n4. Client statistics:")
            stats = client.get_statistics()
            print(f"   Queries sent: {stats['queries_sent']}")
            print(f"   Queries successful: {stats['queries_successful']}")
            print(f"   Success rate: {stats['success_rate']:.1%}")
            print(f"   Average latency: {stats['avg_latency_ms']:.2f}ms")
            print(f"   Bytes received: {stats['bytes_received']}")
            print(f"   Protocol: {stats['protocol']}")

            # Test batch retrieval
            print("\n5. Testing batch retrieval...")
            batch_indices = [0, 10, 20, 30, 40, 50]
            print(f"   Retrieving indices: {batch_indices}")

            batch_start = time.time()
            batch_results = await client.batch_retrieve(batch_indices)
            batch_time = (time.time() - batch_start) * 1000

            print(f"   ✓ Batch query completed in {batch_time:.2f}ms")

            successful_retrievals = sum(1 for r in batch_results if r)
            print(f"   Successfully retrieved: {successful_retrievals}/{len(batch_indices)}")

            # Display first few results
            for i, (idx, data) in enumerate(zip(batch_indices[:3], batch_results[:3])):
                if data:
                    try:
                        decoded = json.loads(data.decode("utf-8"))
                        print(f"     Index {idx}: {decoded.get('variant_id', 'unknown')}")
                    except:
                        print(f"     Index {idx}: Retrieved {len(data)} bytes")

            print("\n" + "=" * 60)
            print("PRIVACY GUARANTEES")
            print("=" * 60)
            print("\nThe PIR protocol ensures:")
            print("✓ Servers cannot determine which index was queried")
            print("✓ Query vectors appear random to each server")
            print("✓ Only the XOR aggregation reveals the target data")
            print("✓ Information-theoretic security (no computational assumptions)")

            print("\n" + "=" * 60)
            print("✅ PIR CLIENT DEMO SUCCESSFUL")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Stop servers
        print("\nStopping PIR servers...")
        server_process.terminate()
        server_process.wait(timeout=5)
        print("✓ Servers stopped")

    return 0


async def test_protocols():
    """Test different PIR protocols."""
    print("\n" + "=" * 60)
    print("PROTOCOL COMPARISON TEST")
    print("=" * 60)

    # Start servers (simplified for testing)
    num_servers = 2
    dataset_size = 50
    target_index = 25

    server_urls = [f"http://127.0.0.1:{9001 + i}" for i in range(num_servers)]
    servers = [ServerConfig(url=url, server_id=i) for i, url in enumerate(server_urls)]

    # Note: Servers should be running before this test

    async with PIRClient(servers=servers, database_size=dataset_size, min_servers=2) as client:
        # Test IT-PIR
        print("\n1. Testing IT-PIR protocol...")
        try:
            start = time.time()
            data_itpir = await client.retrieve(target_index, protocol=QueryProtocol.IT_PIR)
            time_itpir = (time.time() - start) * 1000
            print(f"   ✓ IT-PIR: {time_itpir:.2f}ms, {len(data_itpir)} bytes")
        except Exception as e:
            print(f"   ✗ IT-PIR failed: {e}")

        # Test XOR
        print("\n2. Testing XOR protocol...")
        try:
            start = time.time()
            data_xor = await client.retrieve(target_index, protocol=QueryProtocol.XOR)
            time_xor = (time.time() - start) * 1000
            print(f"   ✓ XOR: {time_xor:.2f}ms, {len(data_xor)} bytes")
        except Exception as e:
            print(f"   ✗ XOR failed: {e}")


if __name__ == "__main__":
    print("\n🔐 GenomeVault PIR Client Demo")
    print("Privacy-Preserving Information Retrieval")

    # Run main demo
    result = asyncio.run(main())

    # Optionally run protocol comparison
    # asyncio.run(test_protocols())

    sys.exit(result)
