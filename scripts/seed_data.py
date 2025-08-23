#!/usr/bin/env python3
"""
Seed data script for GenomeVault development and testing.

Creates sample users, organizations, genomic metadata, and audit logs
for development and testing purposes.
"""

import os
import sys
import uuid
import random
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
import numpy as np
from faker import Faker
from passlib.context import CryptContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Faker for generating test data
fake = Faker()
Faker.seed(42)  # For reproducible test data

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://genomevault:genomevault@localhost/genomevault"
)


class SeedDataGenerator:
    """Generate seed data for GenomeVault database."""
    
    def __init__(self, conn):
        """Initialize seed data generator."""
        self.conn = conn
        self.organizations = []
        self.users = []
        self.genomic_data = []
        
    async def clear_existing_data(self):
        """Clear existing data from tables (except migrations)."""
        logger.info("Clearing existing data...")
        
        # Tables to clear (in reverse dependency order)
        tables = [
            "differential_privacy_logs",
            "query_history",
            "phi_access_logs",
            "audit_logs",
            "hypervector_storage",
            "genomic_data_metadata",
            "api_keys",
            "sessions",
            "users",
            "organizations",
        ]
        
        for table in tables:
            try:
                # For partitioned tables, delete from parent
                if table in ["audit_logs", "phi_access_logs", "query_history"]:
                    await self.conn.execute(f"DELETE FROM {table} WHERE true;")
                else:
                    await self.conn.execute(f"TRUNCATE TABLE {table} CASCADE;")
                logger.info(f"Cleared table: {table}")
            except Exception as e:
                logger.warning(f"Could not clear {table}: {e}")
    
    async def create_organizations(self, count=5):
        """Create sample organizations."""
        logger.info(f"Creating {count} organizations...")
        
        org_types = [
            ("Mayo Clinic", "mayo.edu", "1234567890", True),
            ("Johns Hopkins Medicine", "jhmi.edu", "0987654321", True),
            ("Stanford Health Care", "stanfordhealth.org", "1122334455", True),
            ("GenomeVault Research", "research.genomevault.io", None, False),
            ("Demo Organization", "demo.genomevault.io", None, False),
        ]
        
        for i, (name, domain, npi, baa) in enumerate(org_types[:count]):
            org_id = str(uuid.uuid4())
            
            baa_signed_at = datetime.now(timezone.utc) - timedelta(days=random.randint(30, 365)) if baa else None
            baa_document_hash = hashlib.sha256(f"BAA-{org_id}".encode()).hexdigest() if baa else None
            
            await self.conn.execute("""
                INSERT INTO organizations (
                    id, name, domain, npi_number, baa_signed,
                    baa_signed_at, baa_document_hash, settings,
                    created_at, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 
                org_id, name, domain, npi, baa,
                baa_signed_at, baa_document_hash,
                {"tier": "enterprise" if baa else "basic", "features": ["pir", "zk", "federated"]},
                datetime.now(timezone.utc) - timedelta(days=random.randint(0, 730)),
                True
            )
            
            self.organizations.append({
                "id": org_id,
                "name": name,
                "domain": domain,
                "baa_signed": baa
            })
            
            logger.info(f"Created organization: {name}")
    
    async def create_users(self, count=20):
        """Create sample users with different roles."""
        logger.info(f"Creating {count} users...")
        
        roles_distribution = {
            "clinician": 0.3,
            "researcher": 0.3,
            "patient": 0.3,
            "admin": 0.1,
        }
        
        for i in range(count):
            user_id = str(uuid.uuid4())
            
            # Determine role based on distribution
            rand = random.random()
            cumulative = 0
            role = "patient"
            for r, prob in roles_distribution.items():
                cumulative += prob
                if rand <= cumulative:
                    role = r
                    break
            
            # Generate user data
            first_name = fake.first_name()
            last_name = fake.last_name()
            username = f"{first_name.lower()}.{last_name.lower()}{i}"
            email = f"{username}@{random.choice(self.organizations)['domain'] if self.organizations else 'genomevault.io'}"
            
            # Assign to organization
            org_id = random.choice(self.organizations)["id"] if self.organizations and random.random() > 0.3 else None
            
            # Role-specific fields
            npi_number = fake.numerify("##########") if role == "clinician" else None
            mfa_enabled = role in ["clinician", "admin"]
            
            # Scopes based on role
            scopes_map = {
                "patient": ["read:genomic", "read:clinical", "pir:query"],
                "clinician": ["read:genomic", "read:clinical", "write:clinical", "read:phi", "write:phi"],
                "researcher": ["read:genomic", "read:clinical", "pir:query", "zk:prove", "federated:participate"],
                "admin": ["admin:users", "admin:system", "admin:all"],
            }
            scopes = scopes_map.get(role, [])
            
            # Rate limit tier
            tier_map = {
                "patient": "basic",
                "clinician": "professional",
                "researcher": "professional",
                "admin": "enterprise",
            }
            rate_limit_tier = tier_map.get(role, "basic")
            
            await self.conn.execute("""
                INSERT INTO users (
                    id, username, email, full_name, hashed_password,
                    organization_id, roles, scopes, is_active, is_verified,
                    npi_number, mfa_enabled, mfa_secret,
                    rate_limit_tier, created_at, last_login_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                user_id, username, email, f"{first_name} {last_name}",
                pwd_context.hash("genomevault123"),  # Default password for all test users
                org_id, [role], scopes, True, True,
                npi_number, mfa_enabled,
                secrets.token_urlsafe(32) if mfa_enabled else None,
                rate_limit_tier,
                datetime.now(timezone.utc) - timedelta(days=random.randint(0, 365)),
                datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
            )
            
            self.users.append({
                "id": user_id,
                "username": username,
                "email": email,
                "role": role,
                "org_id": org_id
            })
            
            logger.info(f"Created user: {username} ({role})")
    
    async def create_api_keys(self, count=10):
        """Create API keys for some users."""
        logger.info(f"Creating {count} API keys...")
        
        # Select users who should have API keys (researchers and services)
        api_key_users = [u for u in self.users if u["role"] in ["researcher", "admin"]][:count]
        
        for user in api_key_users:
            key_id = f"key_{secrets.token_hex(8)}"
            key_value = f"gv_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(key_value.encode()).hexdigest()
            
            await self.conn.execute("""
                INSERT INTO api_keys (
                    id, key_id, key_hash, name, description,
                    user_id, scopes, expires_at, created_at,
                    last_rotated_at, rotation_interval_days, is_active
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                str(uuid.uuid4()), key_id, key_hash,
                f"API Key for {user['username']}", f"Development API key for testing",
                user["id"],
                ["read:genomic", "pir:query", "zk:prove"] if user["role"] == "researcher" else ["admin:all"],
                datetime.now(timezone.utc) + timedelta(days=365),
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
                90,  # Rotate every 90 days
                True
            )
            
            logger.info(f"Created API key for user: {user['username']} (Key: {key_value[:20]}...)")
    
    async def create_genomic_metadata(self, count=50):
        """Create genomic data metadata entries."""
        logger.info(f"Creating {count} genomic metadata entries...")
        
        data_types = ["vcf", "fastq", "bam", "cram", "bed"]
        classifications = ["public", "internal", "confidential", "phi", "restricted"]
        reference_genomes = ["GRCh38", "GRCh37", "hg19", "hg38"]
        platforms = ["Illumina NovaSeq", "Illumina HiSeq", "PacBio Sequel", "Oxford Nanopore", "BGI"]
        
        for i in range(count):
            data_id = str(uuid.uuid4())
            owner = random.choice(self.users)
            
            # Generate realistic genomic metadata
            data_type = random.choice(data_types)
            classification = random.choice(classifications)
            
            # PHI data should only belong to patients
            if classification == "phi" and owner["role"] != "patient":
                classification = "confidential"
            
            data_hash = hashlib.sha256(f"genomic-data-{i}".encode()).hexdigest()
            
            await self.conn.execute("""
                INSERT INTO genomic_data_metadata (
                    id, owner_id, organization_id, data_hash, data_type,
                    data_classification, sample_id, sequencing_date,
                    sequencing_platform, reference_genome, variant_count,
                    coverage_mean, quality_score, consent_id,
                    differential_privacy_epsilon, retention_tier,
                    created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            """,
                data_id, owner["id"], owner.get("org_id"),
                data_hash, data_type, classification,
                f"SAMPLE-{fake.uuid4()[:8].upper()}",
                fake.date_between(start_date="-2y", end_date="today"),
                random.choice(platforms),
                random.choice(reference_genomes),
                random.randint(10000, 5000000) if data_type == "vcf" else None,
                random.uniform(10, 100) if data_type in ["bam", "cram"] else None,
                random.uniform(20, 40),
                f"CONSENT-{fake.uuid4()[:8].upper()}" if classification in ["phi", "restricted"] else None,
                random.uniform(0.1, 1.0) if classification != "public" else None,
                random.choice(["hot", "warm", "cold"]),
                datetime.now(timezone.utc) - timedelta(days=random.randint(0, 730))
            )
            
            self.genomic_data.append({
                "id": data_id,
                "owner_id": owner["id"],
                "data_type": data_type,
                "classification": classification
            })
            
            # Create hypervector storage for some entries
            if random.random() > 0.5:
                await self.create_hypervector_storage(data_id)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Created {i + 1} genomic metadata entries...")
    
    async def create_hypervector_storage(self, genomic_data_id):
        """Create hypervector storage for genomic data."""
        dimension = random.choice([8192, 10000, 16384, 32768])
        sparsity = random.uniform(0.01, 0.3)  # 1-30% non-zero elements
        
        # Generate mock compressed hypervector
        original_size = dimension * 4  # 4 bytes per float32
        compressed_size = int(original_size * random.uniform(0.05, 0.2))  # 5-20% compression
        
        # Generate mock vector data (compressed)
        vector_data = secrets.token_bytes(compressed_size)
        
        # Generate hamming signature for similarity search
        hamming_signature = hashlib.sha256(vector_data[:1024]).hexdigest()[:64]
        
        # Generate LSH buckets
        lsh_buckets = [random.randint(0, 1000) for _ in range(random.randint(3, 8))]
        
        await self.conn.execute("""
            INSERT INTO hypervector_storage (
                id, genomic_data_id, vector_data, dimension, sparsity,
                compression_algorithm, compressed_size_bytes, original_size_bytes,
                hamming_signature, lsh_buckets, kan_enabled,
                kan_compression_ratio, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """,
            str(uuid.uuid4()), genomic_data_id, vector_data,
            dimension, sparsity, "zstd", compressed_size, original_size,
            hamming_signature, lsh_buckets,
            random.random() > 0.7,  # 30% use KAN compression
            random.uniform(10, 50) if random.random() > 0.7 else None,
            datetime.now(timezone.utc)
        )
    
    async def create_audit_logs(self, count=100):
        """Create sample audit log entries."""
        logger.info(f"Creating {count} audit log entries...")
        
        event_types = ["login", "logout", "data_access", "data_modification", "phi_access", "export", "api_call"]
        actions = ["view", "create", "update", "delete", "export", "share"]
        results = ["success", "failure", "denied"]
        
        for i in range(count):
            user = random.choice(self.users) if random.random() > 0.1 else None
            event_type = random.choice(event_types)
            
            # PHI access should be from clinicians
            if event_type == "phi_access" and user and user["role"] != "clinician":
                event_type = "data_access"
            
            event_time = datetime.now(timezone.utc) - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            await self.conn.execute("""
                INSERT INTO audit_logs (
                    id, event_time, event_type, user_id, organization_id,
                    resource_type, resource_id, action, result,
                    ip_address, user_agent, request_method, request_path,
                    response_status, response_time_ms, data_classification
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                str(uuid.uuid4()), event_time, event_type,
                user["id"] if user else None,
                user.get("org_id") if user else None,
                "genomic_data" if "data" in event_type else "system",
                random.choice(self.genomic_data)["id"] if self.genomic_data and "data" in event_type else None,
                random.choice(actions),
                random.choice(results),
                fake.ipv4(),
                fake.user_agent(),
                random.choice(["GET", "POST", "PUT", "DELETE"]) if event_type == "api_call" else None,
                f"/api/v1/{fake.word()}/{fake.word()}" if event_type == "api_call" else None,
                random.choice([200, 201, 400, 401, 403, 404, 500]) if event_type == "api_call" else None,
                random.randint(10, 5000) if event_type == "api_call" else None,
                random.choice(["public", "internal", "confidential", "phi"])
            )
            
            # Create PHI access logs for PHI events
            if event_type == "phi_access" and user:
                await self.create_phi_access_log(user, event_time)
        
        logger.info(f"Created {count} audit log entries")
    
    async def create_phi_access_log(self, user, access_time):
        """Create PHI access log entry."""
        patient = random.choice([u for u in self.users if u["role"] == "patient"])
        
        await self.conn.execute("""
            INSERT INTO phi_access_logs (
                id, access_time, user_id, patient_id,
                access_type, resource_type, resource_id,
                access_reason, case_number, ip_address,
                mfa_verified
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """,
            str(uuid.uuid4()), access_time, user["id"],
            patient["id"] if patient else None,
            random.choice(["view", "export", "modify"]),
            "patient_record",
            str(uuid.uuid4()),
            fake.sentence(nb_words=10),
            f"CASE-{fake.random_number(digits=6)}",
            fake.ipv4(),
            user["role"] == "clinician"  # Clinicians should have MFA
        )
    
    async def create_query_history(self, count=200):
        """Create query history entries."""
        logger.info(f"Creating {count} query history entries...")
        
        query_types = ["search", "aggregate", "pir_query", "federated_query", "export"]
        statuses = ["completed", "completed", "completed", "failed", "cancelled"]  # Weight towards completed
        
        for i in range(count):
            user = random.choice(self.users)
            query_time = datetime.now(timezone.utc) - timedelta(
                days=random.randint(0, 7),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            query_type = random.choice(query_types)
            status = random.choice(statuses)
            
            await self.conn.execute("""
                INSERT INTO query_history (
                    id, query_time, user_id, query_type,
                    query_hash, status, execution_time_ms,
                    rows_examined, rows_returned, cache_hit
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                str(uuid.uuid4()), query_time, user["id"],
                query_type,
                hashlib.sha256(f"query-{i}".encode()).hexdigest(),
                status,
                random.randint(10, 5000) if status == "completed" else None,
                random.randint(100, 1000000) if status == "completed" else None,
                random.randint(0, 1000) if status == "completed" else None,
                random.random() > 0.3  # 70% cache hit rate
            )
        
        logger.info(f"Created {count} query history entries")
    
    async def create_data_retention_policies(self):
        """Create default data retention policies."""
        logger.info("Creating data retention policies...")
        
        policies = [
            {
                "name": "HIPAA PHI Retention",
                "description": "7-year retention for PHI data per HIPAA requirements",
                "data_type": "phi",
                "retention_days": 2555,  # 7 years
                "action_on_expiry": "archive",
                "compliance_framework": "HIPAA",
            },
            {
                "name": "Audit Log Retention",
                "description": "7-year retention for audit logs",
                "data_type": "audit_logs",
                "retention_days": 2555,
                "action_on_expiry": "archive",
                "compliance_framework": "HIPAA",
            },
            {
                "name": "Genomic Data Retention",
                "description": "10-year retention for genomic data",
                "data_type": "genomic",
                "retention_days": 3650,
                "tier_transitions": {
                    "hot_to_warm": 30,
                    "warm_to_cold": 90,
                    "cold_to_archive": 365
                },
                "action_on_expiry": "anonymize",
                "compliance_framework": "GDPR",
            },
            {
                "name": "Query History Retention",
                "description": "1-year retention for query history",
                "data_type": "query_history",
                "retention_days": 365,
                "action_on_expiry": "delete",
                "compliance_framework": None,
            },
        ]
        
        for policy in policies:
            await self.conn.execute("""
                INSERT INTO data_retention_policies (
                    id, name, description, data_type, retention_days,
                    tier_transitions, action_on_expiry, notification_days_before,
                    compliance_framework, is_active, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                str(uuid.uuid4()),
                policy["name"],
                policy["description"],
                policy["data_type"],
                policy["retention_days"],
                policy.get("tier_transitions"),
                policy["action_on_expiry"],
                30,  # Notify 30 days before expiry
                policy.get("compliance_framework"),
                True,
                datetime.now(timezone.utc)
            )
            
            logger.info(f"Created retention policy: {policy['name']}")
    
    async def create_differential_privacy_logs(self, count=50):
        """Create differential privacy log entries."""
        logger.info(f"Creating {count} differential privacy log entries...")
        
        mechanisms = ["laplace", "gaussian", "exponential"]
        functions = ["count", "sum", "mean", "median", "histogram"]
        
        for i in range(count):
            user = random.choice([u for u in self.users if u["role"] == "researcher"])
            
            epsilon = random.uniform(0.1, 2.0)
            delta = random.uniform(1e-7, 1e-5) if random.random() > 0.5 else None
            
            await self.conn.execute("""
                INSERT INTO differential_privacy_logs (
                    id, query_id, user_id, epsilon_used, delta_used,
                    privacy_budget_remaining, noise_mechanism,
                    noise_scale, sensitivity, aggregation_function,
                    record_count, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                str(uuid.uuid4()),
                str(uuid.uuid4()),  # Random query ID
                user["id"],
                epsilon,
                delta,
                random.uniform(0, 10),  # Remaining budget
                random.choice(mechanisms),
                random.uniform(0.1, 10),  # Noise scale
                random.uniform(1, 100),  # Sensitivity
                random.choice(functions),
                random.randint(100, 10000),  # Record count
                datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
            )
        
        logger.info(f"Created {count} differential privacy log entries")
    
    async def run(self, clear_existing=True):
        """Run the seed data generation."""
        try:
            if clear_existing:
                await self.clear_existing_data()
            
            # Create data in dependency order
            await self.create_organizations(5)
            await self.create_users(20)
            await self.create_api_keys(10)
            await self.create_genomic_metadata(50)
            await self.create_audit_logs(100)
            await self.create_query_history(200)
            await self.create_data_retention_policies()
            await self.create_differential_privacy_logs(50)
            
            logger.info("Seed data generation completed successfully!")
            
            # Print summary
            await self.print_summary()
            
        except Exception as e:
            logger.error(f"Error generating seed data: {e}")
            raise
    
    async def print_summary(self):
        """Print summary of generated data."""
        logger.info("\n" + "="*50)
        logger.info("SEED DATA SUMMARY")
        logger.info("="*50)
        
        # Count records in each table
        tables = [
            "organizations",
            "users",
            "api_keys",
            "genomic_data_metadata",
            "hypervector_storage",
            "audit_logs",
            "phi_access_logs",
            "query_history",
            "differential_privacy_logs",
            "data_retention_policies"
        ]
        
        for table in tables:
            try:
                result = await self.conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                logger.info(f"{table}: {result} records")
            except Exception as e:
                logger.warning(f"Could not count {table}: {e}")
        
        logger.info("\nDefault credentials for testing:")
        logger.info("  Password for all users: genomevault123")
        logger.info("  Sample users:")
        for user in self.users[:5]:
            logger.info(f"    - {user['username']} ({user['role']})")
        
        logger.info("\n" + "="*50)


async def main():
    """Main function to run seed data generation."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate seed data for GenomeVault")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear existing data")
    parser.add_argument("--database-url", help="Database URL (overrides environment variable)")
    args = parser.parse_args()
    
    # Override database URL if provided
    if args.database_url:
        global DATABASE_URL
        DATABASE_URL = args.database_url
    
    # Connect to database
    logger.info(f"Connecting to database...")
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        # Run seed data generation
        generator = SeedDataGenerator(conn)
        await generator.run(clear_existing=not args.no_clear)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())