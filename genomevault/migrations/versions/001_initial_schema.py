"""
Initial database schema for GenomeVault.

Creates all base tables with HIPAA-compliant audit logging,
user management, genomic metadata storage, and hypervector storage.

Revision ID: 001
Revises: 
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime

# Revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Create initial database schema for GenomeVault.
    """
    
    # Create custom types
    op.execute("CREATE TYPE user_role AS ENUM ('patient', 'clinician', 'researcher', 'admin', 'service');")
    op.execute("CREATE TYPE audit_event_type AS ENUM ('login', 'logout', 'data_access', 'data_modification', 'phi_access', 'export', 'api_call', 'error');")
    op.execute("CREATE TYPE data_classification AS ENUM ('public', 'internal', 'confidential', 'phi', 'restricted');")
    op.execute("CREATE TYPE query_status AS ENUM ('pending', 'processing', 'completed', 'failed', 'cancelled');")
    op.execute("CREATE TYPE retention_tier AS ENUM ('hot', 'warm', 'cold', 'archive');")
    
    # ==================== USER MANAGEMENT TABLES ====================
    
    # Organizations table
    op.create_table(
        'organizations',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('domain', sa.String(255), nullable=True),
        sa.Column('npi_number', sa.String(10), nullable=True),  # For healthcare organizations
        sa.Column('baa_signed', sa.Boolean(), default=False),
        sa.Column('baa_signed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('baa_document_hash', sa.String(64), nullable=True),
        sa.Column('settings', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=datetime.utcnow),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('domain'),
        sa.Index('idx_organizations_npi', 'npi_number'),
    )
    
    # Users table with HIPAA fields
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('username', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('hashed_password', sa.String(255), nullable=True),  # Null for SSO users
        sa.Column('organization_id', sa.UUID(), nullable=True),
        
        # Roles and permissions
        sa.Column('roles', postgresql.ARRAY(sa.String), default=[]),
        sa.Column('scopes', postgresql.ARRAY(sa.String), default=[]),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_verified', sa.Boolean(), default=False),
        
        # HIPAA compliance fields
        sa.Column('npi_number', sa.String(10), nullable=True),
        sa.Column('dea_number', sa.String(9), nullable=True),
        sa.Column('medical_license_number', sa.String(50), nullable=True),
        sa.Column('medical_license_state', sa.String(2), nullable=True),
        
        # Security fields
        sa.Column('mfa_enabled', sa.Boolean(), default=False),
        sa.Column('mfa_secret', sa.String(255), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), default=0),
        sa.Column('account_locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('password_changed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_login_ip', sa.INET(), nullable=True),
        
        # Rate limiting
        sa.Column('rate_limit_tier', sa.String(50), default='basic'),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=datetime.utcnow),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),  # Soft delete
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='SET NULL'),
        sa.Index('idx_users_email', 'email'),
        sa.Index('idx_users_organization', 'organization_id'),
        sa.Index('idx_users_npi', 'npi_number'),
        sa.Index('idx_users_active', 'is_active', 'deleted_at'),
    )
    
    # Sessions table for token management
    op.create_table(
        'sessions',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('token_jti', sa.String(255), nullable=False),  # JWT ID
        sa.Column('refresh_token_id', sa.String(255), nullable=True),
        sa.Column('ip_address', sa.INET(), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('revoked_reason', sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('token_jti'),
        sa.Index('idx_sessions_user', 'user_id'),
        sa.Index('idx_sessions_expires', 'expires_at'),
    )
    
    # API Keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('key_id', sa.String(255), nullable=False),
        sa.Column('key_hash', sa.String(64), nullable=False),  # SHA-256 hash
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('scopes', postgresql.ARRAY(sa.String), default=[]),
        sa.Column('allowed_ips', postgresql.ARRAY(sa.INET), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_rotated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rotation_interval_days', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('key_id'),
        sa.UniqueConstraint('key_hash'),
        sa.Index('idx_api_keys_user', 'user_id'),
        sa.Index('idx_api_keys_hash', 'key_hash'),
    )
    
    # ==================== AUDIT LOGGING TABLES (PARTITIONED) ====================
    
    # Main audit logs table (partitioned by month)
    op.execute("""
        CREATE TABLE audit_logs (
            id UUID DEFAULT gen_random_uuid(),
            event_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            event_type audit_event_type NOT NULL,
            user_id UUID,
            organization_id UUID,
            session_id UUID,
            
            -- Event details
            resource_type VARCHAR(100),
            resource_id VARCHAR(255),
            action VARCHAR(100),
            result VARCHAR(50),
            
            -- Request information
            ip_address INET,
            user_agent TEXT,
            request_method VARCHAR(10),
            request_path TEXT,
            request_params JSONB,
            response_status INTEGER,
            response_time_ms INTEGER,
            
            -- Additional context
            details JSONB,
            error_message TEXT,
            
            -- Data classification
            data_classification data_classification,
            
            PRIMARY KEY (id, event_time),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
        ) PARTITION BY RANGE (event_time);
        
        -- Create indexes on parent table (inherited by partitions)
        CREATE INDEX idx_audit_logs_event_time ON audit_logs (event_time DESC);
        CREATE INDEX idx_audit_logs_user ON audit_logs (user_id, event_time DESC);
        CREATE INDEX idx_audit_logs_event_type ON audit_logs (event_type, event_time DESC);
        CREATE INDEX idx_audit_logs_resource ON audit_logs (resource_type, resource_id, event_time DESC);
    """)
    
    # PHI Access Logs (stricter retention, encrypted)
    op.execute("""
        CREATE TABLE phi_access_logs (
            id UUID DEFAULT gen_random_uuid(),
            access_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id UUID NOT NULL,
            patient_id UUID,
            
            -- Access details
            access_type VARCHAR(50) NOT NULL,  -- view, export, modify, delete
            resource_type VARCHAR(100) NOT NULL,
            resource_id VARCHAR(255),
            
            -- Justification (required for HIPAA)
            access_reason TEXT NOT NULL,
            case_number VARCHAR(100),
            
            -- Security context
            ip_address INET,
            session_id UUID,
            mfa_verified BOOLEAN DEFAULT FALSE,
            
            -- Encrypted data snapshot (for forensics)
            data_snapshot BYTEA,  -- Encrypted with key from KMS
            
            PRIMARY KEY (id, access_time),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE RESTRICT,  -- Never delete PHI logs
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
        ) PARTITION BY RANGE (access_time);
        
        CREATE INDEX idx_phi_access_time ON phi_access_logs (access_time DESC);
        CREATE INDEX idx_phi_access_user ON phi_access_logs (user_id, access_time DESC);
        CREATE INDEX idx_phi_access_patient ON phi_access_logs (patient_id, access_time DESC);
    """)
    
    # ==================== GENOMIC DATA METADATA ====================
    
    # Genomic data metadata (actual data stored in hypervectors/PIR)
    op.create_table(
        'genomic_data_metadata',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('owner_id', sa.UUID(), nullable=False),
        sa.Column('organization_id', sa.UUID(), nullable=True),
        
        # Data identification
        sa.Column('data_hash', sa.String(64), nullable=False),  # SHA-256 of original data
        sa.Column('data_type', sa.String(50), nullable=False),  # vcf, fastq, bam, etc.
        sa.Column('data_classification', sa.Enum('public', 'internal', 'confidential', 'phi', 'restricted', name='data_classification'), nullable=False),
        
        # Hypervector reference
        sa.Column('hypervector_id', sa.UUID(), nullable=True),
        sa.Column('hypervector_dimension', sa.Integer(), nullable=True),
        sa.Column('compression_ratio', sa.Float(), nullable=True),
        
        # PIR storage reference
        sa.Column('pir_server_ids', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('pir_shard_keys', postgresql.JSONB(), nullable=True),
        
        # Metadata
        sa.Column('sample_id', sa.String(255), nullable=True),
        sa.Column('sequencing_date', sa.Date(), nullable=True),
        sa.Column('sequencing_platform', sa.String(100), nullable=True),
        sa.Column('reference_genome', sa.String(50), nullable=True),
        sa.Column('variant_count', sa.Integer(), nullable=True),
        sa.Column('coverage_mean', sa.Float(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        
        # Privacy settings
        sa.Column('consent_id', sa.String(255), nullable=True),
        sa.Column('sharing_permissions', postgresql.JSONB(), default={}),
        sa.Column('differential_privacy_epsilon', sa.Float(), nullable=True),
        
        # Retention
        sa.Column('retention_tier', sa.Enum('hot', 'warm', 'cold', 'archive', name='retention_tier'), default='hot'),
        sa.Column('retention_expires_at', sa.DateTime(timezone=True), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=datetime.utcnow),
        sa.Column('accessed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='RESTRICT'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='SET NULL'),
        sa.UniqueConstraint('data_hash'),
        sa.Index('idx_genomic_metadata_owner', 'owner_id'),
        sa.Index('idx_genomic_metadata_hash', 'data_hash'),
        sa.Index('idx_genomic_metadata_hypervector', 'hypervector_id'),
        sa.Index('idx_genomic_metadata_retention', 'retention_tier', 'retention_expires_at'),
    )
    
    # ==================== HYPERVECTOR STORAGE ====================
    
    # Hypervector storage with compression
    op.create_table(
        'hypervector_storage',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('genomic_data_id', sa.UUID(), nullable=False),
        
        # Vector data (compressed)
        sa.Column('vector_data', sa.LargeBinary(), nullable=False),  # Compressed binary
        sa.Column('dimension', sa.Integer(), nullable=False),
        sa.Column('sparsity', sa.Float(), nullable=True),  # Percentage of non-zero elements
        sa.Column('compression_algorithm', sa.String(50), default='zstd'),
        sa.Column('compressed_size_bytes', sa.Integer(), nullable=False),
        sa.Column('original_size_bytes', sa.Integer(), nullable=False),
        
        # Indexing support
        sa.Column('hamming_signature', sa.String(255), nullable=True),  # For similarity search
        sa.Column('lsh_buckets', postgresql.ARRAY(sa.Integer), nullable=True),  # LSH for approximate search
        
        # KAN compression parameters
        sa.Column('kan_enabled', sa.Boolean(), default=False),
        sa.Column('kan_spline_degree', sa.Integer(), nullable=True),
        sa.Column('kan_compression_ratio', sa.Float(), nullable=True),
        
        # Version control
        sa.Column('version', sa.Integer(), default=1),
        sa.Column('previous_version_id', sa.UUID(), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('accessed_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['genomic_data_id'], ['genomic_data_metadata.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['previous_version_id'], ['hypervector_storage.id'], ondelete='SET NULL'),
        sa.Index('idx_hypervector_genomic', 'genomic_data_id'),
        sa.Index('idx_hypervector_hamming', 'hamming_signature'),
        # GIN index for LSH buckets
        sa.Index('idx_hypervector_lsh', 'lsh_buckets', postgresql_using='gin'),
    )
    
    # ==================== QUERY HISTORY AND DIFFERENTIAL PRIVACY ====================
    
    # Query history table (partitioned daily for high volume)
    op.execute("""
        CREATE TABLE query_history (
            id UUID DEFAULT gen_random_uuid(),
            query_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id UUID NOT NULL,
            session_id UUID,
            
            -- Query details
            query_type VARCHAR(100) NOT NULL,
            query_hash VARCHAR(64),  -- For detecting repeated queries
            query_params JSONB,
            
            -- Resources accessed
            resource_ids TEXT[],
            data_classifications data_classification[],
            
            -- Performance metrics
            execution_time_ms INTEGER,
            rows_examined INTEGER,
            rows_returned INTEGER,
            cache_hit BOOLEAN DEFAULT FALSE,
            
            -- PIR details
            pir_servers_queried TEXT[],
            pir_response_sizes INTEGER[],
            
            -- Status
            status query_status NOT NULL DEFAULT 'pending',
            error_message TEXT,
            
            PRIMARY KEY (id, query_time),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
        ) PARTITION BY RANGE (query_time);
        
        CREATE INDEX idx_query_history_time ON query_history (query_time DESC);
        CREATE INDEX idx_query_history_user ON query_history (user_id, query_time DESC);
        CREATE INDEX idx_query_history_type ON query_history (query_type, query_time DESC);
        CREATE INDEX idx_query_history_hash ON query_history (query_hash);
    """)
    
    # Differential privacy tracking
    op.create_table(
        'differential_privacy_logs',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('query_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        
        # Privacy budget tracking
        sa.Column('epsilon_used', sa.Float(), nullable=False),
        sa.Column('delta_used', sa.Float(), nullable=True),
        sa.Column('privacy_budget_remaining', sa.Float(), nullable=False),
        
        # Noise parameters
        sa.Column('noise_mechanism', sa.String(50), nullable=False),  # laplace, gaussian, exponential
        sa.Column('noise_scale', sa.Float(), nullable=False),
        sa.Column('sensitivity', sa.Float(), nullable=False),
        
        # Aggregation details
        sa.Column('aggregation_function', sa.String(100), nullable=True),
        sa.Column('group_by_columns', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('record_count', sa.Integer(), nullable=True),
        
        # Timestamp
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.Index('idx_dp_logs_user', 'user_id'),
        sa.Index('idx_dp_logs_query', 'query_id'),
    )
    
    # ==================== DATA RETENTION POLICIES ====================
    
    op.create_table(
        'data_retention_policies',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        
        # Policy configuration
        sa.Column('data_type', sa.String(100), nullable=False),
        sa.Column('retention_days', sa.Integer(), nullable=False),
        sa.Column('tier_transitions', postgresql.JSONB(), nullable=True),  # hot -> warm -> cold -> archive
        
        # Actions
        sa.Column('action_on_expiry', sa.String(50), nullable=False),  # delete, archive, anonymize
        sa.Column('notification_days_before', sa.Integer(), default=30),
        
        # Compliance
        sa.Column('compliance_framework', sa.String(50), nullable=True),  # HIPAA, GDPR, etc.
        sa.Column('legal_hold', sa.Boolean(), default=False),
        
        # Application
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('apply_to_existing', sa.Boolean(), default=False),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=datetime.utcnow),
        
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        sa.Index('idx_retention_policies_type', 'data_type'),
    )
    
    # ==================== BACKUP AND DISASTER RECOVERY ====================
    
    op.create_table(
        'backup_history',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('backup_type', sa.String(50), nullable=False),  # full, incremental, differential
        sa.Column('backup_status', sa.String(50), nullable=False),  # started, completed, failed
        
        # Backup details
        sa.Column('backup_location', sa.String(500), nullable=False),
        sa.Column('backup_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('encrypted', sa.Boolean(), default=True),
        sa.Column('encryption_key_id', sa.String(255), nullable=True),
        sa.Column('checksum', sa.String(64), nullable=True),
        
        # Timing
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        
        # Recovery point objective (RPO) tracking
        sa.Column('data_start_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('data_end_time', sa.DateTime(timezone=True), nullable=True),
        
        # Metadata
        sa.Column('tables_backed_up', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('row_counts', postgresql.JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        
        sa.PrimaryKeyConstraint('id'),
        sa.Index('idx_backup_history_time', 'started_at' DESC),
        sa.Index('idx_backup_history_status', 'backup_status'),
    )
    
    # ==================== CREATE INITIAL PARTITIONS ====================
    
    # Create monthly partitions for audit_logs (next 12 months)
    for year in [2025, 2026]:
        for month in range(1, 13):
            if year == 2025 or month <= 6:  # Create 18 months of partitions
                partition_name = f"audit_logs_{year}_{month:02d}"
                start_date = f"{year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{year + 1}-01-01"
                else:
                    end_date = f"{year}-{month + 1:02d}-01"
                
                op.execute(f"""
                    CREATE TABLE IF NOT EXISTS {partition_name}
                    PARTITION OF audit_logs
                    FOR VALUES FROM ('{start_date}') TO ('{end_date}');
                """)
    
    # Create monthly partitions for PHI access logs
    for year in [2025, 2026]:
        for month in range(1, 13):
            if year == 2025 or month <= 6:
                partition_name = f"phi_access_logs_{year}_{month:02d}"
                start_date = f"{year}-{month:02d}-01"
                if month == 12:
                    end_date = f"{year + 1}-01-01"
                else:
                    end_date = f"{year}-{month + 1:02d}-01"
                
                op.execute(f"""
                    CREATE TABLE IF NOT EXISTS {partition_name}
                    PARTITION OF phi_access_logs
                    FOR VALUES FROM ('{start_date}') TO ('{end_date}');
                """)
    
    # Create daily partitions for query_history (next 30 days)
    op.execute("""
        -- Create a function to automatically create daily partitions
        CREATE OR REPLACE FUNCTION create_daily_partitions()
        RETURNS void AS $$
        DECLARE
            partition_date date;
            partition_name text;
        BEGIN
            FOR partition_date IN 
                SELECT generate_series(CURRENT_DATE, CURRENT_DATE + interval '30 days', interval '1 day')::date
            LOOP
                partition_name := 'query_history_' || to_char(partition_date, 'YYYY_MM_DD');
                
                EXECUTE format('
                    CREATE TABLE IF NOT EXISTS %I
                    PARTITION OF query_history
                    FOR VALUES FROM (%L) TO (%L)',
                    partition_name,
                    partition_date,
                    partition_date + interval '1 day'
                );
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Create initial partitions
        SELECT create_daily_partitions();
    """)
    
    # ==================== SECURITY POLICIES ====================
    
    # Create row-level security policies
    op.execute("""
        -- Enable RLS on sensitive tables
        ALTER TABLE users ENABLE ROW LEVEL SECURITY;
        ALTER TABLE genomic_data_metadata ENABLE ROW LEVEL SECURITY;
        ALTER TABLE hypervector_storage ENABLE ROW LEVEL SECURITY;
        
        -- Users can only see their own data (except admins)
        CREATE POLICY users_isolation ON users
            FOR ALL
            USING (
                id = current_setting('app.current_user_id')::uuid
                OR current_setting('app.current_user_role') = 'admin'
            );
        
        -- Genomic data access control
        CREATE POLICY genomic_data_access ON genomic_data_metadata
            FOR SELECT
            USING (
                owner_id = current_setting('app.current_user_id')::uuid
                OR current_setting('app.current_user_role') IN ('admin', 'researcher')
                OR EXISTS (
                    SELECT 1 FROM jsonb_array_elements_text(sharing_permissions->'users') AS u
                    WHERE u = current_setting('app.current_user_id')
                )
            );
        
        -- Hypervector access follows genomic data permissions
        CREATE POLICY hypervector_access ON hypervector_storage
            FOR SELECT
            USING (
                EXISTS (
                    SELECT 1 FROM genomic_data_metadata g
                    WHERE g.id = genomic_data_id
                    AND (
                        g.owner_id = current_setting('app.current_user_id')::uuid
                        OR current_setting('app.current_user_role') IN ('admin', 'researcher')
                    )
                )
            );
    """)
    
    # ==================== PERFORMANCE INDEXES ====================
    
    # Create additional performance indexes
    op.execute("""
        -- Full-text search on audit logs
        CREATE INDEX idx_audit_logs_details_gin ON audit_logs USING gin(details);
        
        -- Trigram search for user lookups
        CREATE INDEX idx_users_full_name_trgm ON users USING gin(full_name gin_trgm_ops);
        
        -- BRIN index for time-series data (very efficient for large tables)
        CREATE INDEX idx_audit_logs_time_brin ON audit_logs USING brin(event_time);
        CREATE INDEX idx_query_history_time_brin ON query_history USING brin(query_time);
        
        -- Partial indexes for common queries
        CREATE INDEX idx_users_active_verified ON users(id) WHERE is_active = true AND is_verified = true;
        CREATE INDEX idx_genomic_data_hot ON genomic_data_metadata(id, owner_id) WHERE retention_tier = 'hot';
    """)
    
    # ==================== STORED PROCEDURES ====================
    
    # Create stored procedure for data retention enforcement
    op.execute("""
        CREATE OR REPLACE FUNCTION enforce_data_retention()
        RETURNS void AS $$
        DECLARE
            policy RECORD;
        BEGIN
            FOR policy IN SELECT * FROM data_retention_policies WHERE is_active = true
            LOOP
                -- Move data between tiers based on age
                IF policy.tier_transitions IS NOT NULL THEN
                    -- Implementation would go here
                    RAISE NOTICE 'Processing tier transitions for policy %', policy.name;
                END IF;
                
                -- Handle expired data
                IF policy.action_on_expiry = 'delete' THEN
                    DELETE FROM genomic_data_metadata
                    WHERE data_type = policy.data_type
                    AND created_at < CURRENT_TIMESTAMP - (policy.retention_days || ' days')::interval
                    AND NOT EXISTS (
                        SELECT 1 FROM data_retention_policies
                        WHERE legal_hold = true
                        AND data_type = policy.data_type
                    );
                ELSIF policy.action_on_expiry = 'archive' THEN
                    UPDATE genomic_data_metadata
                    SET retention_tier = 'archive'
                    WHERE data_type = policy.data_type
                    AND created_at < CURRENT_TIMESTAMP - (policy.retention_days || ' days')::interval;
                END IF;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create trigger for automatic partition creation
    op.execute("""
        CREATE OR REPLACE FUNCTION create_partition_if_not_exists()
        RETURNS trigger AS $$
        DECLARE
            partition_name text;
            start_date date;
            end_date date;
        BEGIN
            -- For audit_logs (monthly partitions)
            IF TG_TABLE_NAME = 'audit_logs' THEN
                partition_name := 'audit_logs_' || to_char(NEW.event_time, 'YYYY_MM');
                start_date := date_trunc('month', NEW.event_time)::date;
                end_date := (date_trunc('month', NEW.event_time) + interval '1 month')::date;
                
                EXECUTE format('
                    CREATE TABLE IF NOT EXISTS %I
                    PARTITION OF audit_logs
                    FOR VALUES FROM (%L) TO (%L)',
                    partition_name, start_date, end_date
                );
            END IF;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        
        -- Note: Trigger would be created on parent table in production
    """)


def downgrade() -> None:
    """
    Drop all tables and types created in upgrade.
    """
    # Drop stored procedures and functions
    op.execute("DROP FUNCTION IF EXISTS enforce_data_retention() CASCADE;")
    op.execute("DROP FUNCTION IF EXISTS create_partition_if_not_exists() CASCADE;")
    op.execute("DROP FUNCTION IF EXISTS create_daily_partitions() CASCADE;")
    
    # Drop policies
    op.execute("DROP POLICY IF EXISTS users_isolation ON users;")
    op.execute("DROP POLICY IF EXISTS genomic_data_access ON genomic_data_metadata;")
    op.execute("DROP POLICY IF EXISTS hypervector_access ON hypervector_storage;")
    
    # Drop tables in reverse order of dependencies
    op.drop_table('backup_history')
    op.drop_table('data_retention_policies')
    op.drop_table('differential_privacy_logs')
    op.execute("DROP TABLE IF EXISTS query_history CASCADE;")  # Partitioned table
    op.drop_table('hypervector_storage')
    op.drop_table('genomic_data_metadata')
    op.execute("DROP TABLE IF EXISTS phi_access_logs CASCADE;")  # Partitioned table
    op.execute("DROP TABLE IF EXISTS audit_logs CASCADE;")  # Partitioned table
    op.drop_table('api_keys')
    op.drop_table('sessions')
    op.drop_table('users')
    op.drop_table('organizations')
    
    # Drop custom types
    op.execute("DROP TYPE IF EXISTS retention_tier;")
    op.execute("DROP TYPE IF EXISTS query_status;")
    op.execute("DROP TYPE IF EXISTS data_classification;")
    op.execute("DROP TYPE IF EXISTS audit_event_type;")
    op.execute("DROP TYPE IF EXISTS user_role;")