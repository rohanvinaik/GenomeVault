-- GenomeVault Database Initialization Script
-- This script sets up the basic database structure and extensions

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS genomevault;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS metrics;

-- Set default search path
ALTER DATABASE genomevault SET search_path TO genomevault, public;

-- Create basic tables for GenomeVault

-- Users table for authentication
CREATE TABLE IF NOT EXISTS genomevault.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    api_key_hash VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    role VARCHAR(50) DEFAULT 'user'
);

-- Hypervector encodings table
CREATE TABLE IF NOT EXISTS genomevault.hypervector_encodings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES genomevault.users(id),
    encoding_hash VARCHAR(64) UNIQUE NOT NULL,
    dimension INTEGER NOT NULL,
    is_binary BOOLEAN DEFAULT false,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- PIR queries table for audit trail
CREATE TABLE IF NOT EXISTS genomevault.pir_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES genomevault.users(id),
    query_hash VARCHAR(64) NOT NULL,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ZK proofs table
CREATE TABLE IF NOT EXISTS genomevault.zk_proofs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES genomevault.users(id),
    proof_id VARCHAR(255) UNIQUE NOT NULL,
    proof_type VARCHAR(50) NOT NULL,
    verification_key TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- API usage metrics
CREATE TABLE IF NOT EXISTS metrics.api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES genomevault.users(id),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit log for compliance
CREATE TABLE IF NOT EXISTS audit.activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES genomevault.users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON genomevault.users(email);
CREATE INDEX IF NOT EXISTS idx_users_api_key_hash ON genomevault.users(api_key_hash);
CREATE INDEX IF NOT EXISTS idx_hypervector_encodings_user_id ON genomevault.hypervector_encodings(user_id);
CREATE INDEX IF NOT EXISTS idx_hypervector_encodings_encoding_hash ON genomevault.hypervector_encodings(encoding_hash);
CREATE INDEX IF NOT EXISTS idx_pir_queries_user_id ON genomevault.pir_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_pir_queries_created_at ON genomevault.pir_queries(created_at);
CREATE INDEX IF NOT EXISTS idx_zk_proofs_user_id ON genomevault.zk_proofs(user_id);
CREATE INDEX IF NOT EXISTS idx_zk_proofs_proof_id ON genomevault.zk_proofs(proof_id);
CREATE INDEX IF NOT EXISTS idx_api_requests_user_id ON metrics.api_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_api_requests_created_at ON metrics.api_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_api_requests_endpoint ON metrics.api_requests(endpoint);
CREATE INDEX IF NOT EXISTS idx_activity_log_user_id ON audit.activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_activity_log_created_at ON audit.activity_log(created_at);
CREATE INDEX IF NOT EXISTS idx_activity_log_action ON audit.activity_log(action);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to users table
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON genomevault.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create a default admin user (for development only)
INSERT INTO genomevault.users (email, password_hash, api_key_hash, role, is_active)
VALUES (
    'admin@genomevault.dev',
    crypt('genomevault_admin_dev', gen_salt('bf')),
    encode(sha256('genomevault_admin_dev_api_key'::bytea), 'hex'),
    'admin',
    true
) ON CONFLICT (email) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA genomevault TO PUBLIC;
GRANT USAGE ON SCHEMA audit TO PUBLIC;
GRANT USAGE ON SCHEMA metrics TO PUBLIC;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA genomevault TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA audit TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA metrics TO PUBLIC;

GRANT USAGE ON ALL SEQUENCES IN SCHEMA genomevault TO PUBLIC;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA audit TO PUBLIC;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA metrics TO PUBLIC;

-- Display initialization status
DO $$
BEGIN
    RAISE NOTICE 'GenomeVault database initialized successfully';
    RAISE NOTICE 'Schemas created: genomevault, audit, metrics';
    RAISE NOTICE 'Extensions enabled: uuid-ossp, pgcrypto, pg_stat_statements';
    RAISE NOTICE 'Default admin user created: admin@genomevault.dev';
END $$;
