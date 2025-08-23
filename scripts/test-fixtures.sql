-- GenomeVault Test Fixtures
-- This script creates test data specifically for automated testing

-- Create test users with known credentials
INSERT INTO genomevault.users (id, email, password_hash, api_key_hash, role, is_active)
VALUES
    (
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'test@genomevault.test',
        crypt('test_password', gen_salt('bf')),
        encode(sha256('test_api_key_fixed'::bytea), 'hex'),
        'user',
        true
    ),
    (
        '550e8400-e29b-41d4-a716-446655440002'::uuid,
        'admin@genomevault.test',
        crypt('admin_password', gen_salt('bf')),
        encode(sha256('admin_api_key_fixed'::bytea), 'hex'),
        'admin',
        true
    ),
    (
        '550e8400-e29b-41d4-a716-446655440003'::uuid,
        'inactive@genomevault.test',
        crypt('inactive_password', gen_salt('bf')),
        encode(sha256('inactive_api_key_fixed'::bytea), 'hex'),
        'user',
        false
    )
ON CONFLICT (id) DO UPDATE SET
    email = EXCLUDED.email,
    password_hash = EXCLUDED.password_hash,
    api_key_hash = EXCLUDED.api_key_hash,
    role = EXCLUDED.role,
    is_active = EXCLUDED.is_active;

-- Create test hypervector encodings with known properties
INSERT INTO genomevault.hypervector_encodings (id, user_id, encoding_hash, dimension, is_binary, metadata)
VALUES
    (
        '660e8400-e29b-41d4-a716-446655440001'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'test_encoding_hash_1234567890abcdef',
        8192,
        false,
        '{"variant_count": 100, "compression_ratio": 85.5, "privacy_level": "standard", "sample_type": "WGS"}'::jsonb
    ),
    (
        '660e8400-e29b-41d4-a716-446655440002'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'test_encoding_hash_binary_fedcba0987654321',
        16384,
        true,
        '{"variant_count": 250, "compression_ratio": 92.1, "privacy_level": "high", "sample_type": "WES"}'::jsonb
    )
ON CONFLICT (id) DO UPDATE SET
    encoding_hash = EXCLUDED.encoding_hash,
    dimension = EXCLUDED.dimension,
    is_binary = EXCLUDED.is_binary,
    metadata = EXCLUDED.metadata;

-- Create test PIR queries with predictable timing
INSERT INTO genomevault.pir_queries (id, user_id, query_hash, response_time_ms, created_at)
VALUES
    (
        '770e8400-e29b-41d4-a716-446655440001'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'test_pir_query_hash_1234567890',
        1250,
        '2024-01-15 10:00:00+00'::timestamp with time zone
    ),
    (
        '770e8400-e29b-41d4-a716-446655440002'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'test_pir_query_hash_0987654321',
        875,
        '2024-01-15 11:00:00+00'::timestamp with time zone
    )
ON CONFLICT (id) DO UPDATE SET
    query_hash = EXCLUDED.query_hash,
    response_time_ms = EXCLUDED.response_time_ms,
    created_at = EXCLUDED.created_at;

-- Create test ZK proofs with known verification keys
INSERT INTO genomevault.zk_proofs (id, user_id, proof_id, proof_type, verification_key, created_at, expires_at)
VALUES
    (
        '880e8400-e29b-41d4-a716-446655440001'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'test_proof_genomic_001',
        'genomic',
        'test_verification_key_genomic_12345678',
        '2024-01-15 12:00:00+00'::timestamp with time zone,
        '2024-02-15 12:00:00+00'::timestamp with time zone
    ),
    (
        '880e8400-e29b-41d4-a716-446655440002'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'test_proof_clinical_002',
        'clinical',
        'test_verification_key_clinical_87654321',
        '2024-01-15 13:00:00+00'::timestamp with time zone,
        '2024-02-15 13:00:00+00'::timestamp with time zone
    )
ON CONFLICT (id) DO UPDATE SET
    proof_id = EXCLUDED.proof_id,
    proof_type = EXCLUDED.proof_type,
    verification_key = EXCLUDED.verification_key,
    created_at = EXCLUDED.created_at,
    expires_at = EXCLUDED.expires_at;

-- Create test API request metrics for performance testing
INSERT INTO metrics.api_requests (id, user_id, endpoint, method, status_code, response_time_ms, request_size_bytes, response_size_bytes, created_at)
VALUES
    (
        '990e8400-e29b-41d4-a716-446655440001'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        '/v1/hv/encode',
        'POST',
        200,
        1500,
        5000,
        25000,
        '2024-01-15 14:00:00+00'::timestamp with time zone
    ),
    (
        '990e8400-e29b-41d4-a716-446655440002'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        '/v1/pir/query',
        'POST',
        200,
        800,
        1000,
        2000,
        '2024-01-15 14:05:00+00'::timestamp with time zone
    ),
    (
        '990e8400-e29b-41d4-a716-446655440003'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        '/v1/zk/prove',
        'POST',
        200,
        3200,
        3000,
        1500,
        '2024-01-15 14:10:00+00'::timestamp with time zone
    ),
    (
        '990e8400-e29b-41d4-a716-446655440004'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        '/v1/health',
        'GET',
        200,
        50,
        0,
        500,
        '2024-01-15 14:15:00+00'::timestamp with time zone
    )
ON CONFLICT (id) DO UPDATE SET
    endpoint = EXCLUDED.endpoint,
    method = EXCLUDED.method,
    status_code = EXCLUDED.status_code,
    response_time_ms = EXCLUDED.response_time_ms,
    request_size_bytes = EXCLUDED.request_size_bytes,
    response_size_bytes = EXCLUDED.response_size_bytes,
    created_at = EXCLUDED.created_at;

-- Create test audit log entries for compliance testing
INSERT INTO audit.activity_log (id, user_id, action, resource_type, resource_id, details, ip_address, user_agent, created_at)
VALUES
    (
        'aa0e8400-e29b-41d4-a716-446655440001'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'LOGIN',
        'user',
        '550e8400-e29b-41d4-a716-446655440001',
        '{"method": "api_key", "success": true}'::jsonb,
        '192.168.1.100'::inet,
        'GenomeVault-Test-Suite/1.0',
        '2024-01-15 15:00:00+00'::timestamp with time zone
    ),
    (
        'aa0e8400-e29b-41d4-a716-446655440002'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'ENCODE_VARIANTS',
        'encoding',
        '660e8400-e29b-41d4-a716-446655440001',
        '{"variant_count": 100, "dimension": 8192, "compression_achieved": 85.5}'::jsonb,
        '192.168.1.100'::inet,
        'GenomeVault-Test-Suite/1.0',
        '2024-01-15 15:05:00+00'::timestamp with time zone
    ),
    (
        'aa0e8400-e29b-41d4-a716-446655440003'::uuid,
        '550e8400-e29b-41d4-a716-446655440001'::uuid,
        'PIR_QUERY',
        'query',
        '770e8400-e29b-41d4-a716-446655440001',
        '{"query_time_ms": 1250, "privacy_preserved": true}'::jsonb,
        '192.168.1.100'::inet,
        'GenomeVault-Test-Suite/1.0',
        '2024-01-15 15:10:00+00'::timestamp with time zone
    )
ON CONFLICT (id) DO UPDATE SET
    action = EXCLUDED.action,
    resource_type = EXCLUDED.resource_type,
    resource_id = EXCLUDED.resource_id,
    details = EXCLUDED.details,
    ip_address = EXCLUDED.ip_address,
    user_agent = EXCLUDED.user_agent,
    created_at = EXCLUDED.created_at;

-- Create test-specific functions for data validation

-- Function to validate test user authentication
CREATE OR REPLACE FUNCTION test_validate_user_auth(test_email TEXT, test_password TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    stored_hash TEXT;
BEGIN
    SELECT password_hash INTO stored_hash
    FROM genomevault.users
    WHERE email = test_email AND is_active = true;

    IF stored_hash IS NULL THEN
        RETURN false;
    END IF;

    RETURN stored_hash = crypt(test_password, stored_hash);
END;
$$ LANGUAGE plpgsql;

-- Function to get test API key hash
CREATE OR REPLACE FUNCTION test_get_api_key_hash(test_email TEXT)
RETURNS TEXT AS $$
BEGIN
    RETURN (SELECT api_key_hash FROM genomevault.users WHERE email = test_email);
END;
$$ LANGUAGE plpgsql;

-- Function to clean up test data
CREATE OR REPLACE FUNCTION test_cleanup_data()
RETURNS VOID AS $$
BEGIN
    DELETE FROM audit.activity_log WHERE user_agent LIKE 'GenomeVault-Test-Suite%';
    DELETE FROM metrics.api_requests WHERE user_id IN (
        SELECT id FROM genomevault.users WHERE email LIKE '%@genomevault.test'
    );
    DELETE FROM genomevault.zk_proofs WHERE user_id IN (
        SELECT id FROM genomevault.users WHERE email LIKE '%@genomevault.test'
    );
    DELETE FROM genomevault.pir_queries WHERE user_id IN (
        SELECT id FROM genomevault.users WHERE email LIKE '%@genomevault.test'
    );
    DELETE FROM genomevault.hypervector_encodings WHERE user_id IN (
        SELECT id FROM genomevault.users WHERE email LIKE '%@genomevault.test'
    );
    DELETE FROM genomevault.users WHERE email LIKE '%@genomevault.test';

    RAISE NOTICE 'Test data cleaned up successfully';
END;
$$ LANGUAGE plpgsql;

-- Display test fixtures summary
DO $$
BEGIN
    RAISE NOTICE 'Test fixtures created successfully:';
    RAISE NOTICE '  Test users: test@genomevault.test, admin@genomevault.test, inactive@genomevault.test';
    RAISE NOTICE '  Test encodings: 2 hypervector encodings with known properties';
    RAISE NOTICE '  Test PIR queries: 2 queries with predictable response times';
    RAISE NOTICE '  Test ZK proofs: 2 proofs with known verification keys';
    RAISE NOTICE '  Test API metrics: 4 request records with known values';
    RAISE NOTICE '  Test audit logs: 3 activity records for compliance testing';
    RAISE NOTICE '  Test functions: test_validate_user_auth, test_get_api_key_hash, test_cleanup_data';
END $$;
