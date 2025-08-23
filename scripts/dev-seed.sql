-- GenomeVault Development Seed Data
-- This script populates the database with sample data for development

-- Insert additional development users
INSERT INTO genomevault.users (email, password_hash, api_key_hash, role, is_active)
VALUES
    (
        'developer@genomevault.dev',
        crypt('dev_password', gen_salt('bf')),
        encode(sha256('dev_api_key_12345'::bytea), 'hex'),
        'developer',
        true
    ),
    (
        'researcher@genomevault.dev',
        crypt('research_password', gen_salt('bf')),
        encode(sha256('research_api_key_67890'::bytea), 'hex'),
        'researcher',
        true
    ),
    (
        'clinician@genomevault.dev',
        crypt('clinical_password', gen_salt('bf')),
        encode(sha256('clinical_api_key_abcde'::bytea), 'hex'),
        'clinician',
        true
    )
ON CONFLICT (email) DO NOTHING;

-- Insert sample hypervector encodings
INSERT INTO genomevault.hypervector_encodings (user_id, encoding_hash, dimension, is_binary, metadata)
SELECT
    u.id,
    encode(sha256((u.email || '_sample_encoding_' || generate_random_uuid()::text)::bytea), 'hex'),
    CASE WHEN random() < 0.3 THEN 1000
         WHEN random() < 0.6 THEN 8192
         ELSE 16384 END,
    random() < 0.5,
    json_build_object(
        'variant_count', floor(random() * 1000 + 1)::int,
        'compression_ratio', round((random() * 50 + 50)::numeric, 2),
        'privacy_level', CASE WHEN random() < 0.3 THEN 'basic'
                             WHEN random() < 0.6 THEN 'standard'
                             ELSE 'high' END,
        'sample_type', CASE WHEN random() < 0.25 THEN 'WGS'
                           WHEN random() < 0.5 THEN 'WES'
                           WHEN random() < 0.75 THEN 'targeted_panel'
                           ELSE 'SNP_array' END
    )
FROM genomevault.users u
CROSS JOIN generate_series(1, floor(random() * 10 + 1)::int)
WHERE u.role IN ('developer', 'researcher', 'clinician');

-- Insert sample PIR queries with realistic timing
INSERT INTO genomevault.pir_queries (user_id, query_hash, response_time_ms, created_at)
SELECT
    u.id,
    encode(sha256((u.email || '_pir_query_' || s.i || '_' || generate_random_uuid()::text)::bytea), 'hex'),
    floor(random() * 5000 + 100)::int, -- 100-5100ms response time
    NOW() - (random() * interval '30 days')
FROM genomevault.users u
CROSS JOIN generate_series(1, floor(random() * 50 + 5)::int) s(i)
WHERE u.role IN ('developer', 'researcher', 'clinician');

-- Insert sample ZK proofs
INSERT INTO genomevault.zk_proofs (user_id, proof_id, proof_type, verification_key, created_at, expires_at)
SELECT
    u.id,
    'proof_' || encode(sha256((u.email || '_' || s.i::text)::bytea), 'hex')[1:16],
    CASE WHEN random() < 0.4 THEN 'genomic'
         WHEN random() < 0.7 THEN 'clinical'
         ELSE 'research' END,
    'vk_' || encode(sha256(generate_random_uuid()::text::bytea), 'hex')[1:32],
    NOW() - (random() * interval '7 days'),
    NOW() + (random() * interval '30 days')
FROM genomevault.users u
CROSS JOIN generate_series(1, floor(random() * 20 + 2)::int) s(i)
WHERE u.role IN ('developer', 'researcher', 'clinician');

-- Insert sample API request metrics for the last 7 days
INSERT INTO metrics.api_requests (user_id, endpoint, method, status_code, response_time_ms, request_size_bytes, response_size_bytes, created_at)
SELECT
    u.id,
    CASE floor(random() * 6)
        WHEN 0 THEN '/v1/hv/encode'
        WHEN 1 THEN '/v1/pir/query'
        WHEN 2 THEN '/v1/zk/prove'
        WHEN 3 THEN '/v1/clinical/analyze'
        WHEN 4 THEN '/v1/health'
        ELSE '/v1/auth/token'
    END,
    CASE floor(random() * 4)
        WHEN 0 THEN 'GET'
        WHEN 1 THEN 'POST'
        WHEN 2 THEN 'PUT'
        ELSE 'DELETE'
    END,
    CASE
        WHEN random() < 0.85 THEN 200
        WHEN random() < 0.92 THEN 400
        WHEN random() < 0.97 THEN 404
        WHEN random() < 0.99 THEN 429
        ELSE 500
    END,
    floor(random() * 3000 + 50)::int, -- 50-3050ms response time
    floor(random() * 100000 + 1000)::int, -- 1KB-100KB request size
    floor(random() * 1000000 + 5000)::int, -- 5KB-1MB response size
    NOW() - (random() * interval '7 days')
FROM genomevault.users u
CROSS JOIN generate_series(1, floor(random() * 100 + 20)::int) s(i)
WHERE u.role IN ('developer', 'researcher', 'clinician');

-- Insert sample audit log entries
INSERT INTO audit.activity_log (user_id, action, resource_type, resource_id, details, ip_address, user_agent, created_at)
SELECT
    u.id,
    CASE floor(random() * 8)
        WHEN 0 THEN 'LOGIN'
        WHEN 1 THEN 'LOGOUT'
        WHEN 2 THEN 'ENCODE_VARIANTS'
        WHEN 3 THEN 'PIR_QUERY'
        WHEN 4 THEN 'GENERATE_PROOF'
        WHEN 5 THEN 'CLINICAL_ANALYSIS'
        WHEN 6 THEN 'API_KEY_GENERATED'
        ELSE 'DATA_EXPORT'
    END,
    CASE floor(random() * 4)
        WHEN 0 THEN 'user'
        WHEN 1 THEN 'encoding'
        WHEN 2 THEN 'proof'
        ELSE 'query'
    END,
    generate_random_uuid()::text,
    json_build_object(
        'timestamp', NOW() - (random() * interval '7 days'),
        'success', random() < 0.95,
        'details', 'Sample audit log entry for development'
    ),
    ('192.168.' || floor(random() * 255) || '.' || floor(random() * 255))::inet,
    CASE floor(random() * 3)
        WHEN 0 THEN 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) GenomeVault-SDK/1.0'
        WHEN 1 THEN 'GenomeVault-CLI/1.0 Python/3.11'
        ELSE 'GenomeVault-JS-SDK/1.0'
    END,
    NOW() - (random() * interval '7 days')
FROM genomevault.users u
CROSS JOIN generate_series(1, floor(random() * 30 + 5)::int) s(i)
WHERE u.role IN ('developer', 'researcher', 'clinician');

-- Create some sample views for development convenience
CREATE OR REPLACE VIEW genomevault.user_activity_summary AS
SELECT
    u.id,
    u.email,
    u.role,
    COUNT(DISTINCT he.id) as encoding_count,
    COUNT(DISTINCT pq.id) as pir_query_count,
    COUNT(DISTINCT zp.id) as proof_count,
    COUNT(DISTINCT ar.id) as api_request_count,
    u.last_login,
    u.created_at
FROM genomevault.users u
LEFT JOIN genomevault.hypervector_encodings he ON u.id = he.user_id
LEFT JOIN genomevault.pir_queries pq ON u.id = pq.user_id
LEFT JOIN genomevault.zk_proofs zp ON u.id = zp.user_id
LEFT JOIN metrics.api_requests ar ON u.id = ar.user_id
GROUP BY u.id, u.email, u.role, u.last_login, u.created_at
ORDER BY u.created_at DESC;

-- Create a view for recent API activity
CREATE OR REPLACE VIEW metrics.recent_api_activity AS
SELECT
    ar.endpoint,
    ar.method,
    COUNT(*) as request_count,
    AVG(ar.response_time_ms) as avg_response_time_ms,
    COUNT(*) FILTER (WHERE ar.status_code = 200) as success_count,
    COUNT(*) FILTER (WHERE ar.status_code >= 400) as error_count,
    MAX(ar.created_at) as last_request_at
FROM metrics.api_requests ar
WHERE ar.created_at >= NOW() - interval '24 hours'
GROUP BY ar.endpoint, ar.method
ORDER BY request_count DESC;

-- Display seed data summary
DO $$
DECLARE
    user_count INTEGER;
    encoding_count INTEGER;
    query_count INTEGER;
    proof_count INTEGER;
    metric_count INTEGER;
    audit_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO user_count FROM genomevault.users;
    SELECT COUNT(*) INTO encoding_count FROM genomevault.hypervector_encodings;
    SELECT COUNT(*) INTO query_count FROM genomevault.pir_queries;
    SELECT COUNT(*) INTO proof_count FROM genomevault.zk_proofs;
    SELECT COUNT(*) INTO metric_count FROM metrics.api_requests;
    SELECT COUNT(*) INTO audit_count FROM audit.activity_log;

    RAISE NOTICE 'Development seed data created successfully:';
    RAISE NOTICE '  Users: %', user_count;
    RAISE NOTICE '  Hypervector encodings: %', encoding_count;
    RAISE NOTICE '  PIR queries: %', query_count;
    RAISE NOTICE '  ZK proofs: %', proof_count;
    RAISE NOTICE '  API request metrics: %', metric_count;
    RAISE NOTICE '  Audit log entries: %', audit_count;
END $$;
