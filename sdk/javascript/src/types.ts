/**
 * TypeScript type definitions for GenomeVault API
 */

/**
 * Configuration options for the GenomeVault client
 */
export interface ClientConfig {
  /** Base URL for the GenomeVault API */
  baseUrl?: string;
  /** API key for authentication */
  apiKey?: string;
  /** OAuth2 bearer token for authentication */
  oauthToken?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Maximum number of retries for failed requests */
  maxRetries?: number;
  /** Backoff factor for retries */
  retryBackoff?: number;
  /** Additional headers to include with all requests */
  headers?: Record<string, string>;
}

/**
 * Genomic variant representation
 */
export interface GenomicVariant {
  /** Chromosome (1-22, X, Y, M) */
  chrom: string;
  /** Genomic position (1-based) */
  pos: number;
  /** Reference allele */
  ref: string;
  /** Alternative allele */
  alt: string;
  /** Predicted functional impact */
  impact?: 'missense' | 'nonsense' | 'synonymous' | 'frameshift' | 'splice_site' | 'intron' | 'intergenic';
  /** Variant quality score (0-100) */
  quality?: number;
}

/**
 * Request model for hypervector encoding
 */
export interface EncodeRequest {
  /** Numeric feature array (alternative to variants) */
  numeric?: number[];
  /** Genomic variants to encode (alternative to numeric) */
  variants?: GenomicVariant[];
  /** Hypervector dimension */
  dim?: number;
  /** Return binary (-1/+1) or continuous values */
  binary?: boolean;
}

/**
 * Response model for hypervector encoding
 */
export interface EncodeResponse {
  /** Hypervector dimension */
  dim: number;
  /** Whether vector contains binary values */
  binary: boolean;
  /** Encoded hypervector */
  vector: number[];
  /** Privacy guarantee level */
  privacyLevel?: 'k-anonymous' | 'differential_private' | 'information_theoretic';
  /** Data compression ratio achieved */
  compressionRatio?: number;
}

/**
 * Request model for PIR queries
 */
export interface PIRQueryRequest {
  /** Index to query (kept private from server) */
  index: number;
  /** Unique query identifier for tracking */
  queryId?: string;
  /** Query timeout in seconds */
  timeoutSeconds?: number;
}

/**
 * Response model for PIR queries
 */
export interface PIRQueryResponse {
  /** Queried index (for client verification) */
  index: number;
  /** Base64-encoded retrieved item */
  itemBase64: string;
  /** Cryptographic proof of privacy preservation */
  privacyProof?: string;
  /** Query execution time in milliseconds */
  queryTimeMs?: number;
}

/**
 * Request model for zero-knowledge proofs
 */
export interface ProofRequest {
  /** Type of proof to generate */
  proofType: 'genomic' | 'clinical' | 'research';
  /** Public inputs visible to verifiers */
  publicInputs: Record<string, any>;
  /** SHA-256 hash of private inputs */
  privateInputsHash: string;
  /** Circuit-specific parameters */
  circuitParams?: Record<string, any>;
}

/**
 * Response model for zero-knowledge proofs
 */
export interface ProofResponse {
  /** Unique proof identifier */
  proofId: string;
  /** Hex-encoded zk-SNARK proof */
  proofData: string;
  /** Verification key for proof validation */
  verificationKey: string;
  /** Public signals from the proof */
  publicSignals: string[];
  /** Proof validity period in hours */
  validityPeriodHours?: number;
}

/**
 * Clinical variant for analysis
 */
export interface ClinicalVariant {
  /** Gene symbol (HGNC approved) */
  gene: string;
  /** HGVS notation variant */
  variant: string;
  /** Clinical variant classification */
  classification?: 'pathogenic' | 'likely_pathogenic' | 'uncertain_significance' | 'likely_benign' | 'benign';
  /** Evidence level (ClinGen guidelines) */
  evidenceLevel?: 'A' | 'B' | 'C' | 'D';
}

/**
 * Request model for clinical analysis
 */
export interface ClinicalAnalysisRequest {
  /** SHA-256 hash of patient identifier */
  patientIdHash: string;
  /** Clinical variants for analysis */
  variants: ClinicalVariant[];
  /** Type of clinical analysis */
  analysisType: 'risk_assessment' | 'pharmacogenomics' | 'carrier_screening' | 'diagnostic';
  /** Population reference database */
  populationReference?: 'gnomAD' | '1000G' | 'ESP' | 'ExAC';
  /** Hash of patient consent documentation */
  consentHash?: string;
}

/**
 * Response model for clinical analysis
 */
export interface ClinicalAnalysisResponse {
  /** Unique analysis identifier */
  analysisId: string;
  /** Calculated risk score (0-1) */
  riskScore: number;
  /** 95% confidence interval [lower, upper] */
  confidenceInterval: [number, number];
  /** Clinical recommendations */
  recommendations: string[];
  /** Cryptographic hash of audit trail */
  auditTrailHash: string;
  /** Differential privacy parameter used */
  differentialPrivacyEpsilon?: number;
}

/**
 * Response model for health checks
 */
export interface HealthResponse {
  /** Overall system status */
  status: 'healthy' | 'degraded' | 'unhealthy';
  /** Health check timestamp */
  timestamp: string;
  /** API version */
  version: string;
  /** Individual service health status */
  services: Record<string, 'healthy' | 'unhealthy'>;
}

/**
 * Standard API error response
 */
export interface APIError {
  /** Error type classification */
  type: string;
  /** Machine-readable error code */
  code: string;
  /** Human-readable error message (PHI-safe) */
  message: string;
  /** Additional error context */
  details?: Record<string, any>;
  /** Field-level validation errors */
  errors?: ErrorDetail[];
  /** Unique request identifier for support */
  requestId: string;
  /** Error timestamp */
  timestamp: string;
  /** Distributed tracing identifier */
  traceId?: string;
}

/**
 * Individual validation error detail
 */
export interface ErrorDetail {
  /** Field that caused the error */
  field?: string;
  /** Error message for this field */
  message: string;
  /** Error code for this field */
  code: string;
  /** Invalid value (if safe to expose) */
  value?: any;
  /** List of allowed values */
  allowedValues?: string[];
}

/**
 * Batch encoding request
 */
export interface BatchEncodeRequest {
  /** List of encoding requests */
  requests: EncodeRequest[];
  /** Maximum concurrent requests */
  maxConcurrent?: number;
}

/**
 * Batch encoding response
 */
export interface BatchEncodeResponse {
  /** List of successful encoding responses */
  results: EncodeResponse[];
  /** Number of successful requests */
  successCount: number;
  /** Number of failed requests */
  errorCount: number;
  /** List of errors for failed requests */
  errors: APIError[];
}

/**
 * Pagination parameters
 */
export interface PaginationParams {
  /** Page number (1-based) */
  page?: number;
  /** Number of items per page */
  pageSize?: number;
  /** Field to sort by */
  sortBy?: string;
  /** Sort order */
  sortOrder?: 'asc' | 'desc';
}

/**
 * Paginated response wrapper
 */
export interface PaginatedResponse<T> {
  /** Response data items */
  data: T[];
  /** Current page number */
  page: number;
  /** Items per page */
  pageSize: number;
  /** Total number of items */
  totalCount: number;
  /** Total number of pages */
  totalPages: number;
  /** Whether there is a next page */
  hasNext: boolean;
  /** Whether there is a previous page */
  hasPrevious: boolean;
}

/**
 * HTTP request options
 */
export interface RequestOptions {
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Additional headers */
  headers?: Record<string, string>;
  /** Query parameters */
  params?: Record<string, any>;
  /** Request body */
  data?: any;
  /** Whether to retry on failure */
  retry?: boolean;
}

/**
 * Upload progress callback
 */
export type ProgressCallback = (progress: {
  /** Bytes uploaded */
  loaded: number;
  /** Total bytes */
  total: number;
  /** Progress percentage (0-100) */
  percent: number;
}) => void;

/**
 * File upload options
 */
export interface UploadOptions extends RequestOptions {
  /** Progress callback for file uploads */
  onProgress?: ProgressCallback;
  /** Content type override */
  contentType?: string;
}

/**
 * Webhook payload for real-time updates
 */
export interface WebhookPayload {
  /** Event type */
  event: string;
  /** Event timestamp */
  timestamp: string;
  /** Event data */
  data: Record<string, any>;
  /** Webhook signature for verification */
  signature?: string;
}

/**
 * Rate limit information
 */
export interface RateLimitInfo {
  /** Request limit per time window */
  limit: number;
  /** Requests remaining in current window */
  remaining: number;
  /** Time when rate limit resets (Unix timestamp) */
  reset: number;
  /** Time to wait before next request (seconds) */
  retryAfter?: number;
}

/**
 * Authentication token information
 */
export interface TokenInfo {
  /** Access token */
  accessToken: string;
  /** Token type (usually 'Bearer') */
  tokenType: string;
  /** Token expiration time (Unix timestamp) */
  expiresAt: number;
  /** Refresh token */
  refreshToken?: string;
  /** Token scopes */
  scopes: string[];
}

/**
 * API version information
 */
export interface VersionInfo {
  /** API version */
  version: string;
  /** Version status */
  status: 'active' | 'deprecated' | 'sunset';
  /** Deprecation date */
  deprecationDate?: string;
  /** Sunset date */
  sunsetDate?: string;
  /** Successor version */
  successorVersion?: string;
}
