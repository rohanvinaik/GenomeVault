/**
 * GenomeVault TypeScript/JavaScript Client
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError, AxiosRequestConfig } from 'axios';
import {
  ClientConfig,
  GenomicVariant,
  EncodeRequest,
  EncodeResponse,
  PIRQueryRequest,
  PIRQueryResponse,
  ProofRequest,
  ProofResponse,
  ClinicalAnalysisRequest,
  ClinicalAnalysisResponse,
  HealthResponse,
  APIError,
  BatchEncodeRequest,
  BatchEncodeResponse,
  RequestOptions,
  RateLimitInfo,
} from './types';
import {
  GenomeVaultAPIError,
  AuthenticationError,
  ValidationError,
  RateLimitError,
  ServiceUnavailableError,
  createErrorFromResponse,
} from './errors';
import { validateGenomicVariant, generateUUID, sleep } from './utils';

/**
 * GenomeVault API Client for TypeScript/JavaScript
 */
export class GenomeVaultClient {
  private readonly client: AxiosInstance;
  private readonly config: Required<ClientConfig>;

  /**
   * Create a new GenomeVault client
   */
  constructor(config: ClientConfig = {}) {
    this.config = {
      baseUrl: config.baseUrl || 'https://api.genomevault.io',
      apiKey: config.apiKey || '',
      oauthToken: config.oauthToken || '',
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
      retryBackoff: config.retryBackoff || 1000,
      headers: config.headers || {},
    };

    // Create axios instance
    this.client = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'genomevault-js-sdk/1.0.0',
        ...this.config.headers,
      },
    });

    // Setup authentication
    this.setupAuthentication();

    // Setup request/response interceptors
    this.setupInterceptors();
  }

  /**
   * Setup authentication headers
   */
  private setupAuthentication(): void {
    if (this.config.apiKey) {
      this.client.defaults.headers['X-API-Key'] = this.config.apiKey;
    } else if (this.config.oauthToken) {
      this.client.defaults.headers.Authorization = `Bearer ${this.config.oauthToken}`;
    }
  }

  /**
   * Setup axios interceptors for error handling and retries
   */
  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add trace ID for request tracking
        if (!config.headers['X-Trace-Id']) {
          config.headers['X-Trace-Id'] = generateUUID();
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean; _retryCount?: number };

        // Handle rate limiting with retry
        if (error.response?.status === 429 && !originalRequest._retry) {
          originalRequest._retry = true;
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;

          if (originalRequest._retryCount <= this.config.maxRetries) {
            const retryAfter = error.response.headers['retry-after'] || this.config.retryBackoff;
            const delay = parseInt(retryAfter.toString()) * 1000;

            await sleep(delay);
            return this.client(originalRequest);
          }
        }

        // Handle server errors with exponential backoff
        if (
          error.response?.status &&
          error.response.status >= 500 &&
          !originalRequest._retry
        ) {
          originalRequest._retry = true;
          originalRequest._retryCount = (originalRequest._retryCount || 0) + 1;

          if (originalRequest._retryCount <= this.config.maxRetries) {
            const delay = this.config.retryBackoff * Math.pow(2, originalRequest._retryCount - 1);
            await sleep(delay);
            return this.client(originalRequest);
          }
        }

        // Convert axios error to GenomeVault error
        throw createErrorFromResponse(error);
      }
    );
  }

  /**
   * Make HTTP request with error handling
   */
  private async makeRequest<T>(
    method: 'get' | 'post' | 'put' | 'delete',
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const config: AxiosRequestConfig = {
      method,
      url: endpoint,
      timeout: options.timeout || this.config.timeout,
      headers: options.headers,
      params: options.params,
      data: options.data,
    };

    try {
      const response: AxiosResponse<T> = await this.client.request(config);
      return response.data;
    } catch (error) {
      if (error instanceof GenomeVaultAPIError) {
        throw error;
      }
      throw new GenomeVaultAPIError('Request failed', error);
    }
  }

  /**
   * Get rate limit information from response headers
   */
  private extractRateLimitInfo(headers: Record<string, string>): RateLimitInfo | null {
    const limit = headers['x-ratelimit-limit'];
    const remaining = headers['x-ratelimit-remaining'];
    const reset = headers['x-ratelimit-reset'];
    const retryAfter = headers['retry-after'];

    if (limit && remaining && reset) {
      return {
        limit: parseInt(limit),
        remaining: parseInt(remaining),
        reset: parseInt(reset),
        retryAfter: retryAfter ? parseInt(retryAfter) : undefined,
      };
    }

    return null;
  }

  // Health endpoints
  /**
   * Check system health status
   */
  async healthCheck(): Promise<HealthResponse> {
    return this.makeRequest<HealthResponse>('get', '/v1/health');
  }

  // Hypervector endpoints
  /**
   * Encode genomic variants into hypervectors
   */
  async encodeVariants(
    variants: GenomicVariant[],
    options: { dim?: number; binary?: boolean } = {}
  ): Promise<EncodeResponse> {
    // Validate variants
    variants.forEach((variant, index) => {
      try {
        validateGenomicVariant(variant);
      } catch (error) {
        throw new ValidationError(`Invalid variant at index ${index}: ${error.message}`);
      }
    });

    const request: EncodeRequest = {
      variants,
      dim: options.dim || 8192,
      binary: options.binary || false,
    };

    return this.makeRequest<EncodeResponse>('post', '/v1/hv/encode', { data: request });
  }

  /**
   * Encode numeric features into hypervectors
   */
  async encodeNumeric(
    numeric: number[],
    options: { dim?: number; binary?: boolean } = {}
  ): Promise<EncodeResponse> {
    if (!numeric || numeric.length === 0) {
      throw new ValidationError('Numeric array cannot be empty');
    }

    const request: EncodeRequest = {
      numeric,
      dim: options.dim || 8192,
      binary: options.binary || false,
    };

    return this.makeRequest<EncodeResponse>('post', '/v1/hv/encode', { data: request });
  }

  /**
   * Batch encode multiple requests
   */
  async batchEncode(requests: EncodeRequest[], maxConcurrent: number = 10): Promise<BatchEncodeResponse> {
    const batchRequest: BatchEncodeRequest = {
      requests,
      maxConcurrent,
    };

    return this.makeRequest<BatchEncodeResponse>('post', '/v1/hv/batch-encode', { data: batchRequest });
  }

  // PIR endpoints
  /**
   * Execute a Private Information Retrieval query
   */
  async pirQuery(
    index: number,
    options: { queryId?: string; timeoutSeconds?: number } = {}
  ): Promise<PIRQueryResponse> {
    if (index < 0) {
      throw new ValidationError('PIR query index must be non-negative');
    }

    const request: PIRQueryRequest = {
      index,
      queryId: options.queryId || generateUUID(),
      timeoutSeconds: options.timeoutSeconds || 30,
    };

    return this.makeRequest<PIRQueryResponse>('post', '/v1/pir/query', { data: request });
  }

  // Zero-knowledge proof endpoints
  /**
   * Generate a zero-knowledge proof
   */
  async generateProof(
    proofType: 'genomic' | 'clinical' | 'research',
    publicInputs: Record<string, any>,
    privateInputsHash: string,
    circuitParams: Record<string, any> = {}
  ): Promise<ProofResponse> {
    // Validate SHA-256 hash format
    if (!/^[a-f0-9]{64}$/i.test(privateInputsHash)) {
      throw new ValidationError('privateInputsHash must be a valid SHA-256 hash');
    }

    const request: ProofRequest = {
      proofType,
      publicInputs,
      privateInputsHash: privateInputsHash.toLowerCase(),
      circuitParams,
    };

    return this.makeRequest<ProofResponse>('post', '/v1/zk/prove', { data: request });
  }

  // Clinical endpoints
  /**
   * Perform clinical genomic analysis
   */
  async clinicalAnalysis(
    patientIdHash: string,
    variants: any[], // ClinicalVariant[] - using any to avoid circular dependency
    analysisType: 'risk_assessment' | 'pharmacogenomics' | 'carrier_screening' | 'diagnostic',
    options: {
      populationReference?: 'gnomAD' | '1000G' | 'ESP' | 'ExAC';
      consentHash?: string;
    } = {}
  ): Promise<ClinicalAnalysisResponse> {
    // Validate patient ID hash
    if (!/^[a-f0-9]{64}$/i.test(patientIdHash)) {
      throw new ValidationError('patientIdHash must be a valid SHA-256 hash');
    }

    // Validate consent hash if provided
    if (options.consentHash && !/^[a-f0-9]{64}$/i.test(options.consentHash)) {
      throw new ValidationError('consentHash must be a valid SHA-256 hash');
    }

    const request: ClinicalAnalysisRequest = {
      patientIdHash: patientIdHash.toLowerCase(),
      variants,
      analysisType,
      populationReference: options.populationReference || 'gnomAD',
      consentHash: options.consentHash?.toLowerCase(),
    };

    return this.makeRequest<ClinicalAnalysisResponse>('post', '/v1/clinical/analyze', { data: request });
  }

  // Convenience methods
  /**
   * Validate multiple genomic variants
   */
  validateVariants(variants: GenomicVariant[]): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    variants.forEach((variant, index) => {
      try {
        validateGenomicVariant(variant);
      } catch (error) {
        errors.push(`Variant ${index}: ${error.message}`);
      }
    });

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  /**
   * Convert VCF record to GenomicVariant
   */
  static vcfToVariant(vcfRecord: {
    CHROM: string;
    POS: number;
    REF: string;
    ALT: string;
    QUAL?: number;
  }): GenomicVariant {
    return {
      chrom: vcfRecord.CHROM.replace(/^chr/i, ''),
      pos: vcfRecord.POS,
      ref: vcfRecord.REF.toUpperCase(),
      alt: vcfRecord.ALT.toUpperCase(),
      quality: vcfRecord.QUAL,
    };
  }

  /**
   * Decode base64 PIR response
   */
  static decodePIRResponse(itemBase64: string): Uint8Array {
    try {
      // Handle both browser and Node.js environments
      if (typeof Buffer !== 'undefined') {
        // Node.js
        return new Uint8Array(Buffer.from(itemBase64, 'base64'));
      } else {
        // Browser
        const binaryString = atob(itemBase64);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes;
      }
    } catch (error) {
      throw new ValidationError('Invalid base64 encoding in PIR response');
    }
  }

  /**
   * Calculate SHA-256 hash
   */
  static async sha256(data: string): Promise<string> {
    if (typeof crypto !== 'undefined' && crypto.subtle) {
      // Browser environment
      const encoder = new TextEncoder();
      const dataArray = encoder.encode(data);
      const hashBuffer = await crypto.subtle.digest('SHA-256', dataArray);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    } else {
      // Node.js environment
      const crypto = require('crypto');
      return crypto.createHash('sha256').update(data).digest('hex');
    }
  }

  /**
   * Update authentication credentials
   */
  updateAuth(credentials: { apiKey?: string; oauthToken?: string }): void {
    if (credentials.apiKey) {
      this.config.apiKey = credentials.apiKey;
      this.client.defaults.headers['X-API-Key'] = credentials.apiKey;
      delete this.client.defaults.headers.Authorization;
    } else if (credentials.oauthToken) {
      this.config.oauthToken = credentials.oauthToken;
      this.client.defaults.headers.Authorization = `Bearer ${credentials.oauthToken}`;
      delete this.client.defaults.headers['X-API-Key'];
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<Required<ClientConfig>> {
    return { ...this.config };
  }

  /**
   * Set request timeout
   */
  setTimeout(timeout: number): void {
    this.config.timeout = timeout;
    this.client.defaults.timeout = timeout;
  }

  /**
   * Add custom header
   */
  setHeader(key: string, value: string): void {
    this.client.defaults.headers[key] = value;
  }

  /**
   * Remove custom header
   */
  removeHeader(key: string): void {
    delete this.client.defaults.headers[key];
  }
}
