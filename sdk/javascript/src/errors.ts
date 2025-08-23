/**
 * Error classes for GenomeVault JavaScript SDK
 */

import { AxiosError } from 'axios';
import { APIError, ErrorDetail } from './types';

/**
 * Base class for all GenomeVault API errors
 */
export class GenomeVaultAPIError extends Error {
  public readonly response?: any;
  public readonly requestId?: string;
  public readonly errorCode?: string;
  public readonly statusCode?: number;
  public readonly headers?: Record<string, string>;
  public readonly details?: Record<string, any>;
  public readonly traceId?: string;

  constructor(
    message: string,
    options: {
      response?: any;
      requestId?: string;
      errorCode?: string;
      statusCode?: number;
      headers?: Record<string, string>;
      details?: Record<string, any>;
      traceId?: string;
    } = {}
  ) {
    super(message);
    this.name = this.constructor.name;
    this.response = options.response;
    this.requestId = options.requestId;
    this.errorCode = options.errorCode;
    this.statusCode = options.statusCode;
    this.headers = options.headers;
    this.details = options.details;
    this.traceId = options.traceId;

    // Maintain proper stack trace for where our error was thrown (Node.js only)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Convert error to JSON representation
   */
  toJSON(): Record<string, any> {
    return {
      name: this.name,
      message: this.message,
      errorCode: this.errorCode,
      statusCode: this.statusCode,
      requestId: this.requestId,
      traceId: this.traceId,
      details: this.details,
    };
  }

  /**
   * String representation with additional context
   */
  toString(): string {
    const parts = [this.message];

    if (this.errorCode) {
      parts.push(`Code: ${this.errorCode}`);
    }

    if (this.requestId) {
      parts.push(`Request ID: ${this.requestId}`);
    }

    if (this.statusCode) {
      parts.push(`HTTP ${this.statusCode}`);
    }

    return parts.join(' | ');
  }
}

/**
 * Authentication error (401)
 */
export class AuthenticationError extends GenomeVaultAPIError {
  constructor(message: string = 'Authentication required', options: any = {}) {
    super(message, { ...options, statusCode: 401 });
  }
}

/**
 * Authorization error (403)
 */
export class AuthorizationError extends GenomeVaultAPIError {
  constructor(message: string = 'Insufficient permissions', options: any = {}) {
    super(message, { ...options, statusCode: 403 });
  }
}

/**
 * Validation error (422)
 */
export class ValidationError extends GenomeVaultAPIError {
  public readonly validationErrors: ErrorDetail[];

  constructor(
    message: string = 'Request validation failed',
    options: { validationErrors?: ErrorDetail[] } & any = {}
  ) {
    super(message, { ...options, statusCode: 422 });
    this.validationErrors = options.validationErrors || [];
  }

  /**
   * Get validation errors as a formatted string
   */
  getValidationErrorsString(): string {
    if (this.validationErrors.length === 0) {
      return '';
    }

    const errorStrings = this.validationErrors.map(error => {
      const field = error.field || 'unknown';
      return `${field}: ${error.message}`;
    });

    return errorStrings.join(', ');
  }

  toString(): string {
    const baseString = super.toString();
    const validationErrorsString = this.getValidationErrorsString();

    if (validationErrorsString) {
      return `${baseString} | Errors: ${validationErrorsString}`;
    }

    return baseString;
  }

  toJSON(): Record<string, any> {
    return {
      ...super.toJSON(),
      validationErrors: this.validationErrors,
    };
  }
}

/**
 * Rate limit error (429)
 */
export class RateLimitError extends GenomeVaultAPIError {
  public readonly limit?: number;
  public readonly remaining?: number;
  public readonly resetTime?: number;
  public readonly retryAfter?: number;

  constructor(
    message: string = 'Rate limit exceeded',
    options: {
      limit?: number;
      remaining?: number;
      resetTime?: number;
      retryAfter?: number;
    } & any = {}
  ) {
    super(message, { ...options, statusCode: 429 });
    this.limit = options.limit;
    this.remaining = options.remaining;
    this.resetTime = options.resetTime;
    this.retryAfter = options.retryAfter;
  }

  toString(): string {
    const parts = [super.toString()];

    if (this.retryAfter) {
      parts.push(`Retry after: ${this.retryAfter}s`);
    }

    if (this.limit !== undefined && this.remaining !== undefined) {
      parts.push(`Limit: ${this.remaining}/${this.limit}`);
    }

    return parts.join(' | ');
  }

  toJSON(): Record<string, any> {
    return {
      ...super.toJSON(),
      limit: this.limit,
      remaining: this.remaining,
      resetTime: this.resetTime,
      retryAfter: this.retryAfter,
    };
  }
}

/**
 * Service unavailable error (503)
 */
export class ServiceUnavailableError extends GenomeVaultAPIError {
  public readonly retryAfter?: number;

  constructor(
    message: string = 'Service temporarily unavailable',
    options: { retryAfter?: number } & any = {}
  ) {
    super(message, { ...options, statusCode: 503 });
    this.retryAfter = options.retryAfter;
  }

  toString(): string {
    const baseString = super.toString();

    if (this.retryAfter) {
      return `${baseString} | Retry after: ${this.retryAfter}s`;
    }

    return baseString;
  }

  toJSON(): Record<string, any> {
    return {
      ...super.toJSON(),
      retryAfter: this.retryAfter,
    };
  }
}

/**
 * Not found error (404)
 */
export class NotFoundError extends GenomeVaultAPIError {
  constructor(message: string = 'Resource not found', options: any = {}) {
    super(message, { ...options, statusCode: 404 });
  }
}

/**
 * Conflict error (409)
 */
export class ConflictError extends GenomeVaultAPIError {
  constructor(message: string = 'Resource conflict', options: any = {}) {
    super(message, { ...options, statusCode: 409 });
  }
}

/**
 * Timeout error
 */
export class TimeoutError extends GenomeVaultAPIError {
  constructor(message: string = 'Request timed out', options: any = {}) {
    super(message, options);
  }
}

/**
 * Network error
 */
export class NetworkError extends GenomeVaultAPIError {
  constructor(message: string = 'Network error', options: any = {}) {
    super(message, options);
  }
}

// Domain-specific errors

/**
 * Genomic data validation error
 */
export class GenomicDataError extends ValidationError {
  constructor(message: string = 'Invalid genomic data', options: any = {}) {
    super(message, options);
  }
}

/**
 * PIR query execution error
 */
export class PIRQueryError extends GenomeVaultAPIError {
  constructor(message: string = 'PIR query failed', options: any = {}) {
    super(message, options);
  }
}

/**
 * Zero-knowledge proof generation error
 */
export class ProofGenerationError extends GenomeVaultAPIError {
  constructor(message: string = 'Proof generation failed', options: any = {}) {
    super(message, options);
  }
}

/**
 * Clinical analysis error
 */
export class ClinicalAnalysisError extends GenomeVaultAPIError {
  constructor(message: string = 'Clinical analysis failed', options: any = {}) {
    super(message, options);
  }
}

/**
 * Protected Health Information detected error
 */
export class PHIDetectedError extends ValidationError {
  public readonly phiFields: string[];

  constructor(
    message: string = 'Protected health information detected',
    options: { phiFields?: string[] } & any = {}
  ) {
    super(message, options);
    this.phiFields = options.phiFields || [];
  }

  toString(): string {
    const baseString = super.toString();

    if (this.phiFields.length > 0) {
      return `${baseString} | Fields: ${this.phiFields.join(', ')}`;
    }

    return baseString;
  }

  toJSON(): Record<string, any> {
    return {
      ...super.toJSON(),
      phiFields: this.phiFields,
    };
  }
}

/**
 * Create appropriate error from HTTP response
 */
export function createErrorFromResponse(axiosError: AxiosError): GenomeVaultAPIError {
  const response = axiosError.response;
  const message = axiosError.message || 'Request failed';

  // Handle network errors
  if (!response) {
    if (axiosError.code === 'ECONNABORTED') {
      return new TimeoutError('Request timed out');
    }
    return new NetworkError(`Network error: ${message}`);
  }

  // Extract error data from response
  let errorData: APIError | null = null;
  try {
    errorData = response.data as APIError;
  } catch {
    // Response body is not JSON or doesn't match expected format
  }

  const statusCode = response.status;
  const headers = response.headers as Record<string, string>;
  const errorMessage = errorData?.message || message;
  const requestId = errorData?.requestId || headers['x-request-id'];
  const errorCode = errorData?.code;
  const traceId = errorData?.traceId || headers['x-trace-id'];
  const details = errorData?.details;

  const baseOptions = {
    response,
    requestId,
    errorCode,
    statusCode,
    headers,
    details,
    traceId,
  };

  // Map status codes to specific error types
  switch (statusCode) {
    case 401:
      return new AuthenticationError(errorMessage, baseOptions);

    case 403:
      return new AuthorizationError(errorMessage, baseOptions);

    case 404:
      return new NotFoundError(errorMessage, baseOptions);

    case 409:
      return new ConflictError(errorMessage, baseOptions);

    case 422:
      return new ValidationError(errorMessage, {
        ...baseOptions,
        validationErrors: errorData?.errors || [],
      });

    case 429:
      return new RateLimitError(errorMessage, {
        ...baseOptions,
        limit: headers['x-ratelimit-limit'] ? parseInt(headers['x-ratelimit-limit']) : undefined,
        remaining: headers['x-ratelimit-remaining'] ? parseInt(headers['x-ratelimit-remaining']) : undefined,
        resetTime: headers['x-ratelimit-reset'] ? parseInt(headers['x-ratelimit-reset']) : undefined,
        retryAfter: headers['retry-after'] ? parseInt(headers['retry-after']) : undefined,
      });

    case 503:
      return new ServiceUnavailableError(errorMessage, {
        ...baseOptions,
        retryAfter: headers['retry-after'] ? parseInt(headers['retry-after']) : undefined,
      });

    default:
      // Map error codes to domain-specific errors
      if (errorCode) {
        switch (errorCode) {
          case 'GV_PHI_DETECTED':
            const phiFields = errorData?.errors?.map(e => e.field).filter(Boolean) || [];
            return new PHIDetectedError(errorMessage, { ...baseOptions, phiFields });

          case 'GV_PIR_QUERY_FAILED':
            return new PIRQueryError(errorMessage, baseOptions);

          case 'GV_PROOF_VERIFICATION_FAILED':
            return new ProofGenerationError(errorMessage, baseOptions);

          case 'GV_CLINICAL_DATA_INCOMPLETE':
          case 'GV_CONSENT_REQUIRED':
            return new ClinicalAnalysisError(errorMessage, baseOptions);

          case 'GV_INVALID_GENOMIC_COORDINATE':
          case 'GV_INVALID_VARIANT_FORMAT':
            return new GenomicDataError(errorMessage, baseOptions);
        }
      }

      return new GenomeVaultAPIError(errorMessage, baseOptions);
  }
}

/**
 * Error handler utility class
 */
export class ErrorHandler {
  /**
   * Check if error is retryable
   */
  static isRetryable(error: GenomeVaultAPIError): boolean {
    // Retry on 5xx errors and specific 4xx errors
    if (!error.statusCode) {
      return true; // Network errors are retryable
    }

    if (error.statusCode >= 500) {
      return true; // Server errors are retryable
    }

    if (error instanceof RateLimitError) {
      return true; // Rate limit errors are retryable
    }

    return false;
  }

  /**
   * Get retry delay for error
   */
  static getRetryDelay(error: GenomeVaultAPIError, attempt: number): number {
    if (error instanceof RateLimitError && error.retryAfter) {
      return error.retryAfter * 1000; // Convert to milliseconds
    }

    if (error instanceof ServiceUnavailableError && error.retryAfter) {
      return error.retryAfter * 1000;
    }

    // Exponential backoff with jitter
    const baseDelay = 1000; // 1 second
    const exponentialDelay = baseDelay * Math.pow(2, attempt - 1);
    const jitter = Math.random() * 1000; // Up to 1 second of jitter

    return Math.min(exponentialDelay + jitter, 30000); // Cap at 30 seconds
  }

  /**
   * Format error for logging
   */
  static formatError(error: GenomeVaultAPIError): Record<string, any> {
    return {
      name: error.name,
      message: error.message,
      errorCode: error.errorCode,
      statusCode: error.statusCode,
      requestId: error.requestId,
      traceId: error.traceId,
      timestamp: new Date().toISOString(),
    };
  }
}
