/**
 * GenomeVault TypeScript/JavaScript SDK
 *
 * Privacy-preserving genomic computing platform client library.
 */

export { GenomeVaultClient } from './client';
export * from './types';
export * from './errors';
export * from './utils';

// Re-export commonly used types for convenience
export type {
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
  ClientConfig,
} from './types';

// Default export
export { GenomeVaultClient as default } from './client';
