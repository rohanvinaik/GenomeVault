/**
 * Utility functions for GenomeVault SDK
 */

import { GenomicVariant } from './types';
import { ValidationError } from './errors';

/**
 * Generate a UUID v4
 */
export function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }

  // Fallback implementation
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Validate genomic variant format
 */
export function validateGenomicVariant(variant: GenomicVariant): void {
  if (!variant) {
    throw new ValidationError('Variant cannot be null or undefined');
  }

  // Validate chromosome
  if (!variant.chrom) {
    throw new ValidationError('Chromosome is required');
  }

  const normalizedChrom = variant.chrom.replace(/^chr/i, '').toUpperCase();
  const validChroms = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', 'X', 'Y', 'M', 'MT'
  ];

  if (!validChroms.includes(normalizedChrom)) {
    throw new ValidationError(`Invalid chromosome: ${variant.chrom}`);
  }

  // Validate position
  if (!variant.pos || variant.pos < 1) {
    throw new ValidationError('Position must be a positive integer');
  }

  // Validate reference allele
  if (!variant.ref) {
    throw new ValidationError('Reference allele is required');
  }

  if (!isValidNucleotideSequence(variant.ref)) {
    throw new ValidationError(`Invalid reference allele: ${variant.ref}`);
  }

  // Validate alternative allele
  if (!variant.alt) {
    throw new ValidationError('Alternative allele is required');
  }

  if (!isValidNucleotideSequence(variant.alt)) {
    throw new ValidationError(`Invalid alternative allele: ${variant.alt}`);
  }

  // Validate quality score if provided
  if (variant.quality !== undefined) {
    if (variant.quality < 0 || variant.quality > 100) {
      throw new ValidationError('Quality score must be between 0 and 100');
    }
  }

  // Validate impact if provided
  if (variant.impact !== undefined) {
    const validImpacts = [
      'missense', 'nonsense', 'synonymous', 'frameshift',
      'splice_site', 'intron', 'intergenic'
    ];

    if (!validImpacts.includes(variant.impact)) {
      throw new ValidationError(`Invalid impact: ${variant.impact}`);
    }
  }
}

/**
 * Check if string is a valid nucleotide sequence
 */
export function isValidNucleotideSequence(sequence: string): boolean {
  if (!sequence) {
    return false;
  }

  return /^[ATCGN]+$/i.test(sequence);
}

/**
 * Normalize chromosome name
 */
export function normalizeChromosome(chrom: string): string {
  return chrom.replace(/^chr/i, '').toUpperCase();
}

/**
 * Convert genomic coordinate to string representation
 */
export function formatGenomicCoordinate(chrom: string, pos: number): string {
  const normalizedChrom = normalizeChromosome(chrom);
  return `${normalizedChrom}:${pos}`;
}

/**
 * Parse genomic coordinate string
 */
export function parseGenomicCoordinate(coordinate: string): { chrom: string; pos: number } {
  const match = coordinate.match(/^(chr)?([0-9XYM]{1,2}):(\d+)$/i);

  if (!match) {
    throw new ValidationError(`Invalid genomic coordinate format: ${coordinate}`);
  }

  return {
    chrom: normalizeChromosome(match[2]),
    pos: parseInt(match[3])
  };
}

/**
 * Validate SHA-256 hash format
 */
export function isValidSHA256(hash: string): boolean {
  return /^[a-f0-9]{64}$/i.test(hash);
}

/**
 * Validate base64 string
 */
export function isValidBase64(str: string): boolean {
  try {
    // Check if string is valid base64
    if (typeof Buffer !== 'undefined') {
      // Node.js
      return Buffer.from(str, 'base64').toString('base64') === str;
    } else {
      // Browser
      return btoa(atob(str)) === str;
    }
  } catch {
    return false;
  }
}

/**
 * Format bytes to human readable size
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Deep clone an object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as T;
  }

  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as T;
  }

  if (typeof obj === 'object') {
    const cloned: any = {};
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = deepClone(obj[key]);
      }
    }
    return cloned;
  }

  return obj;
}

/**
 * Debounce function execution
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | number | null = null;

  return function (...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout !== null) {
      clearTimeout(timeout as number);
    }

    timeout = setTimeout(later, wait);
  };
}

/**
 * Throttle function execution
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean = false;

  return function (...args: Parameters<T>) {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

/**
 * Retry function with exponential backoff
 */
export async function retry<T>(
  fn: () => Promise<T>,
  options: {
    maxAttempts?: number;
    baseDelay?: number;
    maxDelay?: number;
    retryIf?: (error: any) => boolean;
  } = {}
): Promise<T> {
  const {
    maxAttempts = 3,
    baseDelay = 1000,
    maxDelay = 30000,
    retryIf = () => true
  } = options;

  let lastError: any;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === maxAttempts || !retryIf(error)) {
        throw error;
      }

      // Calculate delay with exponential backoff
      const delay = Math.min(baseDelay * Math.pow(2, attempt - 1), maxDelay);

      // Add jitter to prevent thundering herd
      const jitteredDelay = delay + Math.random() * 1000;

      await sleep(jitteredDelay);
    }
  }

  throw lastError;
}

/**
 * Chunk array into smaller arrays
 */
export function chunk<T>(array: T[], size: number): T[][] {
  const chunks: T[][] = [];

  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }

  return chunks;
}

/**
 * Create a promise that resolves after specified time
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Create a timeout promise that rejects after specified time
 */
export function timeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(`Operation timed out after ${ms}ms`)), ms);
  });

  return Promise.race([promise, timeoutPromise]);
}

/**
 * Check if code is running in browser environment
 */
export function isBrowser(): boolean {
  return typeof window !== 'undefined' && typeof window.document !== 'undefined';
}

/**
 * Check if code is running in Node.js environment
 */
export function isNode(): boolean {
  return typeof process !== 'undefined' && process.versions && process.versions.node;
}

/**
 * Safe JSON parse with default value
 */
export function safeJSONParse<T>(str: string, defaultValue: T): T {
  try {
    return JSON.parse(str);
  } catch {
    return defaultValue;
  }
}

/**
 * Safe JSON stringify
 */
export function safeJSONStringify(obj: any): string {
  try {
    return JSON.stringify(obj);
  } catch {
    return '{}';
  }
}

/**
 * URL builder utility
 */
export class URLBuilder {
  private baseUrl: string;
  private pathSegments: string[] = [];
  private queryParams: Record<string, string | number | boolean> = {};

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  path(...segments: string[]): this {
    this.pathSegments.push(...segments.map(s => encodeURIComponent(s)));
    return this;
  }

  query(params: Record<string, string | number | boolean | undefined>): this {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        this.queryParams[key] = value;
      }
    });
    return this;
  }

  build(): string {
    let url = this.baseUrl;

    if (this.pathSegments.length > 0) {
      url += '/' + this.pathSegments.join('/');
    }

    const queryString = Object.entries(this.queryParams)
      .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`)
      .join('&');

    if (queryString) {
      url += '?' + queryString;
    }

    return url;
  }
}

/**
 * Event emitter for SDK events
 */
export class EventEmitter {
  private events: Map<string, Function[]> = new Map();

  on(event: string, listener: Function): this {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event)!.push(listener);
    return this;
  }

  off(event: string, listener: Function): this {
    const listeners = this.events.get(event);
    if (listeners) {
      const index = listeners.indexOf(listener);
      if (index !== -1) {
        listeners.splice(index, 1);
      }
    }
    return this;
  }

  emit(event: string, ...args: any[]): boolean {
    const listeners = this.events.get(event);
    if (listeners && listeners.length > 0) {
      listeners.forEach(listener => listener(...args));
      return true;
    }
    return false;
  }

  removeAllListeners(event?: string): this {
    if (event) {
      this.events.delete(event);
    } else {
      this.events.clear();
    }
    return this;
  }
}
