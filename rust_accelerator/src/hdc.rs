// Hyperdimensional Computing operations
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::f32;
use blake3;

/// Compute cosine similarity between two vectors using SIMD when available
pub fn compute_cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let chunks = 8; // Process 8 elements at a time for SIMD
    let len = a.len();
    
    let (dot, norm_a_sq, norm_b_sq) = (0..len)
        .into_par_iter()
        .chunks(chunks)
        .map(|chunk| {
            let mut local_dot = 0.0f32;
            let mut local_norm_a = 0.0f32;
            let mut local_norm_b = 0.0f32;
            
            for i in chunk {
                if i < len {
                    let a_val = a[i];
                    let b_val = b[i];
                    local_dot += a_val * b_val;
                    local_norm_a += a_val * a_val;
                    local_norm_b += b_val * b_val;
                }
            }
            
            (local_dot, local_norm_a, local_norm_b)
        })
        .reduce(
            || (0.0, 0.0, 0.0),
            |acc, x| (acc.0 + x.0, acc.1 + x.1, acc.2 + x.2)
        );
    
    dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt() + 1e-10)
}

/// Encode a genomic variant into a hypervector
pub fn encode_variant_to_hypervector(
    chromosome: u8,
    position: u32,
    ref_allele: &str,
    alt_allele: &str,
    dimension: usize,
) -> Result<Vec<f32>, String> {
    // Create unique hash for variant
    let variant_str = format!("chr{}:{}:{}>{}", chromosome, position, ref_allele, alt_allele);
    let hash = blake3::hash(variant_str.as_bytes());
    let hash_bytes = hash.as_bytes();
    
    // Generate hypervector from hash
    let mut hypervector = vec![0.0f32; dimension];
    let mut rng = XorShiftRng::from_bytes(hash_bytes);
    
    // Use hash to determine sparse activation pattern
    let sparsity = 0.1; // 10% active
    let num_active = (dimension as f32 * sparsity) as usize;
    
    // Generate random indices for active positions
    let mut indices: Vec<usize> = (0..dimension).collect();
    shuffle_array(&mut indices, &mut rng);
    
    // Set active positions
    for i in 0..num_active {
        let idx = indices[i];
        // Random value between -1 and 1
        hypervector[idx] = rng.next_f32() * 2.0 - 1.0;
    }
    
    // Normalize
    let norm: f32 = hypervector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut hypervector {
            *val /= norm;
        }
    }
    
    Ok(hypervector)
}

/// Batch encode multiple variants
pub fn batch_encode_variants_to_hypervectors(
    chromosomes: &ArrayView1<u8>,
    positions: &ArrayView1<u32>,
    dimension: usize,
) -> Result<Vec<f32>, String> {
    let n = chromosomes.len();
    let mut result = vec![0.0f32; n * dimension];
    
    result.par_chunks_mut(dimension)
        .enumerate()
        .try_for_each(|(i, chunk)| {
            let hv = encode_variant_to_hypervector(
                chromosomes[i],
                positions[i],
                "N", // placeholder
                "N", // placeholder
                dimension,
            )?;
            chunk.copy_from_slice(&hv);
            Ok::<(), String>(())
        })?;
    
    Ok(result)
}

/// Compute sparse dot product
pub fn sparse_dot_product(
    indices: &ArrayView1<usize>,
    values: &ArrayView1<f32>,
    dense: &ArrayView1<f32>,
) -> Result<f32, String> {
    let result = indices.iter()
        .zip(values.iter())
        .map(|(&idx, &val)| {
            if idx < dense.len() {
                val * dense[idx]
            } else {
                0.0
            }
        })
        .sum();
    
    Ok(result)
}

/// K-nearest neighbors search
pub fn knn_search(
    database: &ArrayView2<f32>,
    query: &ArrayView1<f32>,
    k: usize,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    if k > database.nrows() {
        return Err("k larger than database size".to_string());
    }
    
    // Compute all similarities in parallel
    let mut similarities: Vec<(usize, f32)> = database
        .axis_iter(Axis(0))
        .enumerate()
        .par_bridge()
        .map(|(idx, row)| {
            let sim = compute_cosine_similarity(&row, query);
            (idx, sim)
        })
        .collect();
    
    // Sort by similarity (descending)
    similarities.par_sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Take top k
    let indices: Vec<usize> = similarities.iter().take(k).map(|&(idx, _)| idx).collect();
    let distances: Vec<f32> = similarities.iter().take(k).map(|&(_, sim)| sim).collect();
    
    Ok((indices, distances))
}

// Simple XorShift RNG for deterministic random numbers from hash
struct XorShiftRng {
    state: [u32; 4],
}

impl XorShiftRng {
    fn from_bytes(bytes: &[u8]) -> Self {
        let mut state = [0u32; 4];
        for i in 0..4 {
            let offset = i * 4;
            state[i] = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
        }
        XorShiftRng { state }
    }
    
    fn next(&mut self) -> u32 {
        let t = self.state[3];
        let s = self.state[0];
        self.state[3] = self.state[2];
        self.state[2] = self.state[1];
        self.state[1] = s;
        
        let t = t ^ (t << 11);
        let t = t ^ (t >> 8);
        self.state[0] = t ^ s ^ (s >> 19);
        
        self.state[0]
    }
    
    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u32::MAX as f32)
    }
}

fn shuffle_array<T>(array: &mut [T], rng: &mut XorShiftRng) {
    let len = array.len();
    for i in (1..len).rev() {
        let j = (rng.next() as usize) % (i + 1);
        array.swap(i, j);
    }
}