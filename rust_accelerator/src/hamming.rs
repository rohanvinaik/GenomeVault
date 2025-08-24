// Hamming distance computations
use ndarray::{ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// Compute Hamming distance between two binary arrays
pub fn compute_hamming_distance(a: &ArrayView1<u8>, b: &ArrayView1<u8>) -> u32 {
    // Use parallel chunks for large arrays
    const CHUNK_SIZE: usize = 1024;

    if a.len() > CHUNK_SIZE * 4 {
        // Parallel version for large arrays
        a.as_slice()
            .unwrap()
            .par_chunks(CHUNK_SIZE)
            .zip(b.as_slice().unwrap().par_chunks(CHUNK_SIZE))
            .map(|(chunk_a, chunk_b)| {
                chunk_a.iter()
                    .zip(chunk_b.iter())
                    .map(|(&x, &y)| (x ^ y).count_ones())
                    .sum::<u32>()
            })
            .sum()
    } else {
        // Sequential version for small arrays
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
    }
}

/// Compute Hamming weight (number of set bits)
pub fn hamming_weight(data: &ArrayView1<u8>) -> u32 {
    data.par_iter()
        .map(|&byte| byte.count_ones())
        .sum()
}

/// Find minimum Hamming distance in a set of vectors
pub fn find_min_hamming(
    database: &ArrayView2<u8>,
    query: &ArrayView1<u8>,
) -> (usize, u32) {
    database
        .axis_iter(Axis(0))
        .enumerate()
        .par_bridge()
        .map(|(idx, row)| {
            let dist = compute_hamming_distance(&row, query);
            (idx, dist)
        })
        .min_by_key(|&(_, dist)| dist)
        .unwrap_or((0, u32::MAX))
}

/// Compute pairwise Hamming distances
pub fn pairwise_hamming_distances(vectors: &ArrayView2<u8>) -> Vec<Vec<u32>> {
    let n = vectors.nrows();
    let mut distances = vec![vec![0u32; n]; n];

    // Compute upper triangle in parallel
    (0..n).into_par_iter().for_each(|i| {
        for j in (i + 1)..n {
            let dist = compute_hamming_distance(
                &vectors.row(i),
                &vectors.row(j),
            );
            // Use unsafe for performance (bounds already checked)
            unsafe {
                *distances.get_unchecked_mut(i).get_unchecked_mut(j) = dist;
                *distances.get_unchecked_mut(j).get_unchecked_mut(i) = dist;
            }
        }
    });

    distances
}

/// Fast approximate Hamming distance using sampling
pub fn approximate_hamming_distance(
    a: &ArrayView1<u8>,
    b: &ArrayView1<u8>,
    sample_rate: f32,
) -> u32 {
    if sample_rate >= 1.0 {
        return compute_hamming_distance(a, b);
    }

    let sample_size = (a.len() as f32 * sample_rate) as usize;
    let step = a.len() / sample_size;

    let sampled_dist: u32 = (0..sample_size)
        .into_par_iter()
        .map(|i| {
            let idx = i * step;
            if idx < a.len() {
                (a[idx] ^ b[idx]).count_ones()
            } else {
                0
            }
        })
        .sum();

    // Scale up the sampled distance
    (sampled_dist as f32 / sample_rate) as u32
}

/// Hamming distance with early termination
pub fn bounded_hamming_distance(
    a: &ArrayView1<u8>,
    b: &ArrayView1<u8>,
    max_distance: u32,
) -> Option<u32> {
    let mut distance = 0u32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        distance += (x ^ y).count_ones();
        if distance > max_distance {
            return None;
        }
    }

    Some(distance)
}

/// Create Hamming distance lookup table for small binary vectors
pub struct HammingLUT {
    table: Vec<u8>,
}

impl HammingLUT {
    pub fn new() -> Self {
        let mut table = vec![0u8; 256 * 256];

        // Precompute all possible byte-wise Hamming distances
        for i in 0..256 {
            for j in 0..256 {
                table[i * 256 + j] = (i as u8 ^ j as u8).count_ones() as u8;
            }
        }

        HammingLUT { table }
    }

    pub fn distance(&self, a: u8, b: u8) -> u8 {
        self.table[(a as usize) * 256 + (b as usize)]
    }

    pub fn array_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| self.distance(x, y) as u32)
            .sum()
    }
}
