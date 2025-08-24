// GenomeVault Rust Accelerator - High-performance hot path optimizations
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rayon::prelude::*;
use std::sync::Arc;
use num_traits::Float;

mod hdc;
mod pir;
mod hamming;
mod compression;

use crate::hdc::*;
use crate::pir::*;
use crate::hamming::*;
use crate::compression::*;

/// Fast hypervector similarity computation with SIMD optimization
#[pyfunction]
fn fast_hypervector_similarity(
    py: Python,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
) -> PyResult<f32> {
    let a = a.as_array();
    let b = b.as_array();
    
    if a.len() != b.len() {
        return Err(PyValueError::new_err("Vectors must have same length"));
    }
    
    // Release GIL for computation
    py.allow_threads(|| {
        compute_cosine_similarity(&a, &b)
    })
}

/// Batch hypervector similarity computation
#[pyfunction]
fn batch_hypervector_similarity(
    py: Python,
    vectors: PyReadonlyArray2<f32>,
    query: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    let vectors = vectors.as_array();
    let query = query.as_array();
    
    let similarities: Vec<f32> = py.allow_threads(|| {
        vectors.axis_iter(Axis(0))
            .into_par_iter()
            .map(|v| compute_cosine_similarity(&v, &query))
            .collect()
    });
    
    Ok(PyArray1::from_vec(py, similarities).to_owned())
}

/// Fast PIR XOR mask operation
#[pyfunction]
fn fast_pir_xor_mask(
    py: Python,
    data: PyReadonlyArray1<u8>,
    mask: PyReadonlyArray1<u8>,
) -> PyResult<Py<PyArray1<u8>>> {
    let data = data.as_array();
    let mask = mask.as_array();
    
    if data.len() != mask.len() {
        return Err(PyValueError::new_err("Data and mask must have same length"));
    }
    
    let result = py.allow_threads(|| {
        apply_xor_mask(&data, &mask)
    });
    
    Ok(PyArray1::from_vec(py, result).to_owned())
}

/// Batch PIR query processing
#[pyfunction]
fn batch_pir_query(
    py: Python,
    database: PyReadonlyArray2<u8>,
    query_mask: PyReadonlyArray1<u8>,
) -> PyResult<Py<PyArray1<u8>>> {
    let database = database.as_array();
    let query_mask = query_mask.as_array();
    
    if database.nrows() != query_mask.len() {
        return Err(PyValueError::new_err("Query mask length must match database rows"));
    }
    
    let result = py.allow_threads(|| {
        process_pir_query(&database, &query_mask)
    });
    
    Ok(PyArray1::from_vec(py, result).to_owned())
}

/// Fast Hamming distance computation
#[pyfunction]
fn fast_hamming_distance(
    py: Python,
    a: PyReadonlyArray1<u8>,
    b: PyReadonlyArray1<u8>,
) -> PyResult<u32> {
    let a = a.as_array();
    let b = b.as_array();
    
    if a.len() != b.len() {
        return Err(PyValueError::new_err("Arrays must have same length"));
    }
    
    py.allow_threads(|| {
        Ok(compute_hamming_distance(&a, &b))
    })
}

/// Batch Hamming distance computation
#[pyfunction]
fn batch_hamming_distance(
    py: Python,
    vectors: PyReadonlyArray2<u8>,
    query: PyReadonlyArray1<u8>,
) -> PyResult<Py<PyArray1<u32>>> {
    let vectors = vectors.as_array();
    let query = query.as_array();
    
    let distances: Vec<u32> = py.allow_threads(|| {
        vectors.axis_iter(Axis(0))
            .into_par_iter()
            .map(|v| compute_hamming_distance(&v, &query))
            .collect()
    });
    
    Ok(PyArray1::from_vec(py, distances).to_owned())
}

/// Fast variant encoding to hypervector
#[pyfunction]
fn fast_encode_variant(
    py: Python,
    chromosome: u8,
    position: u32,
    ref_allele: &str,
    alt_allele: &str,
    dimension: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let encoded = py.allow_threads(|| {
        encode_variant_to_hypervector(chromosome, position, ref_allele, alt_allele, dimension)
    })?;
    
    Ok(PyArray1::from_vec(py, encoded).to_owned())
}

/// Batch variant encoding
#[pyfunction]
fn batch_encode_variants(
    py: Python,
    chromosomes: PyReadonlyArray1<u8>,
    positions: PyReadonlyArray1<u32>,
    dimension: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let chromosomes = chromosomes.as_array();
    let positions = positions.as_array();
    
    if chromosomes.len() != positions.len() {
        return Err(PyValueError::new_err("Chromosomes and positions must have same length"));
    }
    
    let encoded = py.allow_threads(|| {
        batch_encode_variants_to_hypervectors(&chromosomes, &positions, dimension)
    })?;
    
    let shape = (chromosomes.len(), dimension);
    Ok(PyArray2::from_vec(py, encoded).reshape(shape)?.to_owned())
}

/// Fast sparse hypervector multiplication
#[pyfunction]
fn sparse_hypervector_multiply(
    py: Python,
    indices: PyReadonlyArray1<usize>,
    values: PyReadonlyArray1<f32>,
    dense_vector: PyReadonlyArray1<f32>,
) -> PyResult<f32> {
    let indices = indices.as_array();
    let values = values.as_array();
    let dense = dense_vector.as_array();
    
    if indices.len() != values.len() {
        return Err(PyValueError::new_err("Indices and values must have same length"));
    }
    
    py.allow_threads(|| {
        sparse_dot_product(&indices, &values, &dense)
    })
}

/// Fast compression of hypervector to binary
#[pyfunction]
fn compress_hypervector_binary(
    py: Python,
    vector: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray1<u8>>> {
    let vector = vector.as_array();
    
    let compressed = py.allow_threads(|| {
        compress_to_binary(&vector)
    });
    
    Ok(PyArray1::from_vec(py, compressed).to_owned())
}

/// Fast decompression from binary to hypervector
#[pyfunction]
fn decompress_binary_hypervector(
    py: Python,
    compressed: PyReadonlyArray1<u8>,
    dimension: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let compressed = compressed.as_array();
    
    let decompressed = py.allow_threads(|| {
        decompress_from_binary(&compressed, dimension)
    })?;
    
    Ok(PyArray1::from_vec(py, decompressed).to_owned())
}

/// Fast k-nearest neighbors search in hypervector space
#[pyfunction]
fn fast_knn_search(
    py: Python,
    database: PyReadonlyArray2<f32>,
    query: PyReadonlyArray1<f32>,
    k: usize,
) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray1<f32>>)> {
    let database = database.as_array();
    let query = query.as_array();
    
    let (indices, distances) = py.allow_threads(|| {
        knn_search(&database, &query, k)
    })?;
    
    Ok((
        PyArray1::from_vec(py, indices).to_owned(),
        PyArray1::from_vec(py, distances).to_owned(),
    ))
}

/// Initialize the Python module
#[pymodule]
fn genomevault_accel(_py: Python, m: &PyModule) -> PyResult<()> {
    // HDC operations
    m.add_function(wrap_pyfunction!(fast_hypervector_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_hypervector_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(fast_encode_variant, m)?)?;
    m.add_function(wrap_pyfunction!(batch_encode_variants, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_hypervector_multiply, m)?)?;
    
    // PIR operations
    m.add_function(wrap_pyfunction!(fast_pir_xor_mask, m)?)?;
    m.add_function(wrap_pyfunction!(batch_pir_query, m)?)?;
    
    // Hamming distance operations
    m.add_function(wrap_pyfunction!(fast_hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(batch_hamming_distance, m)?)?;
    
    // Compression operations
    m.add_function(wrap_pyfunction!(compress_hypervector_binary, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_binary_hypervector, m)?)?;
    
    // Search operations
    m.add_function(wrap_pyfunction!(fast_knn_search, m)?)?;
    
    Ok(())
}