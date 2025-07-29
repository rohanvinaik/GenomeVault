"""
Hamming Distance Look-Up Table (LUT) Core
==========================================

A high-performance Hamming distance computation module using 16-bit lookup tables
for accelerated similarity calculations in hyperdimensional computing (HDC).

This module provides:
- Shared LUT generation for CPU, GPU, PULP, and FPGA platforms
- Optimized popcount operations for binary hypervectors
- Platform-specific acceleration paths
- Memory-efficient caching strategies

Performance targets:
- 2-3× speedup on PULPv3 and FPGA fabric
- >1.5× speedup on CPU/GPU platforms
"""

import functools
import os
from typing import Optional, Tuple, Union

import numpy as np
from numba import cuda, jit, prange

# Global LUT cache - shared across all compute contexts
_POPCOUNT_LUT_16: np.ndarray | None = None


def generate_popcount_lut() -> np.ndarray:
    """
    Generate a 16-bit popcount lookup table.

    Returns:
        np.ndarray: Array of size 2^16 with popcount values
    """
    global _POPCOUNT_LUT_16

    if _POPCOUNT_LUT_16 is None:
        # Generate LUT using builtin popcount
        _POPCOUNT_LUT_16 = np.zeros(1 << 16, dtype=np.uint8)
        for i in range(1 << 16):
            _POPCOUNT_LUT_16[i] = bin(i).count("1")

    return _POPCOUNT_LUT_16


@functools.lru_cache(maxsize=1)
def get_cuda_popcount_lut():
    """Get CUDA device memory copy of popcount LUT."""
    lut = generate_popcount_lut()
    return cuda.to_device(lut)


# CPU-optimized implementations
@jit(nopython=True, parallel=True, cache=True)
def hamming_distance_cpu(vec1: np.ndarray, vec2: np.ndarray, lut: np.ndarray) -> int:
    """
    Compute Hamming distance between two binary vectors using LUT.

    Args:
        vec1: First binary vector (uint64 array)
        vec2: Second binary vector (uint64 array)
        lut: Popcount lookup table

    Returns:
        int: Hamming distance
    """
    distance = 0

    # Process vectors as 64-bit words
    for i in prange(len(vec1)):
        xor_val = vec1[i] ^ vec2[i]

        # Split 64-bit word into four 16-bit lookups
        distance += lut[(xor_val >> 0) & 0xFFFF]
        distance += lut[(xor_val >> 16) & 0xFFFF]
        distance += lut[(xor_val >> 32) & 0xFFFF]
        distance += lut[(xor_val >> 48) & 0xFFFF]

    return distance


@jit(nopython=True, parallel=True, cache=True)
def hamming_distance_batch_cpu(vecs1: np.ndarray, vecs2: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Hamming distances for batches of vectors.

    Args:
        vecs1: First batch of binary vectors (N x D//64)
        vecs2: Second batch of binary vectors (M x D//64)
        lut: Popcount lookup table

    Returns:
        np.ndarray: Pairwise distances (N x M)
    """
    n, dim = vecs1.shape
    m, _ = vecs2.shape
    distances = np.zeros((n, m), dtype=np.int32)

    for i in prange(n):
        for j in range(m):
            dist = 0
            for k in range(dim):
                xor_val = vecs1[i, k] ^ vecs2[j, k]

                # Four 16-bit lookups per 64-bit word
                dist += lut[(xor_val >> 0) & 0xFFFF]
                dist += lut[(xor_val >> 16) & 0xFFFF]
                dist += lut[(xor_val >> 32) & 0xFFFF]
                dist += lut[(xor_val >> 48) & 0xFFFF]

            distances[i, j] = dist

    return distances


# GPU-optimized implementations
@cuda.jit
def hamming_distance_kernel(vec1, vec2, lut, result):
    """
    CUDA kernel for computing Hamming distance using LUT.

    Args:
        vec1: First binary vector (device array)
        vec2: Second binary vector (device array)
        lut: Popcount LUT (constant memory)
        result: Output distance (single element array)
    """
    tid = cuda.grid(1)
    block_sum = cuda.shared.array(256, dtype=np.int32)

    # Initialize shared memory
    block_sum[cuda.threadIdx.x] = 0

    # Each thread processes multiple elements
    elements_per_thread = (vec1.shape[0] + cuda.gridsize(1) - 1) // cuda.gridsize(1)
    start = tid * elements_per_thread
    end = min(start + elements_per_thread, vec1.shape[0])

    local_sum = 0
    for i in range(start, end):
        xor_val = vec1[i] ^ vec2[i]

        # Four 16-bit lookups
        local_sum += lut[(xor_val >> 0) & 0xFFFF]
        local_sum += lut[(xor_val >> 16) & 0xFFFF]
        local_sum += lut[(xor_val >> 32) & 0xFFFF]
        local_sum += lut[(xor_val >> 48) & 0xFFFF]

    block_sum[cuda.threadIdx.x] = local_sum
    cuda.syncthreads()

    # Reduction within block
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if cuda.threadIdx.x < stride:
            block_sum[cuda.threadIdx.x] += block_sum[cuda.threadIdx.x + stride]
        cuda.syncthreads()
        stride //= 2

    # First thread writes block result
    if cuda.threadIdx.x == 0:
        cuda.atomic.add(result, 0, block_sum[0])


@cuda.jit
def hamming_distance_batch_kernel(vecs1, vecs2, lut, distances):
    """
    CUDA kernel for batch Hamming distance computation.

    Each thread computes one distance value.
    """
    i, j = cuda.grid(2)

    if i < vecs1.shape[0] and j < vecs2.shape[0]:
        dist = 0
        for k in range(vecs1.shape[1]):
            xor_val = vecs1[i, k] ^ vecs2[j, k]

            # Four 16-bit lookups
            dist += lut[(xor_val >> 0) & 0xFFFF]
            dist += lut[(xor_val >> 16) & 0xFFFF]
            dist += lut[(xor_val >> 32) & 0xFFFF]
            dist += lut[(xor_val >> 48) & 0xFFFF]

        distances[i, j] = dist


class HammingLUT:
    """
    High-level interface for Hamming distance computation with LUT acceleration.
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize Hamming LUT calculator.

        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.lut = generate_popcount_lut()
        self.use_gpu = use_gpu and cuda.is_available()

        if self.use_gpu:
            self.cuda_lut = get_cuda_popcount_lut()

    def distance(self, vec1: np.ndarray, vec2: np.ndarray) -> int:
        """
        Compute Hamming distance between two binary vectors.

        Args:
            vec1: First binary vector
            vec2: Second binary vector

        Returns:
            int: Hamming distance
        """
        # Ensure vectors are in uint64 format
        if vec1.dtype != np.uint64:
            vec1 = vec1.astype(np.uint64)
        if vec2.dtype != np.uint64:
            vec2 = vec2.astype(np.uint64)

        if self.use_gpu:
            # GPU implementation
            d_vec1 = cuda.to_device(vec1)
            d_vec2 = cuda.to_device(vec2)
            d_result = cuda.device_array(1, dtype=np.int32)
            d_result[0] = 0

            threads_per_block = 256
            blocks = (vec1.shape[0] + threads_per_block - 1) // threads_per_block

            hamming_distance_kernel[blocks, threads_per_block](
                d_vec1, d_vec2, self.cuda_lut, d_result
            )

            return int(d_result.copy_to_host()[0])
        else:
            # CPU implementation
            return hamming_distance_cpu(vec1, vec2, self.lut)

    def distance_batch(self, vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Hamming distances for batches of vectors.

        Args:
            vecs1: First batch of binary vectors (N x D)
            vecs2: Second batch of binary vectors (M x D)

        Returns:
            np.ndarray: Pairwise distances (N x M)
        """
        # Ensure vectors are in uint64 format
        if vecs1.dtype != np.uint64:
            vecs1 = vecs1.astype(np.uint64)
        if vecs2.dtype != np.uint64:
            vecs2 = vecs2.astype(np.uint64)

        if self.use_gpu:
            # GPU implementation
            d_vecs1 = cuda.to_device(vecs1)
            d_vecs2 = cuda.to_device(vecs2)
            d_distances = cuda.device_array((vecs1.shape[0], vecs2.shape[0]), dtype=np.int32)

            threads_per_block = (16, 16)
            blocks_x = (vecs1.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_y = (vecs2.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
            blocks = (blocks_x, blocks_y)

            hamming_distance_batch_kernel[blocks, threads_per_block](
                d_vecs1, d_vecs2, self.cuda_lut, d_distances
            )

            return d_distances.copy_to_host()
        else:
            # CPU implementation
            return hamming_distance_batch_cpu(vecs1, vecs2, self.lut)


# PULP-specific implementation placeholder
def generate_pulp_lut_code() -> str:
    """
    Generate PULP-specific C code for LUT implementation.

    Returns:
        str: C code for PULP platform
    """
    lut = generate_popcount_lut()

    code = """
// Auto-generated PULP Hamming LUT implementation
#include <pulp.h>

// 16-bit popcount lookup table
__attribute__((section(".l1_prio")))
const uint8_t POPCOUNT_LUT_16[65536] = {
"""

    # Generate LUT values
    for i in range(0, len(lut), 16):
        values = ", ".join(str(lut[j]) for j in range(i, min(i + 16, len(lut))))
        code += f"    {values},\n"

    code = code.rstrip(",\n") + "\n};\n\n"

    code += """
// PULP-optimized Hamming distance computation
uint32_t hamming_distance_pulp(const uint64_t* vec1, const uint64_t* vec2, size_t len) {
    uint32_t distance = 0;

    // Parallel computation across PULP cores
    #pragma omp parallel for reduction(+:distance)
    for (size_t i = 0; i < len; i++) {
        uint64_t xor_val = vec1[i] ^ vec2[i];

        // Four 16-bit lookups per 64-bit word
        distance += POPCOUNT_LUT_16[(xor_val >> 0) & 0xFFFF];
        distance += POPCOUNT_LUT_16[(xor_val >> 16) & 0xFFFF];
        distance += POPCOUNT_LUT_16[(xor_val >> 32) & 0xFFFF];
        distance += POPCOUNT_LUT_16[(xor_val >> 48) & 0xFFFF];
    }

    return distance;
}
"""

    return code


# FPGA-specific implementation placeholder
def generate_fpga_verilog() -> str:
    """
    Generate Verilog code for FPGA LUT implementation.

    Returns:
        str: Verilog code for FPGA
    """
    return """
// Auto-generated FPGA Hamming LUT implementation
module hamming_lut_core #(
    parameter VECTOR_WIDTH = 10000,
    parameter WORD_WIDTH = 64
) (
    input wire clk,
    input wire rst,
    input wire [VECTOR_WIDTH-1:0] vec1,
    input wire [VECTOR_WIDTH-1:0] vec2,
    output reg [31:0] hamming_distance,
    output reg done
);

    // 16-bit popcount LUT (implemented as distributed RAM)
    reg [7:0] popcount_lut [0:65535];

    // Initialize LUT
    initial begin
        $readmemh("popcount_lut.hex", popcount_lut);
    end

    // State machine
    localparam IDLE = 2'b00;
    localparam COMPUTE = 2'b01;
    localparam DONE = 2'b10;

    reg [1:0] state;
    reg [31:0] word_idx;
    reg [31:0] accumulator;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            hamming_distance <= 0;
            done <= 0;
            word_idx <= 0;
            accumulator <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    word_idx <= 0;
                    accumulator <= 0;
                    state <= COMPUTE;
                end

                COMPUTE: begin
                    if (word_idx < VECTOR_WIDTH/WORD_WIDTH) begin
                        // XOR current 64-bit word
                        wire [63:0] xor_word = vec1[word_idx*64 +: 64] ^
                                              vec2[word_idx*64 +: 64];

                        // Four parallel 16-bit lookups
                        accumulator <= accumulator +
                            popcount_lut[xor_word[15:0]] +
                            popcount_lut[xor_word[31:16]] +
                            popcount_lut[xor_word[47:32]] +
                            popcount_lut[xor_word[63:48]];

                        word_idx <= word_idx + 1;
                    end else begin
                        hamming_distance <= accumulator;
                        state <= DONE;
                    end
                end

                DONE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
"""


# Export convenience functions
def export_platform_implementations(output_dir: str):
    """
    Export platform-specific implementations to files.

    Args:
        output_dir: Directory to write implementation files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Export PULP implementation
    with open(os.path.join(output_dir, "hamming_lut_pulp.c"), "w") as f:
        f.write(generate_pulp_lut_code())

    # Export FPGA implementation
    with open(os.path.join(output_dir, "hamming_lut_fpga.v"), "w") as f:
        f.write(generate_fpga_verilog())

    # Export LUT hex file for FPGA
    lut = generate_popcount_lut()
    with open(os.path.join(output_dir, "popcount_lut.hex"), "w") as f:
        for val in lut:
            f.write(f"{val:02x}\n")
