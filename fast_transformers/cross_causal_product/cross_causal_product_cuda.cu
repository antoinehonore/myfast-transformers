//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
// 
// Antoine Honoré <honore@kth.se>
//

//
// For modifications made inside namespace nvidia (authored by jdemouth):
//
// Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <torch/extension.h>
#include <assert.h>
#include <stdio.h>

#define ENABLE_NVIDIA_OPTIMIZATIONS

typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> float_accessor;
typedef torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> float_accessor1;

#define E_BLOCK_SIZE 8

__global__ void causal_dot_product_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor1 tq,
    const float_accessor1 tkv,
    float_accessor result,
    const int N,
    const int H,
    const int L,
    const int E,
    const int M,
    const int L_kv
) {
    int n = blockIdx.y;
    int h = blockIdx.z;

    int e_start = blockIdx.x * E_BLOCK_SIZE;
    int m = threadIdx.x % M;

    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;

    for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
      shared_kv[m + e_local * M] = 0;
    }
    
    int l_kv = 0;
    float res = 0;

    for (int l=0; l<L; l++) {
      res = 0;
      while ((l_kv<L_kv) && (tq[l] >= tkv[l_kv])) {
        
        for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
          shared_kv[e_local*M + m] += keys[n][h][l_kv][e_local + e_start] * values[n][h][l_kv][m];
        }
        l_kv++;
      }

      for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
        res += queries[n][h][l][e_local + e_start] * shared_kv[e_local*M + m];
      }

      atomicAdd(
          &result[n][h][l][m],
          res
      );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_product_(const torch::Tensor queries,
                         const torch::Tensor keys,
                         const torch::Tensor values,
                         const torch::Tensor tq,
                         const torch::Tensor tkv,
                         torch::Tensor product) {
    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(queries.device());

    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);
    int L_kv = keys.size(2);
    const int blocks_per_sequence = (E + E_BLOCK_SIZE - 1) / E_BLOCK_SIZE;

    dim3 blockDim(M, 1, 1);
    dim3 gridDim(blocks_per_sequence, N, H);
    const int shared_mem_forward = E_BLOCK_SIZE * M * sizeof(float);

    causal_dot_product_kernel<<<gridDim, blockDim, shared_mem_forward>>>(
      queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      tq.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      tkv.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      product.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M, L_kv
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_product(const torch::Tensor queries,
                        const torch::Tensor keys,
                        const torch::Tensor values,
                        const torch::Tensor tq,
                        const torch::Tensor tkv,
                        torch::Tensor product) {

//#ifdef ENABLE_NVIDIA_OPTIMIZATIONS
//  int fallback = nvidia::lmha_fwd(queries, keys, values, product);
//#else
//  int fallback = 1;
//#endif
//  if( fallback ) {
    causal_dot_product_(queries, keys, values, tq, tkv, product);
//  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define M_BLOCK_SIZE 4

// we need shared memory to store
// kv
// Backward direction
// kv_backwards
// Shared memory usage
__global__ void causal_dot_backward_query_key_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor1 tq,
    const float_accessor1 tkv,
    const float_accessor grad_out,
    float_accessor grad_queries,
    float_accessor grad_keys,
    int N,
    int H,
    int L,
    int E,
    int M,
    int L_kv
) {
    int n = blockIdx.y;
    int h = blockIdx.z;

    int m_start = blockIdx.x * M_BLOCK_SIZE;
    int e = threadIdx.x % E;

    extern __shared__ float shared_mem[];
    const int shared_kv_size = M_BLOCK_SIZE * E;
    float* shared_kv = shared_mem;
    float* shared_kv_bw = shared_mem + shared_kv_size;
    
    for (int m_local = 0; m_local < M_BLOCK_SIZE && m_local + m_start < M; m_local++) {
      shared_kv[m_local * E + e] = 0;
      shared_kv_bw[m_local * E + e] = 0;
    }

    int l_kv=0;
    float res = 0;

    // QUERIES
    for (int l=0; l<L; l++) {
      res = 0;

      while ( (l_kv<L_kv) && (tq[l] >= tkv[l_kv])) {
        for (int m_local = 0; m_local < M_BLOCK_SIZE && m_local + m_start < M; m_local++) {
          shared_kv[m_local*E + e] += keys[n][h][l_kv][e] * values[n][h][l_kv][m_start + m_local];
        }
      l_kv++;
      }

      for (int m_local = 0; m_local < M_BLOCK_SIZE && m_local + m_start < M; m_local++) {
        res += grad_out[n][h][l][m_start + m_local] * shared_kv[m_local*E + e];
      }

      atomicAdd(
        &grad_queries[n][h][l][e],
        res
      );
    }

    // KEYS
    int l = L - 1;
    float res_bw = 0;
    for (int l_kv=L_kv-1; l_kv>=0; l_kv--) {
      res_bw = 0;

      while ( (l_kv < L_kv) && (tq[l] >= tkv[l_kv])) {
        for (int m_local = 0; m_local < M_BLOCK_SIZE && m_local + m_start < M; m_local++) {
          shared_kv_bw[m_local*E + e] += queries[n][h][l][e] * grad_out[n][h][l][m_start + m_local];
        }
      l--;
      }

      for (int m_local = 0; m_local < M_BLOCK_SIZE && m_local + m_start < M; m_local++) {
        res_bw += values[n][h][l_kv][m_start + m_local] * shared_kv_bw[m_local*E + e];
      }

      atomicAdd(
        &grad_keys[n][h][l_kv][e],
        res_bw
      );
      
    }
}


__global__ void causal_dot_backward_value_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor1 tq,
    const float_accessor1 tkv,
    const float_accessor grad_out,
    float_accessor grad_keys,
    float_accessor grad_values,
    int N,
    int H,
    int L,
    int E,
    int M,
    int L_kv
) {
    int n = blockIdx.y;
    int h = blockIdx.z;

    int e_start = blockIdx.x * E_BLOCK_SIZE;
    int m = threadIdx.x % M;

    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;
    for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
      shared_kv[m + e_local * M] = 0;
    }

    
    int l = L - 1;
    float res = 0;

    // VALUES
    for (int l_kv = L_kv-1; l_kv>=0; l_kv--) {
        res = 0;
        while ((l>=0) && (tq[l] >= tkv[l_kv])) {
          for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
            shared_kv[e_local*M + m] += queries[n][h][l][e_start + e_local] * grad_out[n][h][l][m];
          }
          l--;
        }

        for (int e_local = 0; e_local < E_BLOCK_SIZE && e_local + e_start < E; e_local++) {
          res += keys[n][h][l_kv][e_start + e_local] * shared_kv[e_local*M + m];
        }

        atomicAdd(
            &grad_values[n][h][l_kv][m],
            res
        );
      }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_backward_(const torch::Tensor queries,
                          const torch::Tensor keys,
                          const torch::Tensor values,
                          const torch::Tensor tq,
                          const torch::Tensor tkv,
                          const torch::Tensor grad_out,
                          torch::Tensor grad_queries,
                          torch::Tensor grad_keys,
                          torch::Tensor grad_values) {

    // Make sure that we are using the correct GPU device
    torch::DeviceGuard _guard(queries.device());

    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);
    int L_kv = keys.size(2);

    const int blocks_per_sequence = (M + M_BLOCK_SIZE - 1) / M_BLOCK_SIZE;

    dim3 blockDim(E, 1, 1);
    dim3 gridDim(blocks_per_sequence, N, H);
    const int shared_mem_qk_backward = 2 * M_BLOCK_SIZE * E * sizeof(float);

    causal_dot_backward_query_key_kernel<<<gridDim, blockDim, shared_mem_qk_backward>>>(
      queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      tq.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      tkv.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M, L_kv
    );

    const int blocks_per_sequence_value = (E + E_BLOCK_SIZE - 1) / E_BLOCK_SIZE;

    dim3 blockDimv(M, 1, 1);
    dim3 gridDimv(blocks_per_sequence_value, N, H);
    const int shared_mem_v_backward = E_BLOCK_SIZE * M * sizeof(float);
    causal_dot_backward_value_kernel<<<gridDimv, blockDimv, shared_mem_v_backward>>>(
      queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      tq.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      tkv.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
      grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      grad_values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
      N, H, L, E, M, L_kv
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void causal_dot_backward(const torch::Tensor queries,
                         const torch::Tensor keys,
                         const torch::Tensor values,
                         const torch::Tensor tq,
                         const torch::Tensor tkv,
                         const torch::Tensor grad_out,
                         torch::Tensor grad_queries,
                         torch::Tensor grad_keys,
                         torch::Tensor grad_values) {

//#ifdef ENABLE_NVIDIA_OPTIMIZATIONS
//  int fallback = nvidia::lmha_bwd(queries,
//                                  keys,
//                                  values,
//                                  grad_out,
//                                  grad_queries,
//                                  grad_keys,
//                                  grad_values);
//#else
//  int fallback = 1;
//#endif
//  if( fallback ) {
    // Make sure that the gradient tensors are 0. This is needed because the
    // bwd pass might have partially executed and filled in some values in
    // grad_queries or grad_keys.
    //
    // This adds a small overhead every time we have to fall back to the old
    // kernel for the backward pass.
    //grad_queries.zero_();
    //grad_keys.zero_();
    causal_dot_backward_(queries, keys, values, tq, tkv, grad_out, grad_queries, grad_keys, grad_values);
//  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cross_causal_dot_product",
        &causal_dot_product,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "cross_causal_dot_backward",
        &causal_dot_backward,
        "Compute the gradients for the causal dot product."
    );
}