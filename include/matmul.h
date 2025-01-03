#pragma once
#include "tensor.h"

/**
 * @brief Perform matrix multiplication with broadcasting enabled.
 *
 * Computes the result of `A @ B` (matrix multiplication) and returns a new tensor.
 * Handles broadcasting for higher dimensions.
 *
 * @param A Pointer to the first input tensor.
 * @param B Pointer to the second input tensor.
 * @return Pointer to the resulting tensor, or NULL on failure.
 */
Tensor *matmul(const Tensor *A, const Tensor *B);
