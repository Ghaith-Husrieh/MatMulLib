#include "../../include/matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static size_t *pad_shape(const size_t *original_shape, const size_t original_ndim, const size_t target_ndim)
{
    size_t *padded_shape = malloc(target_ndim * sizeof(size_t));
    if (!padded_shape)
    {
        fprintf(stderr, "Error: Memory allocation failed for padded_shape.\n");
        return NULL;
    }

    size_t padding_dims = target_ndim - original_ndim;
    for (size_t i = 0; i < padding_dims; i++)
        padded_shape[i] = 1;
    memcpy(padded_shape + padding_dims, original_shape, original_ndim * sizeof(size_t));

    return padded_shape;
}

static size_t *compute_broadcasted_shape(const size_t *shapeA, const size_t *shapeB, size_t ndim)
{
    size_t *computed_shapeC = malloc(ndim * sizeof(size_t));
    if (!computed_shapeC)
    {
        fprintf(stderr, "Error: Memory allocation failed for computed_shapeC.\n");
        return NULL;
    }

    for (size_t i = 0; i < ndim - 2; i++)
    {
        if (shapeA[i] == shapeB[i] || shapeA[i] == 1 || shapeB[i] == 1)
        {
            computed_shapeC[i] = (shapeA[i] == 1) ? shapeB[i] : shapeA[i];
        }
        else
        {
            fprintf(stderr, "Error: Tensors are not broadcast-compatible at dimension %zu (shapeA[%zu] = %zu, shapeB[%zu] = %zu).\n",
                    i, i, shapeA[i], i, shapeB[i]);
            free(computed_shapeC);
            return NULL;
        }
    }

    computed_shapeC[ndim - 2] = shapeA[ndim - 2];
    computed_shapeC[ndim - 1] = shapeB[ndim - 1];

    return computed_shapeC;
}

Tensor *matmul(const Tensor *A, const Tensor *B)
{
    // TODO: Add 0D and 1D Tensor support
    if (A->ndim < 2 || B->ndim < 2)
    {
        fprintf(stderr, "Error: Both tensors must have at least 2 dimensions for matmul.\n");
        return 0;
    }
    if (A->shape[A->ndim - 1] != B->shape[B->ndim - 2])
    {
        fprintf(stderr, "Error: Invalid shapes for matmul operation (%zu,%zu) x (%zu,%zu).\n",
                A->shape[A->ndim - 2], A->shape[A->ndim - 1],
                B->shape[B->ndim - 2], B->shape[B->ndim - 1]);
        return 0;
    }

    const size_t ndim = (A->ndim > B->ndim) ? A->ndim : B->ndim;

    size_t *padded_shapeA = NULL;
    size_t *padded_shapeB = NULL;

    if (A->ndim < B->ndim)
    {
        padded_shapeA = pad_shape(A->shape, A->ndim, B->ndim);
        if (!padded_shapeA)
            return NULL;
    }
    else if (A->ndim > B->ndim)
    {
        padded_shapeB = pad_shape(B->shape, B->ndim, A->ndim);
        if (!padded_shapeB)
            return NULL;
    }

    const size_t *shapeA = padded_shapeA ? padded_shapeA : A->shape;
    const size_t *shapeB = padded_shapeB ? padded_shapeB : B->shape;

    size_t *computed_shapeC = compute_broadcasted_shape(shapeA, shapeB, ndim);
    if (!computed_shapeC)
    {
        free(computed_shapeC);
        free(padded_shapeA);
        free(padded_shapeB);
        return NULL;
    }

    Tensor *C = empty_tensor(computed_shapeC, ndim);
    if (!C)
    {
        free(computed_shapeC);
        free(padded_shapeA);
        free(padded_shapeB);
        return NULL;
    }

    free(computed_shapeC);

    size_t batch_size = 1;
    for (size_t i = 0; i < ndim - 2; i++)
        batch_size *= C->shape[i];

    size_t M = shapeA[ndim - 2]; // rows of A and C
    size_t K = shapeA[ndim - 1]; // columns of A and rows of B
    size_t N = shapeB[ndim - 1]; // columns of B and C

    for (size_t batch = 0; batch < batch_size; batch++)
    {
        size_t index_offsetA = 0, index_offsetB = 0;
        size_t strideA = 1, strideB = 1;

        // Compute offsets for batch indices
        for (size_t i = ndim - 3; i < (size_t)-1; i--)
        {
            size_t idx = (batch / strideA) % C->shape[i];
            if (shapeA[i] != 1)
                index_offsetA += idx * strideA;
            if (shapeB[i] != 1)
                index_offsetB += idx * strideB;

            strideA *= C->shape[i];
            strideB *= C->shape[i];
        }

        // Calculate pointers for the current batch
        const double *A_batch = A->buffer + index_offsetA * M * K;
        const double *B_batch = B->buffer + index_offsetB * K * N;
        double *C_batch = C->buffer + batch * M * N;

#pragma omp parallel for collapse(2) schedule(static)
        // Matrix multiplication for the current batch
        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < K; k++)
                    sum += A_batch[i * K + k] * B_batch[k * N + j];
                C_batch[i * N + j] = sum;
            }
        }
    }

    if (padded_shapeA)
        free(padded_shapeA);
    if (padded_shapeB)
        free(padded_shapeB);

    return C;
}