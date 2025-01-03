#include "../../include/matmul.h"
#include <stdio.h>
#include <stdlib.h>

static size_t compute_broadcasted_shape(const size_t *shapeA, const size_t *shapeB, size_t *computed_shapeC, size_t ndim)
{
    if (ndim < 2)
    {
        fprintf(stderr, "Error: Both tensors must have at least 2 dimensions for matmul.\n");
        return 0;
    }
    if (shapeA[ndim - 1] != shapeB[ndim - 2])
    {
        fprintf(stderr, "Error: Invalid shapes for matmul operation (%zu,%zu) x (%zu,%zu).\n", shapeA[ndim - 2], shapeA[ndim - 1], shapeB[ndim - 2], shapeA[ndim - 1]);
        return 0;
    }
    for (size_t i = 0; i < ndim - 2; i++)
    {
        if (shapeA[i] == shapeB[i] || shapeA[i] == 1 || shapeB[i] == 1)
        {
            computed_shapeC[i] = (shapeA[i] == 1) ? shapeB[i] : shapeA[i];
        }
        else
        {
            fprintf(stderr, "Error: Tensors are not broadcast-compatible at dimension %zu (shapeA[%zu] = %zu, shapeB[%zu] = %zu).\n", i, i, shapeA[i], i, shapeB[i]);
            return 0;
        }
    }

    computed_shapeC[ndim - 2] = shapeA[ndim - 2];
    computed_shapeC[ndim - 1] = shapeB[ndim - 1];

    return 1;
}

Tensor *matmul(const Tensor *A, const Tensor *B)
{
    // TODO: Instead of giving an error we should pad the Tensor with the smaller ndim
    if (A->ndim != B->ndim)
    {
        fprintf(stderr, "Error: Tensors A and B must have the same number of dimensions (ndim). A->ndim = %zu, B->ndim = %zu\n", A->ndim, B->ndim);
        return NULL;
    }
    size_t ndim = A->ndim;

    size_t *computed_shapeC = malloc(ndim * sizeof(size_t));
    if (!computed_shapeC)
    {
        fprintf(stderr, "Error: Memory allocation failed for computed_shapeC.\n");
        return NULL;
    }

    if (!compute_broadcasted_shape(A->shape, B->shape, computed_shapeC, ndim))
    {
        free(computed_shapeC);
        return NULL;
    }

    Tensor *C = empty_tensor(computed_shapeC, ndim);
    if (!C)
    {
        free(computed_shapeC);
        return NULL;
    }

    free(computed_shapeC);

    size_t batch_size = 1;
    for (size_t i = 0; i < ndim - 2; i++)
    {
        batch_size *= C->shape[i];
    }

    size_t M = A->shape[ndim - 2]; // rows of A and C
    size_t K = A->shape[ndim - 1]; // columns of A and rows of B
    size_t N = B->shape[ndim - 1]; // columns of B and C

    for (size_t batch = 0; batch < batch_size; batch++)
    {
        size_t indexOffsetA = 0, indexOffsetB = 0;
        size_t strideA = 1, strideB = 1;

        // Compute offsets for batch indices
        for (size_t i = ndim - 3; i < (size_t)-1; i--)
        {
            size_t idx = (batch / strideA) % C->shape[i];
            if (A->shape[i] != 1)
                indexOffsetA += idx * strideA;
            if (B->shape[i] != 1)
                indexOffsetB += idx * strideB;

            strideA *= C->shape[i];
            strideB *= C->shape[i];
        }

        // Calculate pointers for the current batch
        const double *A_batch = A->buffer + indexOffsetA * M * K;
        const double *B_batch = B->buffer + indexOffsetB * K * N;
        double *C_batch = C->buffer + batch * M * N;

        // Matrix multiplication for the current batch
        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < K; k++)
                {
                    sum += A_batch[i * K + k] * B_batch[k * N + j];
                }
                C_batch[i * N + j] = sum;
            }
        }
    }

    return C;
}