#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/tensor.h"
#include "../../include/matmul.h"
#include "../../include/random.h"

static void print_tensor_recursive(const double *data, const size_t *shape, size_t ndim, size_t level, size_t *index_offset)
{
    if (ndim == 1)
    {
        // Base case: Print 1D array
        printf("[");
        for (size_t i = 0; i < shape[0]; i++)
        {
            printf("%.8f", data[*index_offset]);
            (*index_offset)++;
            if (i < shape[0] - 1)
            {
                printf(", ");
            }
        }
        printf("]");
        return;
    }

    // Recursive case: Print nested arrays
    printf("[\n");
    for (size_t i = 0; i < shape[0]; i++)
    {
        // Indentation for nested levels
        for (size_t j = 0; j < level; j++)
        {
            printf("    ");
        }
        print_tensor_recursive(data, shape + 1, ndim - 1, level + 1, index_offset);
        if (i < shape[0] - 1)
        {
            printf(",\n");
        }
    }
    printf("\n");

    // Indentation for closing brackets
    for (size_t j = 0; j < level - 1; j++)
    {
        printf("    ");
    }
    printf("]");
}

static void print_tensor(const Tensor *tensor)
{
    if (!tensor)
    {
        printf("Tensor is NULL\n");
        return;
    }

    if (!tensor->shape || !tensor->buffer || tensor->ndim == 0)
    {
        printf("Tensor is not initialized properly\n");
        return;
    }

    printf("tensor(");
    size_t index_offset = 0;
    print_tensor_recursive(tensor->buffer, tensor->shape, tensor->ndim, 1, &index_offset);
    printf(")\n");
}

static void free_tensor(Tensor *tensor)
{
    free(tensor->shape);
    free(tensor->buffer);
    free(tensor);
}

static Tensor *tensor_init(const double *data, const size_t *shape, size_t ndim, enum TensorInitMode init_mode)
{
    if (ndim == 0)
    {
        fprintf(stderr, "Error: ndim cannot be zero\n");
        return NULL;
    }

    Tensor *tensor = malloc(sizeof(Tensor));
    if (!tensor)
    {
        fprintf(stderr, "Memory allocation failed for Tensor\n");
        return NULL;
    }

    tensor->ndim = ndim;

    tensor->shape = malloc(ndim * sizeof(size_t));
    if (!tensor->shape)
    {
        fprintf(stderr, "Error: Memory allocation failed for tensor shape\n");
        free(tensor);
        return NULL;
    }
    if (shape)
    {
        memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    }
    else
    {
        fprintf(stderr, "Error: shape is NULL\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    size_t numel = 1;
    for (size_t i = 0; i < ndim; i++)
    {
        numel *= shape[i];
    }
    if (numel == 0)
    {
        fprintf(stderr, "Error: Tensor has zero elements (empty shape)\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    tensor->buffer = malloc(numel * sizeof(double));
    if (!tensor->buffer)
    {
        fprintf(stderr, "Error: Memory allocation failed for tensor data\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    switch (init_mode)
    {
    case TENSOR_WITH_DATA:
        if (!data)
        {
            fprintf(stderr, "Error: data is NULL for TENSOR_WITH_DATA\n");
            free(tensor->shape);
            free(tensor->buffer);
            free(tensor);
            return NULL;
        }
        memcpy(tensor->buffer, data, numel * sizeof(double));
        break;

    case TENSOR_UNINITIALIZED:
        // Leave buffer uninitialized
        break;

    case TENSOR_ZEROS:
        memset(tensor->buffer, 0, numel * sizeof(double));
        break;

    case TENSOR_ONES:
        for (size_t i = 0; i < numel; i++)
        {
            tensor->buffer[i] = 1.0;
        }
        break;
    case TENSOR_RANDN:
        for (size_t i = 0; i < numel; i++)
        {
            tensor->buffer[i] = normal(0.0, 1.0);
        }
        break;
    case TENSOR_RAND:
        for (size_t i = 0; i < numel; i++)
        {
            tensor->buffer[i] = uniform(0.0, 1.0);
        }
        break;
    }

    tensor->free = free_tensor;
    tensor->print = print_tensor;

    return tensor;
}

// Public API implementations
Tensor *tensor(const double *data, const size_t *shape, size_t ndim)
{
    return tensor_init(data, shape, ndim, TENSOR_WITH_DATA);
}

Tensor *empty_tensor(const size_t *shape, size_t ndim)
{
    return tensor_init(NULL, shape, ndim, TENSOR_UNINITIALIZED);
}

Tensor *zeros_tensor(const size_t *shape, size_t ndim)
{
    return tensor_init(NULL, shape, ndim, TENSOR_ZEROS);
}

Tensor *ones_tensor(const size_t *shape, size_t ndim)
{
    return tensor_init(NULL, shape, ndim, TENSOR_ONES);
}

Tensor *randn_tensor(const size_t *shape, size_t ndim)
{
    return tensor_init(NULL, shape, ndim, TENSOR_RANDN);
}

Tensor *rand_tensor(const size_t *shape, size_t ndim)
{
    return tensor_init(NULL, shape, ndim, TENSOR_RAND);
}