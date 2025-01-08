#pragma once
#include <stddef.h>

struct ndarray
{
    size_t ndim;
    size_t *shape;
    double *buffer;

    void (*free)(struct ndarray *self);
    void (*print)(const struct ndarray *self);
};

typedef struct ndarray Tensor;

enum TensorInitMode
{
    TENSOR_WITH_DATA,     // Initialize with provided data
    TENSOR_UNINITIALIZED, // Leave memory uninitialized
    TENSOR_ZEROS,         // Initialize all elements to 0
    TENSOR_ONES           // Initialize all elements to 1
};

/**
 * @brief Creates a multi-dimensional tensor initialized with user-provided data.
 *
 * Allocates memory for a tensor, including its shape and data buffer, and initializes
 * the data with the given values. The shape and data must match the specified number
 * of dimensions (`ndim`).
 *
 * @param data Pointer to a flat array of doubles used to initialize the tensor's data.
 *             The array must contain exactly the product of dimensions in `shape` elements.
 * @param shape Pointer to an array of `size_t` specifying the size of each dimension.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 *
 * @return Pointer to a dynamically allocated `Tensor` structure, or NULL on failure.
 *         The tensor should be freed using its `free` function (`tensor->free`).
 */
Tensor *tensor(const double *data, const size_t *shape, size_t ndim);

/**
 * @brief Creates a multi-dimensional tensor with uninitialized memory.
 *
 * Allocates memory for a tensor, including its shape and data buffer, but does not
 * initialize the buffer contents. This is useful when the buffer will be immediately
 * overwritten, such as in matrix multiplication results.
 *
 * @param shape Pointer to an array of `size_t` specifying the size of each dimension.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 *
 * @return Pointer to a dynamically allocated `Tensor` structure, or NULL on failure.
 *         The tensor should be freed using its `free` function (`tensor->free`).
 */
Tensor *empty_tensor(const size_t *shape, size_t ndim);

/**
 * @brief Creates a multi-dimensional tensor initialized with zeros.
 *
 * Allocates memory for a tensor, including its shape and data buffer, and initializes
 * all elements to 0.0. This is useful when you need a zero-initialized tensor for
 * accumulation or as a starting point for calculations.
 *
 * @param shape Pointer to an array of `size_t` specifying the size of each dimension.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 *
 * @return Pointer to a dynamically allocated `Tensor` structure, or NULL on failure.
 *         The tensor should be freed using its `free` function (`tensor->free`).
 */
Tensor *zeros_tensor(const size_t *shape, size_t ndim);

/**
 * @brief Creates a multi-dimensional tensor initialized with ones.
 *
 * Allocates memory for a tensor, including its shape and data buffer, and initializes
 * all elements to 1.0. This is useful for creating identity-like tensors or as a
 * base for element-wise operations.
 *
 * @param shape Pointer to an array of `size_t` specifying the size of each dimension.
 * @param ndim Number of dimensions in the tensor. Must be greater than 0.
 *
 * @return Pointer to a dynamically allocated `Tensor` structure, or NULL on failure.
 *         The tensor should be freed using its `free` function (`tensor->free`).
 */
Tensor *ones_tensor(const size_t *shape, size_t ndim);