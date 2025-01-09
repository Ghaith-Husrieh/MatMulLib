#pragma once
#include <stddef.h>

/**
 * @brief Generates a random number from a uniform distribution.
 *
 * This function returns a random number uniformly distributed between
 * the specified low and high values.
 *
 * @param low The lower bound of the uniform distribution.
 * @param high The upper bound of the uniform distribution.
 *
 * @return A random number uniformly distributed between low and high.
 *         If low is greater than high, an error message is printed,
 *         and NAN is returned.
 */
double uniform(double low, double high);

/**
 * @brief Generates a random number from a normal distribution.
 *
 * This function returns a random number drawn from a normal (Gaussian)
 * distribution with the specified mean and standard deviation.
 *
 * @param mean The mean (average) of the normal distribution.
 * @param stddev The standard deviation of the normal distribution.
 *
 * @return A random number drawn from the specified normal distribution.
 *         The function avoids log(0) by ensuring u1 is not zero.
 */
double normal(double mean, double stddev);