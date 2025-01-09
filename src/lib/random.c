#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../include/random.h"

double inline uniform(double low, double high)
{
    if (low > high)
    {
        fprintf(stderr, "Error: 'high' must be greater than or equal to 'low' in uniform().\n");
        return NAN;
    }
    return low + ((double)rand() / RAND_MAX) * (high - low);
}

double inline normal(double mean, double stddev)
{
    double u1, u2;
    do
    {
        u1 = (double)rand() / RAND_MAX;
    } while (u1 == 0.0); // Avoid u1 == 0 to prevent log(0)

    u2 = (double)rand() / RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z0 * stddev;
}