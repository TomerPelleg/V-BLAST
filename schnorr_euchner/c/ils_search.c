#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "mkl.h"
#include "mkl_lapacke.h"

// Inline for zero overhead during the tree search
static inline long long get_next_offset(long long cur_offset, bool positive_first) {
    if (positive_first) {
        return (cur_offset <= 0) ? -cur_offset + 1 : -cur_offset;
    } else {
        return (cur_offset >= 0) ? -cur_offset - 1 : -cur_offset;
    }
}

// Main C API
void ils_search_c(
    const double *H_in,
    const double *y_in,
    int m,
    int n,
    long long max_iterations,
    long long *best_x,
    long long *out_iterations,
    long long *out_candidates,
    int *out_status
) {
    // 1. Memory Allocation for QR Preprocessing & Search State
    double *R_work = (double *)malloc((size_t)m * n * sizeof(double));
    memcpy(R_work, H_in, (size_t)m * n * sizeof(double));

    double *y_tilde = (double *)malloc((size_t)m * sizeof(double));
    memcpy(y_tilde, y_in, (size_t)m * sizeof(double));

    double *tau = (double *)malloc((size_t)n * sizeof(double));

    // 2. Perform QR Decomposition (H = Q * R)
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, R_work, n, tau);

    // 3. Project y (y_tilde = Q^T * y)
    LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'T', m, 1, n, R_work, n, tau, y_tilde, 1);
    free(tau);

    // 4. Initialize Search Variables
    double best_residue = INFINITY;

    long long *rounded = (long long *)calloc(n, sizeof(long long));
    long long *offset = (long long *)calloc(n, sizeof(long long));
    long long *xLevel = (long long *)calloc(n, sizeof(long long));
    bool *directions = (bool *)malloc(n * sizeof(bool));

    double *dist = (double *)calloc(n + 1, sizeof(double));
    double *e = (double *)calloc(n, sizeof(double));

    long long level = n - 1;
    e[level] = y_tilde[level];

    double r_diag = R_work[level * n + level];
    double xReal = e[level] / r_diag;

    rounded[level] = (long long)round(xReal);
    directions[level] = (double)rounded[level] < xReal;

    long long iterations = 0;
    long long candidates = 0;
    int status_code = 1; // 1 = RUNNING, 2 = DONE, 3 = MAX_ITERATIONS_REACHED

    // 5. Core ILS Search Loop (Schnorr-Euchner)
    while (1) {
        iterations++;
        if (iterations > max_iterations) {
            status_code = 3;
            break;
        }

        xLevel[level] = rounded[level] + offset[level];

        double diff = (R_work[level * n + level] * (double)xLevel[level] - e[level]);
        double step_cost = diff * diff;
        double current_dist = dist[level + 1] + step_cost;

        if (current_dist >= best_residue || level == 0) {
            if (level == 0 && current_dist < best_residue) {
                memcpy(best_x, xLevel, n * sizeof(long long));
                best_residue = current_dist;
                candidates++;
            }

            level++;
            if (level == n) {
                status_code = 2; // DONE
                break;
            }

            offset[level] = get_next_offset(offset[level], directions[level]);
        } else {
            dist[level] = current_dist;
            level--;

            double dot_product = 0.0;
            for (int j = level + 1; j < n; j++) {
                dot_product += R_work[level * n + j] * (double)xLevel[j];
            }

            e[level] = y_tilde[level] - dot_product;
            double r_curr = R_work[level * n + level];
            double xRealNew = e[level] / r_curr;

            rounded[level] = (long long)round(xRealNew);
            offset[level] = 0;
            directions[level] = (double)rounded[level] < xRealNew;
        }
    }

    // 6. Output & Cleanup
    *out_iterations = iterations;
    *out_candidates = candidates;
    *out_status = status_code;

    free(e);
    free(dist);
    free(directions);
    free(xLevel);
    free(offset);
    free(rounded);
    free(y_tilde);
    free(R_work);
}