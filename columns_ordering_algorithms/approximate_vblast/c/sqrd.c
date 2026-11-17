#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"

void sqrd(const double *A_in, int m, int n, int *perm) {
    size_t n_size = (size_t)n;
    size_t m_size = (size_t)m;

    // 1. Allocate R and copy input A into it
    double *R = (double *)malloc(m_size * n_size * sizeof(double));
    memcpy(R, A_in, m_size * n_size * sizeof(double));

    // Allocate workspace arrays
    double *l2_norms_sq = (double *)malloc(n_size * sizeof(double));
    double *v = (double *)malloc(m_size * sizeof(double));
    double *work = (double *)malloc(n_size * sizeof(double));

    // 2. Initialize permutation and squared column norms
    for (int j = 0; j < n; j++) {
        perm[j] = j;
        l2_norms_sq[j] = 0.0;
        for (int i = 0; i < m; i++) {
            double val = R[i * n + j];
            l2_norms_sq[j] += val * val;
        }
    }

    // 3. Main SQRD Loop (Left-to-Right)
    for (int cur_col = 0; cur_col < n; cur_col++) {

        // Find column with the minimum squared norm in the remaining columns
        int p = cur_col;
        double min_norm = l2_norms_sq[cur_col];
        for (int j = cur_col + 1; j < n; j++) {
            if (l2_norms_sq[j] < min_norm) {
                min_norm = l2_norms_sq[j];
                p = j;
            }
        }

        // Swap the selected column into the current position
        if (p != cur_col) {
            // Swap columns in R (Length: m, Stride: n)
            cblas_dswap(m, &R[p], n, &R[cur_col], n);

            int tmp_p = perm[cur_col];
            perm[cur_col] = perm[p];
            perm[p] = tmp_p;

            double tmp_n = l2_norms_sq[cur_col];
            l2_norms_sq[cur_col] = l2_norms_sq[p];
            l2_norms_sq[p] = tmp_n;
        }

        // Early exit if we exceed the row dimension (for fat matrices)
        if (cur_col == m) {
            break;
        }

        // Apply Householder Reflection
        int len = m - cur_col;
        double norm_x = 0.0;

        // Extract vector x = R[cur_col:, cur_col]
        for (int i = 0; i < len; i++) {
            v[i] = R[(cur_col + i) * n + cur_col];
            norm_x += v[i] * v[i];
        }
        norm_x = sqrt(norm_x);

        if (norm_x > 0.0) {
            double alpha = copysign(norm_x, v[0]);
            v[0] += alpha;

            double v_norm_sq = 0.0;
            for (int i = 0; i < len; i++) {
                v_norm_sq += v[i] * v[i];
            }

            if (v_norm_sq > 0.0) {
                double v_norm = sqrt(v_norm_sq);
                for (int i = 0; i < len; i++) {
                    v[i] /= v_norm;
                }

                // Submatrix dimensions to update
                int cols_to_update = n - cur_col;

                // BLAS-2: work = v^T * R[cur_col:, cur_col:]
                cblas_dgemv(CblasRowMajor, CblasTrans,
                            len, cols_to_update,
                            1.0, &R[cur_col * n + cur_col], n,
                            v, 1,
                            0.0, work, 1);

                // BLAS-2: R[cur_col:, cur_col:] -= 2.0 * v * work^T
                cblas_dger(CblasRowMajor,
                           len, cols_to_update,
                           -2.0, v, 1,
                           work, 1,
                           &R[cur_col * n + cur_col], n);
            }
        }

        // Downdate the norms for the remaining columns
        if (cur_col < n - 1) {
            for (int j = cur_col + 1; j < n; j++) {
                double val = R[cur_col * n + j];
                l2_norms_sq[j] -= val * val;

                // Clip to 0 to prevent negative floating-point drift
                if (l2_norms_sq[j] < 0.0) {
                    l2_norms_sq[j] = 0.0;
                }
            }
        }
    }

    free(l2_norms_sq);
    free(v);
    free(work);
    free(R);
}