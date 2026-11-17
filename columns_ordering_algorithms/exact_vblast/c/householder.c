#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "mkl_lapacke.h"

void householder_full_mkl(const double *A_in, int m, int n, int *perm) {
    size_t n_size = (size_t)n;
    double *A_work = (double *)malloc((size_t)m * n_size * sizeof(double));
    memcpy(A_work, A_in, (size_t)m * n_size * sizeof(double));

    // 1. In-place QR Decomposition (LAPACKE)
    double *tau = (double *)malloc(n_size * sizeof(double));
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A_work, n, tau);
    free(tau);

    // 2. Zero lower triangle -> R
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            A_work[i * n + j] = 0.0;
        }
    }

    // 3. Invert R in-place (Upper Triangular Inverse)
    LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', n, A_work, n);

    // 4. Form L = (R^-1)^T
    double *L = (double *)malloc(n_size * n_size * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[j * n + i] = A_work[i * n + j];
        }
    }
    free(A_work);

    // 5. Householder V-BLAST Loop
    double *l2_norms_sq = (double *)malloc(n_size * sizeof(double));
    double *v = (double *)malloc(n_size * sizeof(double));
    double *work = (double *)malloc(n_size * sizeof(double));

    for (int j = 0; j < n; j++) {
        perm[j] = j;
        l2_norms_sq[j] = 0.0;
        for (int i = 0; i < n; i++) {
            double val = L[i * n + j];
            l2_norms_sq[j] += val * val;
        }
    }

    for (int k = n - 1; k >= 0; k--) {
        int p = 0;
        double min_norm = l2_norms_sq[0];
        for (int j = 1; j <= k; j++) {
            if (l2_norms_sq[j] < min_norm) {
                min_norm = l2_norms_sq[j];
                p = j;
            }
        }

        int tmp_p = perm[k]; perm[k] = perm[p]; perm[p] = tmp_p;
        double tmp_norm = l2_norms_sq[k]; l2_norms_sq[k] = l2_norms_sq[p]; l2_norms_sq[p] = tmp_norm;

        if (p != k) {
            cblas_dswap(n, &L[p], n, &L[k], n);
        }

        double norm_x = sqrt(l2_norms_sq[k]);
        if (norm_x > 0.0) {
            double x_k = L[k * n + k];
            double alpha = copysign(norm_x, x_k);

            double v_norm_sq = 0.0;
            for (int i = 0; i <= k; i++) {
                v[i] = L[i * n + k];
                if (i == k) v[i] += alpha;
                v_norm_sq += v[i] * v[i];
            }

            if (v_norm_sq > 0.0) {
                double v_norm = sqrt(v_norm_sq);
                for (int i = 0; i <= k; i++) v[i] /= v_norm;

                cblas_dgemv(CblasRowMajor, CblasTrans, k + 1, n, 1.0, L, n, v, 1, 0.0, work, 1);
                cblas_dger(CblasRowMajor, k + 1, n, -2.0, v, 1, work, 1, L, n);
            }
        }

        if (k > 0) {
            for (int j = 0; j < k; j++) {
                double val = L[k * n + j];
                l2_norms_sq[j] -= val * val;
                if (l2_norms_sq[j] < 0.0) l2_norms_sq[j] = 0.0;
            }
        }
    }

    free(l2_norms_sq);
    free(v);
    free(work);
    free(L);
}