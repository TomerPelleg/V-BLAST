#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "mkl_lapacke.h"

void chang_paige_full_mkl(const double *A_in, int m, int n, int k_cols, int *perm) {
    size_t n_size = (size_t)n;
    double *A_work = (double *)malloc((size_t)m * n_size * sizeof(double));
    memcpy(A_work, A_in, (size_t)m * n_size * sizeof(double));

    // 1. QR Decomposition
    double *tau = (double *)malloc(n_size * sizeof(double));
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A_work, n, tau);
    free(tau);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            A_work[i * n + j] = 0.0;
        }
    }

    // 2. Triangular Inverse & Transpose to L
    LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', n, A_work, n);

    double *L = (double *)malloc(n_size * n_size * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            L[j * n + i] = A_work[i * n + j];
        }
    }
    free(A_work);

    // 3. Chang-Paige Givens Loop
    double *l2_norms_sq = (double *)malloc(n_size * sizeof(double));

    for (int j = 0; j < n; j++) {
        perm[j] = j;
        l2_norms_sq[j] = 0.0;
        for (int i = 0; i < n; i++) {
            double val = L[i * n + j];
            l2_norms_sq[j] += val * val;
        }
    }

    for (int cur_col = n - 1; cur_col >= n - k_cols; cur_col--) {
        int p = 0;
        double min_norm = l2_norms_sq[0];
        for (int j = 1; j <= cur_col; j++) {
            if (l2_norms_sq[j] < min_norm) {
                min_norm = l2_norms_sq[j];
                p = j;
            }
        }

        for (int idx = p; idx < cur_col; idx++) {
            cblas_dswap(n, &L[idx], n, &L[idx + 1], n);

            int tmp_p = perm[idx]; perm[idx] = perm[idx + 1]; perm[idx + 1] = tmp_p;
            double tmp_n = l2_norms_sq[idx]; l2_norms_sq[idx] = l2_norms_sq[idx + 1]; l2_norms_sq[idx + 1] = tmp_n;

            double a = L[idx * n + idx + 1];
            double b = L[(idx + 1) * n + idx + 1];

            if (fabs(a) > 1e-12) {
                double r = hypot(a, b);
                double c = b / r;
                double s = a / r;
                cblas_drot(n, &L[idx * n], 1, &L[(idx + 1) * n], 1, c, -s);
            }
        }

        if (cur_col > 0) {
            for (int j = 0; j < cur_col; j++) {
                double val = L[cur_col * n + j];
                l2_norms_sq[j] -= val * val;
                if (l2_norms_sq[j] < 0.0) l2_norms_sq[j] = 0.0;
            }
        }
    }

    free(l2_norms_sq);
    free(L);
}