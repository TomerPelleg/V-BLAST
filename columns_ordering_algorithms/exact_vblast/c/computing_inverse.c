#include <math.h>
#include <stdlib.h>
#include "mkl.h"
#include "mkl_lapacke.h"

void computing_inverse_full_mkl(const double *A, int m, int n, int *perm) {
    size_t n_size = (size_t)n;
    double *M = (double *)malloc(n_size * n_size * sizeof(double));

    // 1. M = A^T * A (BLAS-3 Symmetric Rank-K Update)
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                n, m, 1.0, A, n, 0.0, M, n);

    // 2. Cholesky Inversion: M = (A^T A)^-1
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', n, M, n);
    LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U', n, M, n);

    // Fill symmetric lower triangle
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            M[i * n + j] = M[j * n + i];
        }
    }

    // 3. Schur Complement Rank-1 Update Loop
    int *active = (int *)malloc(n_size * sizeof(int));
    for (int i = 0; i < n; i++) active[i] = i;

    for (int step = 0; step < n; step++) {
        int cur_n = n - step;
        int min_idx = 0;
        double min_val = M[0];

        for (int i = 1; i < cur_n; i++) {
            if (M[i * n + i] < min_val) {
                min_val = M[i * n + i];
                min_idx = i;
            }
        }

        perm[n - 1 - step] = active[min_idx];
        int last = cur_n - 1;

        if (min_idx != last) {
            cblas_dswap(cur_n, &M[min_idx * n], 1, &M[last * n], 1);
            cblas_dswap(cur_n, &M[min_idx], n, &M[last], n);
            int tmp = active[min_idx];
            active[min_idx] = active[last];
            active[last] = tmp;
        }

        if (last > 0) {
            double m_kk = M[last * n + last];
            if (fabs(m_kk) > 1e-15) {
                cblas_dger(CblasRowMajor, last, last,
                           -1.0 / m_kk,
                           &M[last], n,
                           &M[last * n], 1,
                           M, n);
            }
        }
    }

    free(active);
    free(M);
}