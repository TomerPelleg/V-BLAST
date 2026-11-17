#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "mkl_lapacke.h"

void repeated_random_sampling_full_mkl(
    const double *A_in, 
    int m_rows, 
    int n, 
    int b, 
    unsigned int seed, 
    const double *omegas_in, 
    int *perm
) {
    size_t n_size = (size_t)n;
    double *A_work = (double *)malloc((size_t)m_rows * n_size * sizeof(double));
    memcpy(A_work, A_in, (size_t)m_rows * n_size * sizeof(double));

    // 1. QR & Triangular Inverse
    double *tau = (double *)malloc(n_size * sizeof(double));
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m_rows, n, A_work, n, tau);
    free(tau);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) A_work[i * n + j] = 0.0;
    }

    LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', n, A_work, n);

    double *L = (double *)malloc(n_size * n_size * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) L[j * n + i] = A_work[i * n + j];
    }
    free(A_work);

    int l_dim = b + 6;
    for (int i = 0; i < n; i++) perm[i] = i;

    VSLStreamStatePtr stream = NULL;
    double *local_omega = NULL;

    if (omegas_in == NULL) {
        vslNewStream(&stream, VSL_BRNG_MT19937, seed);
        local_omega = (double *)malloc((size_t)l_dim * n_size * sizeof(double));
    }

    double *B = (double *)calloc((size_t)l_dim * n, sizeof(double));
    double *V = (double *)calloc(n_size * (size_t)b, sizeof(double));
    double *T = (double *)calloc((size_t)b * (size_t)b, sizeof(double));

    int max_dim = (n > l_dim) ? n : l_dim;
    double *v = (double *)malloc((size_t)max_dim * sizeof(double));
    double *w = (double *)malloc((size_t)max_dim * sizeof(double));

    int columns_found = 0;
    int step = 0;

    while (columns_found < n) {
        int m_active = n - columns_found;
        int b_step = (m_active < b) ? m_active : b;

        memset(V, 0, n_size * (size_t)b * sizeof(double));
        memset(T, 0, (size_t)b * (size_t)b * sizeof(double));

        const double *current_omega;
        int omega_stride;

        if (omegas_in != NULL) {
            current_omega = &omegas_in[step * l_dim * n];
            omega_stride = n;
        } else {
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, (MKL_INT)(l_dim * m_active), local_omega, 0.0, 1.0);
            current_omega = local_omega;
            omega_stride = m_active;
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    l_dim, m_active, m_active,
                    1.0, current_omega, omega_stride, L, n,
                    0.0, B, n);

        for (int i = 0; i < b_step; i++) {
            int cur_col = m_active - 1 - i;
            int cur_l = l_dim - i;

            int p = 0;
            double min_norm = cblas_ddot(cur_l, &B[0], n, &B[0], n);
            for (int j = 1; j <= cur_col; j++) {
                double norm = cblas_ddot(cur_l, &B[j], n, &B[j], n);
                if (norm < min_norm) { min_norm = norm; p = j; }
            }

            if (p != cur_col) {
                cblas_dswap(cur_l, &B[p], n, &B[cur_col], n);
                cblas_dswap(m_active, &L[p], n, &L[cur_col], n);
                int tmp = perm[p]; perm[p] = perm[cur_col]; perm[cur_col] = tmp;
            }

            double norm_x = cblas_dnrm2(cur_l, &B[cur_col], n);
            if (norm_x > 0.0) {
                cblas_dcopy(cur_l, &B[cur_col], n, v, 1);
                double alpha = copysign(norm_x, v[cur_l - 1]);
                v[cur_l - 1] += alpha;
                double vn = cblas_dnrm2(cur_l, v, 1);
                if (vn > 0.0) {
                    cblas_dscal(cur_l, 1.0 / vn, v, 1);
                    if (cur_col > 0) {
                        cblas_dgemv(CblasRowMajor, CblasTrans, cur_l, cur_col, 1.0, B, n, v, 1, 0.0, w, 1);
                        cblas_dger(CblasRowMajor, cur_l, cur_col, -2.0, v, 1, w, 1, B, n);
                    }
                }
            }
        }

        // Build Block Reflector
        for (int i = 0; i < b_step; i++) {
            int cur_col = m_active - 1 - i;
            cblas_dcopy(cur_col + 1, &L[cur_col], n, v, 1);
            double norm_x = cblas_dnrm2(cur_col + 1, v, 1);
            if (norm_x <= 0.0) continue;

            double alpha = copysign(norm_x, v[cur_col]);
            v[cur_col] += alpha;
            double vn = cblas_dnrm2(cur_col + 1, v, 1);
            if (vn <= 0.0) continue;

            cblas_dscal(cur_col + 1, 1.0 / vn, v, 1);
            cblas_dcopy(cur_col + 1, v, 1, &V[i], b);

            if (cur_col > m_active - b_step) {
                int start = m_active - b_step;
                int count = cur_col - start;
                cblas_dgemv(CblasRowMajor, CblasTrans, cur_col + 1, count, 1.0, &L[start], n, v, 1, 0.0, w, 1);
                cblas_dger(CblasRowMajor, cur_col + 1, count, -2.0, v, 1, w, 1, &L[start], n);
            }

            for (int r = 0; r < cur_col; r++) L[r * n + cur_col] = 0.0;
            L[cur_col * n + cur_col] = -alpha;
            T[i * b + i] = 2.0;

            if (i > 0) {
                cblas_dgemv(CblasRowMajor, CblasTrans, m_active, i, 1.0, V, b, &V[i], b, 0.0, w, 1);
                cblas_dgemv(CblasRowMajor, CblasTrans, i, i, -2.0, T, b, w, 1, 0.0, &T[i * b], 1);
            }
        }

        if (m_active > b_step) {
            int remaining = m_active - b_step;
            double *W_mat = (double *)calloc((size_t)m_active * (size_t)b_step, sizeof(double));
            double *tmp_mat = (double *)calloc((size_t)b_step * (size_t)remaining, sizeof(double));

            if (W_mat && tmp_mat) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_active, b_step, b_step, 1.0, V, b, T, b, 0.0, W_mat, b_step);
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, b_step, remaining, m_active, 1.0, V, b, L, n, 0.0, tmp_mat, remaining);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_active, remaining, b_step, -1.0, W_mat, b_step, tmp_mat, remaining, 1.0, L, n);
            }
            free(tmp_mat);
            free(W_mat);
        }

        columns_found += b_step;
        step++;
    }

    if (stream) vslDeleteStream(&stream);
    if (local_omega) free(local_omega);
    free(w); free(v); free(T); free(V); free(B); free(L);
}