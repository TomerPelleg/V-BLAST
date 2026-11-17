#include <stdlib.h>
#include <string.h>
#include "mkl.h"
#include "mkl_lapacke.h"

void qr_blocked_mkl(const double *A_in, int m, int n, int *perm) {
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;

    double *A_work = (double *)malloc(m_size * n_size * sizeof(double));
    memcpy(A_work, A_in, m_size * n_size * sizeof(double));

    int k = (m < n) ? m : n;
    double *tau = (double *)malloc((size_t)k * sizeof(double));

    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A_work, n, tau);

    // Return the identity permutation
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }

    free(tau);
    free(A_work);
}

void qr_unblocked_mkl(const double *A_in, int m, int n, int *perm) {
    size_t m_size = (size_t)m;
    size_t n_size = (size_t)n;

    double *A_work = (double *)malloc(m_size * n_size * sizeof(double));
    memcpy(A_work, A_in, m_size * n_size * sizeof(double));

    int k = (m < n) ? m : n;
    double *tau = (double *)malloc((size_t)k * sizeof(double));

    LAPACKE_dgeqr2(LAPACK_ROW_MAJOR, m, n, A_work, n, tau);

    // Return the identity permutation
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }

    free(tau);
    free(A_work);
}