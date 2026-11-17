import numpy as np

from columns_ordering_algorithms.permutation import Permutation, inplace_swap_cols


def repeated_random_sampling(
    A: np.ndarray, b: int, omegas: np.ndarray | None = None
) -> Permutation:
    """
    Randomized block variation of Householder V-BLAST with oversampling.
    Projects the matrix onto an (b + 6) dimensional subspace to approximate norms,
    revealing 'b' columns per outer loop, then applying a BLAS-3 block update.

    :param A: A real matrix with m rows and n columns
    :param b: The block size (number of columns to reveal per outer iteration)
    :param omegas: Optional dependency injection for the random matrices. Should be of size ((n + b - 1) // b, b+p    , n)
    :return: A permutation of [0,...,n-1] giving the V-BLAST order
    """
    m_rows, n = A.shape
    if m_rows < n:
        raise ValueError(
            f"Matrix must have m >= n (got {m_rows}x{n}) for valid QR inversion."
        )

    best_permutation = Permutation(length=n)

    # Perform initial QR and inversion
    Q, R = np.linalg.qr(A)
    L = np.linalg.inv(R).T

    l_dim = b + 6
    columns_found = 0
    block_number = 0

    while columns_found < n:
        m = n - columns_found
        b_step = min(m, b)

        # 1. SKETCHING: Project active part of L onto a random subspace
        # Because we process from right to left, the active submatrix is always the top-left m x m
        if omegas is not None:
            omega = omegas[block_number, :l_dim, :m]
        else:
            omega = np.random.normal(loc=0.0, scale=1.0, size=(l_dim, m))
        B = omega @ L[:m, :m]

        # 2. FIND PANEL: Identify the best b_step columns using the sketch B
        for i in range(b_step):
            cur_col = m - 1 - i
            cur_l = l_dim - i

            # Submatrix of B we evaluate
            B_sub = B[:cur_l, : cur_col + 1]

            # Find column with minimum norm in the sketch
            l2_norms_sq = np.sum(B_sub**2, axis=0)
            p = np.argmin(l2_norms_sq)

            # Swap the chosen column into position in B, L, and our permutation tracker
            inplace_swap_cols(B, p, cur_col)
            inplace_swap_cols(L, p, cur_col)
            best_permutation.swap(p, cur_col)

            # Apply Householder on B to logically zero out the column in the sketch
            x = B[:cur_l, cur_col]
            norm_x = np.linalg.norm(x)

            if norm_x > 0:
                alpha = np.copysign(norm_x, x[-1])
                v = x.copy()
                v[-1] += alpha
                v_norm = np.linalg.norm(v)

                if v_norm > 0:
                    v /= v_norm
                    # Update remaining columns in sketch
                    if cur_col > 0:
                        B_rest = B[:cur_l, :cur_col]
                        B_rest -= 2.0 * np.outer(v, v @ B_rest)

                    # Explicitly set the target column to avoid floating-point drift
                    B[: cur_l - 1, cur_col] = 0.0
                    B[cur_l - 1, cur_col] = -alpha

        # 3. BUILD BLOCK REFLECTOR: Process the selected panel in L natively
        # We accumulate the Householder vectors into V and the compact representation into T
        V = np.zeros((m, b_step))
        T = np.zeros((b_step, b_step))

        for i in range(b_step):
            cur_col = m - 1 - i

            x = L[: cur_col + 1, cur_col]
            norm_x = np.linalg.norm(x)

            if norm_x > 0:
                alpha = np.copysign(norm_x, x[-1])
                v = x.copy()
                v[-1] += alpha
                v_norm = np.linalg.norm(v)

                if v_norm > 0:
                    v /= v_norm
                    V[: cur_col + 1, i] = v  # Store in block V

                    # Immediate panel update (BLAS-2): Apply v to the remaining columns INSIDE the panel
                    if cur_col > m - b_step:
                        panel_rest = L[: cur_col + 1, m - b_step : cur_col]
                        panel_rest -= 2.0 * np.outer(v, v @ panel_rest)

                    # Explicitly set the target column mathematically to prevent floating-point noise
                    L[:cur_col, cur_col] = 0.0
                    L[cur_col, cur_col] = -alpha

                    # Incrementally build the T matrix
                    if i == 0:
                        T[0, 0] = 2.0
                    else:
                        v_i = V[:, i]
                        V_prev = V[:, :i]
                        T_prev = T[:i, :i]
                        T[i, :i] = -2.0 * (v_i @ V_prev) @ T_prev
                        T[i, i] = 2.0

        # 4. BLAS-3 TRAILING UPDATE: Update all remaining columns simultaneously
        # Using Q = I - V T V^T
        if m > b_step:
            trailing = L[:m, : m - b_step]
            W = V @ T

            # Massive Matrix-Matrix update directly handled by highly optimized C/Fortran
            trailing -= W @ (V.T @ trailing)

        columns_found += b_step
        block_number += 1

    return best_permutation
