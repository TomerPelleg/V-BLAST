import numpy as np

from columns_ordering_algorithms.permutation import Permutation


def change_paige_paper(A: np.ndarray) -> Permutation:
    _, n = A.shape

    Q, R = np.linalg.qr(A)
    L = np.linalg.inv(R).T

    P = np.eye(n)
    final_perm = []

    for j in range(1, n + 1):
        i = (
            np.argmin(np.linalg.norm(L[0 : n - j + 1, :], ord=2, axis=0)[: n - j + 1])
            + 1
        )
        final_perm.append(i)

        new_L = np.eye(n)

        if i >= 2:
            new_L[:, 0 : i - 1] = L[:, 0 : i - 1]

        new_L[:, i - 1 : n - j] = L[:, i : n - j + 1]
        new_L[:, n - j] = L[:, i - 1]
        new_L[:, n - j + 1 : n] = L[:, n - j + 1 : n]
        new_L = np.copy(new_L)

        new_P = np.eye(n)

        if i >= 2:
            new_P[:, 0 : i - 1] = P[:, 0 : i - 1]

        new_P[:, i - 1 : n - j] = P[:, i : n - j + 1]
        new_P[:, n - j] = P[:, i - 1]
        new_P[:, n - j + 1 : n] = P[:, n - j + 1 : n]
        new_P = np.copy(new_P)

        for k in range(i, n - j + 1):
            # Givens for rows k-1, k
            a = new_L[k - 1, n - j]
            b = new_L[k, n - j]

            if abs(a) <= 1e-12:
                continue

            givens_matrix = np.eye(n)
            r = np.sqrt(a**2 + b**2)
            c = a / r
            s = -b / r

            givens_matrix[k - 1, k - 1] = s
            givens_matrix[k - 1, k] = c
            givens_matrix[k, k - 1] = c
            givens_matrix[k, k] = -s

            new_L = givens_matrix @ new_L

        L = new_L
        P = new_P

    perm = np.linalg.inv(P) @ np.asarray([i for i in range(n)], dtype=int)

    return Permutation(length=n, elements=perm.tolist())
