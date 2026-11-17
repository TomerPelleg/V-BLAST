from enum import Enum

import numpy as np

from columns_ordering_algorithms.approximate_vblast.c_wrapper import (
    repeated_random_sampling_c,
    sqrd_c,
)
from columns_ordering_algorithms.exact_vblast.c_wrapper import (
    chang_paige_vblast_c,
    householder_vblast_c,
    vblast_by_computing_inverse_c,
)
from columns_ordering_algorithms.performence_comparison_qr.c_wrapper import (
    qr_blocked_c,
    qr_unblocked_c,
)
from columns_ordering_algorithms.permutation import Permutation


class PermutationGenerationMethods(Enum):
    TRUE_RANDOM = "Random"
    REPEATED_RANDOM_SAMPLING = "Random Sampling"
    VBLAST = "V-BLAST"
    SQRD = "SQRD"
    # The following should return the same permutation as VBLAST, but with specific implementations
    # These are used for counting times
    ChangPaige = "Chang Paige"
    Householder = "Householder"
    DOWNDATING = "Downdating"
    # Native QR returns the identity permutation, but computes a QR decomposition
    # This is only for timing concerns
    QR_BLOCKED = "Blocked QR (dgeqrf)"
    QR_UNBLOCKED = "Unblocked QR (dgeqr2)"


def compute_permutation(
    A: np.ndarray, method: PermutationGenerationMethods
) -> Permutation:
    n, m = A.shape
    match method:
        case PermutationGenerationMethods.TRUE_RANDOM:
            return Permutation.random(length=n)
        case PermutationGenerationMethods.VBLAST:
            return vblast_by_computing_inverse_c(A)
        case PermutationGenerationMethods.REPEATED_RANDOM_SAMPLING:
            return repeated_random_sampling_c(A, b=16)
        case PermutationGenerationMethods.ChangPaige:
            return chang_paige_vblast_c(A)
        case PermutationGenerationMethods.Householder:
            return householder_vblast_c(A)
        case PermutationGenerationMethods.DOWNDATING:
            return vblast_by_computing_inverse_c(A)
        case PermutationGenerationMethods.QR_BLOCKED:
            return qr_blocked_c(A)
        case PermutationGenerationMethods.QR_UNBLOCKED:
            return qr_unblocked_c(A)
        case PermutationGenerationMethods.SQRD:
            return sqrd_c(A)
        case _:
            raise AssertionError("Shouldn't have reached this unreachable code.")


def get_permutation_marker(method: PermutationGenerationMethods) -> str:
    match method:
        case PermutationGenerationMethods.TRUE_RANDOM:
            return "<"
        case PermutationGenerationMethods.VBLAST:
            return "p"
        case PermutationGenerationMethods.REPEATED_RANDOM_SAMPLING:
            return "D"
        case PermutationGenerationMethods.ChangPaige:
            return "o"
        case PermutationGenerationMethods.Householder:
            return "s"
        case PermutationGenerationMethods.DOWNDATING:
            return "p"
        case PermutationGenerationMethods.QR_BLOCKED:
            return "h"
        case PermutationGenerationMethods.QR_UNBLOCKED:
            return "X"
        case PermutationGenerationMethods.SQRD:
            return "X"
        case _:
            raise AssertionError("Shouldn't have reached this unreachable code.")


def get_permutation_color_index(method: PermutationGenerationMethods) -> int:
    match method:
        case PermutationGenerationMethods.TRUE_RANDOM:
            return 7
        case PermutationGenerationMethods.VBLAST:
            return 0
        case PermutationGenerationMethods.REPEATED_RANDOM_SAMPLING:
            return 3
        case PermutationGenerationMethods.ChangPaige:
            return 9
        case PermutationGenerationMethods.Householder:
            return 1
        case PermutationGenerationMethods.DOWNDATING:
            return 0
        case PermutationGenerationMethods.QR_BLOCKED:
            return 5
        case PermutationGenerationMethods.QR_UNBLOCKED:
            return 2
        case PermutationGenerationMethods.SQRD:
            return 6
        case _:
            raise AssertionError("Shouldn't have reached this unreachable code.")
