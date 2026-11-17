import random
from typing import List, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator


class Permutation(BaseModel):
    length: int = Field(gt=0, description="The size of the permutation set")
    elements: List[int] = Field(description="The permutation sequence")

    @model_validator(mode="before")
    @classmethod
    def handle_identity_creation(cls, data: dict | Self) -> Self:
        if isinstance(data, dict):
            length = data.get("length")
            if length is not None and "elements" not in data:
                data["elements"] = list(range(length))
        return data

    @classmethod
    def identity(cls, length: int) -> Self:
        """Generates the identity permutation of a given length."""
        if length <= 0:
            raise ValueError("Length must be greater than 0.")

        # Create an ordered identity list [0, 1, ..., length-1]
        elements = list(range(length))

        return cls(length=length, elements=elements)

    @model_validator(mode="after")
    def validate_permutation(self) -> Self:
        if len(self.elements) != self.length:
            raise ValueError(
                f"Elements length ({len(self.elements)}) must match length ({self.length})"
            )

        if sorted(self.elements) != list(range(self.length)):
            raise ValueError(
                "Invalid permutation: must contain all integers from 0 to length-1 exactly once."
            )
        return self

    def swap(self, i: int, j: int) -> None:
        """
        Swaps the values at two indices in-place.
        """
        if not (0 <= i < self.length and 0 <= j < self.length):
            raise IndexError(
                "Indices are out of bounds for the current permutation length."
            )

        # Perform the swap in-place
        self.elements[i], self.elements[j] = self.elements[j], self.elements[i]

    @classmethod
    def random(cls, length: int) -> Self:
        """Generates a random permutation of a given length."""
        if length <= 0:
            raise ValueError("Length must be greater than 0.")

        # Create an ordered identity list [0, 1, ..., length-1]
        elements = list(range(length))
        # Shuffle the list in-place (Fisher-Yates shuffle algorithm under the hood)
        random.shuffle(elements)

        return cls(length=length, elements=elements)

    def compose(self, other: "Permutation") -> "Permutation":
        """
        Composes this permutation (`self`) with a prior permutation (`other`).

        `other` is applied first, then `self` is applied to `other`'s result.

        Requirements:
          - `self.length <= other.length` (raises ValueError otherwise).
          - `self` acts on the first `self.length` elements of `other.elements`,
            leaving any remaining trailing elements of `other` in place.

        Example:
          other = Permutation(length=3, elements=[2, 1, 0])
          self  = Permutation(length=2, elements=[1, 0])
          self.compose(other) -> Permutation(length=3, elements=[1, 2, 0])
        """
        if self.length > other.length:
            raise ValueError(
                f"The permutation applied second (length {self.length}) cannot be larger "
                f"than the permutation applied first (length {other.length})."
            )

        # Start with a copy of the elements from the first permutation
        result = list(other.elements)

        # Permute only the first self.length elements according to self
        # self.elements[i] tells us which element from result's prefix to move to index i
        prefix_copy = result[: self.length]
        for i in range(self.length):
            result[i] = prefix_copy[self.elements[i]]

        return Permutation(length=other.length, elements=result)


def inplace_swap_cols(A: np.ndarray, c1: int, c2: int) -> None:
    """
    In place swaps columns c1 and c2 of A
    """
    tmp = np.copy(A[:, c1])
    A[:, c1] = A[:, c2]
    A[:, c2] = tmp


def permute_matrix_columns(matrix: np.ndarray, perm: Permutation) -> np.ndarray:
    """
    Returns a new matrix with columns reordered according to the permutation.

    Args:
        matrix: A 2D numpy array.
        perm: An instance of the Permutation class.

    Returns:
        A new np.ndarray with permuted columns.
    """
    # Verify dimensions are compatible
    if matrix.shape[1] != perm.length:
        raise ValueError(
            f"Matrix columns ({matrix.shape[1]}) must match "
            f"permutation length ({perm.length})."
        )

    # Use NumPy fancy indexing to return a new matrix with permuted columns
    return matrix[:, perm.elements]
