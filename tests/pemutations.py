import pytest

from columns_ordering_algorithms.permutation import Permutation


def test_composition():
    """Applying [2, 1, 0] first, then [1, 0] second returns [1, 2, 0]."""
    p_first = Permutation(length=3, elements=[2, 1, 0])
    p_second = Permutation(length=2, elements=[1, 0])

    result = p_second.compose(p_first)

    assert result.length == 3
    assert result.elements == [1, 2, 0]


def test_compose_equal_lengths():
    """Standard composition when both permutations have the same length."""
    p_first = Permutation(length=3, elements=[2, 0, 1])
    p_second = Permutation(length=3, elements=[1, 2, 0])

    # p_first: [2, 0, 1]
    # p_second maps index 0<-1, 1<-2, 2<-0
    # Result: [p_first[1], p_first[2], p_first[0]] = [0, 1, 2]
    result = p_second.compose(p_first)

    assert result.length == 3
    assert result.elements == [0, 1, 2]


def test_compose_partial_prefix_leaves_suffix_unchanged():
    """Ensure elements beyond second permutation's length remain untouched."""
    p_first = Permutation(length=5, elements=[4, 3, 2, 1, 0])
    p_second = Permutation(length=3, elements=[2, 0, 1])

    # Prefix of p_first is [4, 3, 2]
    # p_second permutes prefix [4, 3, 2] with [2, 0, 1] -> [2, 4, 3]
    # Trailing elements [1, 0] remain as-is
    result = p_second.compose(p_first)

    assert result.length == 5
    assert result.elements == [2, 4, 3, 1, 0]


def test_compose_invalid_larger_second_raises_error():
    """Applying a larger permutation second must raise a ValueError."""
    p_first = Permutation(length=2, elements=[1, 0])
    p_second = Permutation(length=4, elements=[3, 2, 1, 0])

    with pytest.raises(ValueError, match="cannot be larger than"):
        p_second.compose(p_first)


def test_compose_immutability():
    """Verify that calling compose() does not mutate either original Permutation instance."""
    p_first = Permutation(length=3, elements=[2, 1, 0])
    p_second = Permutation(length=2, elements=[1, 0])

    # Keep references to original state
    first_elements_before = list(p_first.elements)
    second_elements_before = list(p_second.elements)

    # Perform composition
    result = p_second.compose(p_first)

    # Assert original instances remain untouched
    assert p_first.elements == first_elements_before
    assert p_second.elements == second_elements_before

    # Assert result is a new distinct object
    assert result is not p_first
    assert result is not p_second
