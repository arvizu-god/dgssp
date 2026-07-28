"""Tests for the offset + guard-bit sum-register encoding."""

from __future__ import annotations

import itertools

import pytest

from dgssp.encoding import build_encoding

CASES = [
    ([1, 2, 3], 5),
    ([3, 5, 7, 10], 15),
    ([3, -2, 4], 1),
    ([-1, -2, -3], -4),
    ([-5, 5], 0),
    ([2, 2, 2], 4),
    ([1], 1),
    ([0, 1], 1),
]


@pytest.mark.parametrize("items,target", CASES)
def test_guard_bit_gives_headroom(items, target):
    """The guard bit guarantees modulus >= 2 * range_len."""
    enc = build_encoding(items, target)
    assert enc.modulus >= 2 * enc.range_len


@pytest.mark.parametrize("items,target", CASES)
def test_encoded_target_in_window(items, target):
    """A reachable target always encodes into [0, range_len - 1]."""
    enc = build_encoding(items, target)
    assert 0 <= enc.encoded_target < enc.range_len
    assert enc.target_is_representable


@pytest.mark.parametrize("items,target", CASES)
def test_reachable_sums_are_injective(items, target):
    """Distinct reachable sums map to distinct residues inside the window."""
    enc = build_encoding(items, target)

    residues = {}
    for mask in itertools.product([0, 1], repeat=len(items)):
        total = sum(a for a, b in zip(items, mask, strict=True) if b)
        encoded = enc.encode(total)
        assert 0 <= encoded < enc.range_len, "encoded sum escaped the window"
        residues.setdefault(encoded, total)
        assert residues[encoded] == total, "two different sums aliased"


@pytest.mark.parametrize("items,target", CASES)
def test_encode_decode_roundtrip(items, target):
    """decode(encode(x)) == x for every reachable sum."""
    enc = build_encoding(items, target)
    for mask in itertools.product([0, 1], repeat=len(items)):
        total = sum(a for a, b in zip(items, mask, strict=True) if b)
        assert enc.decode(enc.encode(total)) == total


def test_offset_is_non_negative_and_cancels_negatives():
    """The offset equals -sum_neg and lifts the minimum sum to zero."""
    enc = build_encoding([3, -2, -4, 5], 1)
    assert enc.offset == 6
    assert enc.encode(enc.sum_neg) == 0


def test_unreachable_target_is_flagged():
    """A target outside the reachable interval is reported, not silently encoded."""
    enc = build_encoding([1, 2], 99)
    assert not enc.target_is_representable


def test_empty_items_rejected():
    """An empty item list is a hard error."""
    with pytest.raises(ValueError):
        build_encoding([], 0)
