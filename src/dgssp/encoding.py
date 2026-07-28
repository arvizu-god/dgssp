"""
encoding.py

Register arithmetic for the Draper--Grover Subset Sum solver.

This module is the *single source of truth* for how the (possibly signed)
partial sums of a Subset Sum instance are mapped onto the unsigned integer
register manipulated by the QFT adder.  The solver, the phase oracle and the
result decoder must all agree on this mapping, so it lives here and nowhere
else.

The mapping has two ingredients:

* **Offset encoding.**  Every reachable subset sum lies in
  ``[sum_neg, sum_pos]``.  Adding ``offset = -sum_neg`` shifts that interval to
  ``[0, range_len - 1]``, so the encoded target is always a well-defined
  non-negative bit pattern and the oracle never has to shift a negative
  Python integer.
* **Guard bit.**  The register is sized with one extra bit beyond what
  ``range_len`` strictly needs, guaranteeing ``modulus >= 2 * range_len``.
  This headroom means the encoded region never wraps around the modulus, so an
  encoded sum can never alias a *different* reachable sum.

The module is deliberately free of any Qiskit dependency: it is pure integer
arithmetic and can be unit-tested (and reasoned about) in isolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SumRegisterEncoding:
    """
    Immutable description of how subset sums are encoded in the sum register.

    Attributes
    ----------
    items:
        The instance items, in their original order.
    target:
        The *unencoded* target sum requested by the user (may be negative).
    sum_neg:
        Sum of all strictly negative items (``<= 0``).  Lower bound of the
        reachable-sum interval.
    sum_pos:
        Sum of all strictly positive items (``>= 0``).  Upper bound of the
        reachable-sum interval.
    offset:
        ``-sum_neg``; the non-negative constant added to every sum so that the
        encoded value is never negative.
    range_len:
        ``sum_pos - sum_neg + 1``; the number of integers in the reachable
        interval (an upper bound on the number of distinct reachable sums).
    n_sum:
        Number of qubits in the sum register, *including* the guard bit.
    modulus:
        ``2 ** n_sum``; the ring in which the phase adder operates.
    encoded_target:
        ``target + offset``.  Guaranteed to lie in ``[0, range_len - 1]``
        whenever the target is itself reachable.
    """

    items: list[int]
    target: int
    sum_neg: int
    sum_pos: int
    offset: int
    range_len: int
    n_sum: int
    modulus: int
    encoded_target: int

    # -- convenience -----------------------------------------------------

    @property
    def n_ind(self) -> int:
        """Number of index-register qubits (one per item)."""
        return len(self.items)

    @property
    def target_is_representable(self) -> bool:
        """
        Whether ``encoded_target`` falls inside the encodable window
        ``[0, range_len - 1]``.

        A target outside the window can never be reached by any subset, so the
        Grover search would have zero marked states.
        """
        return 0 <= self.encoded_target < self.range_len

    def encode(self, value: int) -> int:
        """
        Map a signed subset sum onto its register value.

        Parameters
        ----------
        value:
            A signed sum (typically the total of some subset of ``items``).

        Returns
        -------
        int
            ``(value + offset) mod modulus`` -- the bit pattern the sum
            register holds for that sum.
        """
        return (value + self.offset) % self.modulus

    def decode(self, register_value: int) -> int:
        """
        Inverse of :meth:`encode`.

        Parameters
        ----------
        register_value:
            An integer in ``[0, modulus)`` read out of the sum register.

        Returns
        -------
        int
            The signed subset sum it represents.
        """
        return register_value - self.offset


def build_encoding(items: Sequence[int], target: int) -> SumRegisterEncoding:
    """
    Compute the offset/guard-bit encoding for a Subset Sum instance.

    Parameters
    ----------
    items:
        Instance items.  May contain negative values, zeros and duplicates.
    target:
        Target sum.  May be negative.

    Returns
    -------
    SumRegisterEncoding
        A frozen record with the register width, modulus, offset and encoded
        target.

    Raises
    ------
    ValueError
        If ``items`` is empty or ``range_len`` is non-positive (which cannot
        happen for a well-formed instance, but is checked defensively).
    """
    items_list = [int(a) for a in items]
    if not items_list:
        raise ValueError("build_encoding requires at least one item.")

    sum_neg = sum(a for a in items_list if a < 0)
    sum_pos = sum(a for a in items_list if a > 0)
    offset = -sum_neg  # >= 0
    range_len = sum_pos - sum_neg + 1

    if range_len <= 0:
        raise ValueError(
            f"Invalid reachable range: sum_pos={sum_pos}, sum_neg={sum_neg}."
        )

    n_core = max(1, math.ceil(math.log2(range_len)))
    n_sum = n_core + 1  # guard bit: guarantees modulus >= 2 * range_len
    modulus = 2**n_sum
    encoded_target = int(target) + offset

    return SumRegisterEncoding(
        items=items_list,
        target=int(target),
        sum_neg=sum_neg,
        sum_pos=sum_pos,
        offset=offset,
        range_len=range_len,
        n_sum=n_sum,
        modulus=modulus,
        encoded_target=encoded_target,
    )


__all__ = [
    "SumRegisterEncoding",
    "build_encoding",
]
