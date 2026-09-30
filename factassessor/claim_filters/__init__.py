"""Claim filters: atoms -> the factual claims worth checking. `ClaimFilter` is the role; pick one with
`FactAssessor(claim_filter=...)`, or `None` for no filter."""

from factassessor.claim_filters._base import KINDS, ClaimFilter
from factassessor.claim_filters.decision import DecisionClaimFilter

__all__ = ["ClaimFilter", "DecisionClaimFilter", "KINDS"]
