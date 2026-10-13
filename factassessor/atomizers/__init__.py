"""Atomizers: text -> self-contained atomic claims. `Atomizer` is the role; pick one with
`FactAssessor(atomizer=...)`."""

from factassessor.atomizers._base import Atomizer
from factassessor.atomizers.llm import DEFAULT_MODEL, LLMAtomizer

__all__ = ["Atomizer", "DEFAULT_MODEL", "LLMAtomizer"]
