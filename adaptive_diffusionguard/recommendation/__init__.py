"""Candidate generation interfaces."""

from .base import Candidate, CandidateGenerator
from .candidate_generator import OASISCandidateGenerator

__all__ = ["Candidate", "CandidateGenerator", "OASISCandidateGenerator"]
