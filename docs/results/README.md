# Sanitized result summaries

This directory contains release-safe, machine-readable aggregates extracted
from frozen local runs. It intentionally excludes databases, response caches,
prompts, rationales, profiles, logs, failed generations and credentials.

- [`groq_pilots.json`](groq_pilots.json) records protocol-specific engineering
  outcomes for the v1, v2 and v2.1 Groq pilots. Results are not pooled across
  protocol versions and failed decisions are not imputed.
- [`cosref_validation.json`](cosref_validation.json) separates the official
  COSREF reference-oracle check from project-specific exposure calibration,
  robustness and threshold-response findings.

Each entry points to an ignored local source under `runs/` and records its
SHA-256. The source paths support a local audit but the underlying research
artifacts are deliberately not release candidates. A missing ignored source in
a fresh clone is therefore expected; the checked aggregate remains usable as
the public record of the frozen result.

These summaries describe small synthetic experiments. They do not establish
real-human behavioral validity, production reliability, real-world governance
effectiveness or causal superiority. The project used oracle synthetic risk
labels, and `paper_omega_*` is not equivalent to `oasis_keep_*`.
