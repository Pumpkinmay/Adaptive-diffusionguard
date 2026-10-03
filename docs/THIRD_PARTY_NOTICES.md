# Third-party sources and notices

This document records provenance; it is not legal advice.

## OASIS

- Source: <https://github.com/camel-ai/oasis>
- Installed package: `camel-oasis==0.2.5`
- Pinned source commit: `ca8caa52aeec21e0db30926797e765ad07e75015`
- License stated by upstream: Apache License 2.0
- Citation: Ziyi Yang et al., *OASIS: Open Agent Social Interaction Simulations
  with One Million Agents*, arXiv:2411.11581 (2024).

Adaptive DiffusionGuard depends on OASIS through a pinned editable/direct Git
dependency and does not vendor or patch its source tree.

## CAMEL

- Source: <https://github.com/camel-ai/camel>
- Locked version: `camel-ai==0.2.90`
- License stated by upstream: Apache License 2.0

CAMEL supplies model backend interfaces used by the optional remote runtime.

## Groq and model service

Groq is an optional hosted inference provider. API use is subject to Groq's
service terms and model availability, not this repository's source-code
license. `openai/gpt-oss-20b` was used as an existing Groq-hosted model; this
project did not fine-tune it.

## COSREF article and reference software

- Article: Chen et al., *Community structure-regulation coupling reveals
  optimal information diffusion*, Nature Communications 17, 4879 (2026),
  <https://doi.org/10.1038/s41467-026-73665-1>.
- Official software: <https://github.com/fanjingfang/Epidemic/tree/v1.0.0>
- Archived commit: `9629dd7a7adecaadfffd53f2ae0f3a28a75a54eb`
- Zenodo archive: <https://doi.org/10.5281/zenodo.19841214>
- Downloaded archive SHA-256:
  `c78f746b3b4dc3566f63e6015ff9019ec8dc5e077d93bd22bfcb2a324109a253`
- Recorded source hashes:
  - `README.md`: `13f65f754a833ae0d0ae9552b8f1b5f9da114fae0763a7e358c62e3834919f75`
  - `network.cpp`: `7f0d0c0ff971b51464ed6bb173e4e4cb08ec1a1cb08c1ee8082b5f140141f194`
  - `custom_functions.py`: `23161652710dff3be54d90506464073f6f98970d076f6121db37ed5a4ca1bc14`
  - `run_boundary_plot.py`: `1ddad932ed6340bab5b5ad242d18430039b4e7020d518689509a5977898a329c`

The inspected GitHub/Zenodo v1.0.0 materials did not provide a verifiable,
explicit license text in the archived record. Therefore the reference code was
executed only in an isolated temporary copy; it is not vendored into the
production package and must not be redistributed from this repository without
separate license clarification.

The root [`LICENSE`](../LICENSE) applies to Adaptive DiffusionGuard's original
code and documentation only. It does **not** automatically relicense or cover
the COSREF reference source, article or supplementary materials, nor any
third-party datasets. Those materials remain subject to their respective
rightsholders' terms. The source/hash audit found no byte-identical copy of the
recorded COSREF v1.0.0 production source in this repository.

## Project dependency reproducibility

`uv.lock` records the complete environment resolution. The project metadata
and root license both declare Apache License 2.0 for this project's original
work.
