# GitHub release candidate checklist

Release checks were completed on 2026-10-03. No LLM call, model download or
experiment rerun was performed as part of those checks.

## Candidate status

`release_candidate_ready=true`

This means the local file set passed the checks below. It is not a legal
opinion, a production-readiness claim or evidence of real-world governance
effectiveness.

## Recommended files to submit

The proposed first commit is a source-and-documentation release only:

- Root metadata: `.env.example`, `.gitignore`, `LICENSE`, `OASIS_COMMIT`,
  `README.md`, `pyproject.toml`, `uv.lock`.
- Python package: all 39 files currently under `adaptive_diffusionguard/`.
- Frozen non-secret configurations: all 8 files currently under `configs/`.
- Documentation and release-safe aggregates: all 21 files currently under
  `docs/`, including the three SVGs and `docs/results/`.
- Offline utilities: all 12 files currently under `scripts/`.
- Tests: all 21 files currently under `tests/`.
- Training *code and configuration only*: all 9 files under `training/`.

The sanitized result JSON files contain aggregates and source SHA-256 values,
not databases, responses, prompts, rationales, profiles, logs or credentials.

## Files that must remain excluded

Do not stage local secrets or generated research/runtime artifacts:

```text
.env
runs/
*.db
*.sqlite
*.sqlite3
.cache/
.venv/
.venv*/
log/
*.log
__pycache__/
.pytest_cache/
.ruff_cache/
build/
dist/
```

The checked `.gitignore` covers these paths. Both `.venv` and the local stale
environment directory are ignored. `runs/` remains the authoritative local
record but is deliberately outside the release candidate.

The root-level residual files `testtorch.py`, `pyproject.toml.orig`, and
`.DS_Store` were identified as local leftovers and are not release candidates.
They were untracked and had no documentation references; the cleanup task
removed them without changing the tracked `pyproject.toml`.

## Current read-only Git status

The repository has an initial commit dated 2026-10-03 20:49 (+08:00):
`feat: add adaptive diffusionguard research prototype`. Immediately before
this documentation-and-cleanup task, `git status --short` showed only
`uv.lock` as modified; it was a pre-existing change and is outside this task's
allowed edit scope. There were no staged files.

## License and provenance

- Root `LICENSE` is the standard Apache License 2.0 and matches
  `pyproject.toml`'s `Apache-2.0` declaration.
- No recorded COSREF v1.0.0 production source filename or SHA-256 was found in
  project production code. Official code was run only as an isolated reference
  oracle.
- Apache-2.0 covers this project's original work only. It does not relicense
  COSREF reference source, article/supplementary material, or third-party data.
- OASIS, CAMEL, Groq, the hosted model, the article, Zenodo archive, official
  code commit and recorded hashes are identified in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Verification results

- `uv run pytest`: **195 passed**, 20 OASIS/igraph deprecation warnings.
- `uv run ruff check adaptive_diffusionguard training tests scripts`: passed
  (executed with cache disabled after the sandbox blocked a cache temp file).
- CPU-safe smoke: passed on 80 nodes, 3 timesteps; `llm_agents_used=0` and no
  remote API call.
- `uv lock --check --offline`: passed; 177 packages resolved from `uv.lock`.
- Local Markdown link check: 17 files checked, 0 missing local links.
- SVG XML validation: 3/3 valid.
- Sanitized results: both JSON files parse; all 12 referenced source hashes
  match the frozen local files.
- `.env.example`: Groq, OpenAI and Hugging Face secret fields are empty.
- Candidate secret scan (excluding `.env`, ignored runs, environments, caches
  and logs): no credential/private-key pattern match.
- Ignored-path check: `.env`, representative `runs/`, cache, virtual
  environment, log, database and SQLite paths all resolve to `.gitignore`
  rules.

## Manual checks before publishing

- Review the Apache-2.0 copyright ownership/attribution decision.
- Confirm that public release of every original document and diagram is
  intended.
- Re-read the sanitized aggregates against the local source hashes.
- Review GitHub repository visibility, branch protection and repository
  description before adding a remote.
- Inspect the exact staged diff and repeat the secret scan after staging.
- Confirm no ignored artifact was force-added.

Recommended first commit message:

```text
feat: publish Adaptive DiffusionGuard research prototype
```

## Commands for the user to run manually

These commands are documentation only and were **not** executed:

```bash
git status --short --untracked-files=all
git add -- .env.example .gitignore LICENSE OASIS_COMMIT README.md \
  adaptive_diffusionguard configs docs pyproject.toml scripts tests training uv.lock
git diff --cached --check
git status --short
git commit -m "feat: publish Adaptive DiffusionGuard research prototype"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Before committing, inspect `git diff --cached` locally; do not replace the
remote placeholder until the intended GitHub repository has been verified.
