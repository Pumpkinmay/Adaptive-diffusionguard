"""Summarize an isolated fixed-seed run of the official v1.0.0 simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

OFFICIAL_COMMIT = "9629dd7a7adecaadfffd53f2ae0f3a28a75a54eb"
ZENODO_DOI = "10.5281/zenodo.19841214"
ARTICLE_DOI = "10.1038/s41467-026-73665-1"
SOURCE_HASHES = {
    "README.md": "13f65f754a833ae0d0ae9552b8f1b5f9da114fae0763a7e358c62e3834919f75",
    "network.cpp": "7f0d0c0ff971b51464ed6bb173e4e4cb08ec1a1cb08c1ee8082b5f140141f194",
    "custom_functions.py": "23161652710dff3be54d90506464073f6f98970d076f6121db37ed5a4ca1bc14",
    "run_boundary_plot.py": "1ddad932ed6340bab5b5ad242d18430039b4e7020d518689509a5977898a329c",
    "zenodo_archive": "c78f746b3b4dc3566f63e6015ff9019ec8dc5e077d93bd22bfcb2a324109a253",
}
PAPER_REFERENCE = {0.2: (0.0, 0.85), 0.8: (0.78, 0.10)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paper_cost(omega_intra: float, omega_inter: float) -> float:
    return (
        math.exp(-(omega_intra + omega_inter)) - math.exp(-2.0)
    ) / (1.0 - math.exp(-2.0))


def summarize_reference_run(
    raw_samples: Path,
    output: Path,
    instrumentation_patch: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reference output: {output}")
    output.mkdir(parents=True)
    grouped: dict[tuple[float, float, float], list[float]] = defaultdict(list)
    with raw_samples.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (
                float(row["mu"]),
                float(row["omega_intra"]),
                float(row["omega_inter"]),
            )
            grouped[key].append((float(row["rho_A"]) + float(row["rho_B"])) / 2)
    if not grouped:
        raise ValueError("reference raw sample file is empty")
    means = [
        {
            "mu": key[0],
            "omega_intra": key[1],
            "omega_inter": key[2],
            "mean_rho": sum(values) / len(values),
            "replicates": len(values),
            "paper_cost": _paper_cost(key[1], key[2]),
        }
        for key, values in sorted(grouped.items())
    ]
    optima = []
    for mu in sorted({row["mu"] for row in means}):
        controllable = [
            row for row in means if row["mu"] == mu and row["mean_rho"] <= 0.3
        ]
        if not controllable:
            optima.append({"mu": mu, "available": False})
            continue
        optimum = min(
            controllable,
            key=lambda row: (
                row["paper_cost"],
                row["omega_intra"],
                row["omega_inter"],
            ),
        )
        direction = "intra" if mu < 0.5 else "inter"
        direction_observed = (
            optimum["omega_intra"] < optimum["omega_inter"]
            if direction == "intra"
            else optimum["omega_inter"] < optimum["omega_intra"]
        )
        optima.append(
            {
                **optimum,
                "available": True,
                "paper_priority": direction,
                "priority_observed": direction_observed,
                "supplementary_figure_16_reference": PAPER_REFERENCE.get(mu),
            }
        )
    summary: dict[str, object] = {
        "status": "completed",
        "oracle": "official_v1.0.0_cpp_monte_carlo_with_isolated_parameter_patch",
        "official_commit": OFFICIAL_COMMIT,
        "article_doi": ARTICLE_DOI,
        "zenodo_doi": ZENODO_DOI,
        "parameters": {
            "nodes": 2000,
            "average_degree": 20,
            "initial_density": 0.17,
            "threshold": 0.1,
            "seed": 20260929,
            "replicates": 8,
            "mu_values": [0.2, 0.8],
            "omega_step": 0.1,
            "non_diffusion_cutoff": 0.3,
        },
        "source_hashes_sha256": SOURCE_HASHES,
        "raw_samples_sha256": _sha256(raw_samples),
        "grid_points": len(grouped),
        "raw_sample_count": sum(len(values) for values in grouped.values()),
        "optima": optima,
        "limitations": [
            "N=2000 and eight replicates are far below the paper's main N=200000 setup.",
            "The omega grid has step 0.1, so it cannot reproduce continuous tangency points.",
            "The official time seed was replaced by a fixed seed only in an isolated copy.",
            "This checks directional behavior at two mu values, not the full phase diagram.",
        ],
    }
    shutil.copyfile(raw_samples, output / "raw_samples.tsv")
    shutil.copyfile(instrumentation_patch, output / "instrumentation.patch")
    (output / "grid_means.json").write_text(
        json.dumps(means, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run_log.txt").write_text(
        "compile: mpic++ -m64 network.cpp mt19937ar.cpp -o network "
        "-Wall -DMPIPRIMES\n"
        "run: mpirun -np 1 ./network\n"
        f"raw_sample_count: {summary['raw_sample_count']}\n"
        f"raw_samples_sha256: {summary['raw_samples_sha256']}\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-samples", type=Path, required=True)
    parser.add_argument("--instrumentation-patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_reference_run(
        args.raw_samples, args.output, args.instrumentation_patch
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
