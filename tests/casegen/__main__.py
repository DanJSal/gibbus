"""Generate a deterministic manifest of case specifications, not datasets/fits."""

import argparse
import json
from pathlib import Path

from . import (
    CaseUniverse,
    ObservationSpec,
    case_from_dict,
    coverage_counts,
    default_distributions,
    manifest,
    support,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "standard"), default="standard")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="expand every eligible pairwise sum and shared-support mixture",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-core",
        type=int,
        default=0,
        metavar="N",
        help="include N deterministic core reference scenarios",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="exclude these case IDs from the new selection",
    )
    parser.add_argument(
        "--completed-manifest",
        type=Path,
        help="balance against a manifest of previously COMPLETED runs",
    )
    parser.add_argument("--output", type=Path, default=Path("case_manifest.json"))
    args = parser.parse_args()
    universe = CaseUniverse(default_distributions(args.profile, pairwise=args.pairwise))
    fixed = ()
    if args.fixed_core:
        core = CaseUniverse(
            default_distributions("core"),
            observations=(ObservationSpec("points"),),
            degrees=(2,),
            tails=("none",),
            sample_sizes=(500,),
            component_counts=(1,),
        )
        fixed = tuple(core.select(args.fixed_core, seed=0, explore_fraction=0))
    excluded = []
    if args.exclude_manifest:
        old = json.loads(args.exclude_manifest.read_text())
        excluded = [x["case_id"] for x in old["cases"]]
    prior = {}
    if args.completed_manifest:
        old = json.loads(args.completed_manifest.read_text())
        completed = [case_from_dict(x["spec"]) for x in old["cases"]]
        prior = coverage_counts(completed)
        # A completed run should not silently be selected again.
        excluded.extend(x["case_id"] for x in old["cases"])
    selected = universe.select(
        args.count,
        seed=args.seed,
        fixed=fixed,
        coverage_counts=prior,
        exclude_case_ids=excluded,
    )
    doc = manifest(selected, args.seed)
    args.output.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(selected)} specifications to {args.output}")
    print(
        "Fit categories:",
        {
            name: sum(
                s.fit.fit_category(support(s.distribution)) == name for s in selected
            )
            for name in ("S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09")
        },
    )
    print("No datasets generated; no fits executed.")


if __name__ == "__main__":
    main()
