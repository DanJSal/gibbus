"""Release gate: validate metadata; a tag also requires a successful main CI run."""

import json
import os
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    if f'version: "{version}"' not in Path("CITATION.cff").read_text():
        raise SystemExit("CITATION.cff version disagrees with pyproject.toml")
    if not any(
        re.fullmatch(r"##\s+" + re.escape(version) + r"\s+(?:-|—)\s+.+", line)
        for line in Path("CHANGELOG.md").read_text().splitlines()
    ):
        raise SystemExit("CHANGELOG.md has no matching release heading")
    event = os.environ["GITHUB_EVENT_NAME"]
    if event == "push":
        if os.environ["GITHUB_REF_NAME"] != f"v{version}":
            raise SystemExit("Tag must exactly match v<project.version>")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"], check=True
        )
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        repository = os.environ["GITHUB_REPOSITORY"]
        query = urlencode(
            {"head_sha": sha, "event": "push", "branch": "main", "per_page": 100}
        )
        request = Request(
            f"https://api.github.com/repos/{repository}/actions/workflows/ci.yml/runs?{query}",
            headers={
                "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urlopen(request, timeout=30) as response:
            runs = json.load(response)["workflow_runs"]
        runs = [
            r
            for r in runs
            if r["head_sha"] == sha
            and r["head_branch"] == "main"
            and r["event"] == "push"
        ]
        if not runs:
            raise SystemExit("No main push CI run exists for the tagged commit")
        latest = max(runs, key=lambda r: (r["run_number"], r.get("run_attempt", 1)))
        if latest["status"] != "completed" or latest["conclusion"] != "success":
            raise SystemExit(
                "Latest main push CI run for this commit did not complete successfully"
            )
    elif event != "workflow_dispatch":
        raise SystemExit(f"Unsupported release event: {event}")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"version={version}\n")
    print(f"Validated metadata for {version}; expected tag v{version}")


if __name__ == "__main__":
    main()
