"""Report pytest failures in a JUnit file as GitHub Actions annotations.

Usage: python annotate_junit.py JUNIT_XML

One annotation lists every failure with the platform and dependency versions
(a step shows at most ten annotations); the first failures also get their
own annotation with the condensed call chain.
"""

import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy
import scipy

_DETAILED = 8
_DETAIL_CHARS = 6000


def _escape(text):
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def _property(text):
    return _escape(text).replace(":", "%3A").replace(",", "%2C")


def _condense(text):
    # Frame locations, the marked source lines and pytest's error lines.
    keep = [
        line
        for line in text.splitlines()
        if line.startswith(("E ", ">")) or re.search(r"\.py:\d+(: in |:\s)", line)
    ]
    return "\n".join(keep)[:_DETAIL_CHARS]


def main(path):
    failures = []
    if path.exists():
        for case in ET.parse(path).iter("testcase"):
            for kind in ("failure", "error"):
                node = case.find(kind)
                if node is not None:
                    name = f"{case.get('classname')}::{case.get('name')}"
                    detail = (node.get("message") or "") + "\n" + (node.text or "")
                    failures.append((kind, name, detail.strip()))
    if not failures:
        print(f"::error title=No test results::{path} has no failures to report")
        return
    versions = (
        f"{platform.platform()} {platform.machine()} Python "
        f"{platform.python_version()} NumPy {numpy.__version__} "
        f"SciPy {scipy.__version__}"
    )
    listing = "\n".join(f"{kind}: {name}" for kind, name, _ in failures)
    print(
        f"::error title={len(failures)} failing tests::"
        f"{_escape(versions + chr(10) + listing)}"
    )
    for _, name, detail in failures[:_DETAILED]:
        print(f"::error title={_property(name)}::{_escape(_condense(detail))}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
