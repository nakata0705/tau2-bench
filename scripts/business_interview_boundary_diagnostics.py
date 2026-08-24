"""Run the diagnostic-only structural START/END boundary experiment.

This command reads saved artifacts but never rewrites them.  It writes two
separate derived JSON files: a boundary/topology audit and the A/B/C/D
current-vs-normalized alignment comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tau2.domains.business_interview.boundary_diagnostics import (  # pyright: ignore[reportMissingImports]
    write_deliverables,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/business_interview_real_llm"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/business_interview_real_llm/boundary_audit.json"),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("artifacts/business_interview_real_llm/boundary_comparison.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[9002, 9003, 9004])
    args = parser.parse_args()

    audit, comparison = write_deliverables(
        args.artifact_dir,
        args.audit_output,
        args.comparison_output,
        seeds=args.seeds,
    )
    print(
        f"wrote boundary audit: {args.audit_output} "
        f"({len(audit['saved_real_llm'])} saved seeds)"
    )
    print(
        f"wrote boundary comparison: {args.comparison_output} "
        f"({len(comparison['saved_real_llm'])} saved seeds)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
