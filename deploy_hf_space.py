"""
deploy_hf_space.py
──────────────────
Pushes the backend (Stage 1/ + retrieval_layer/) to a Hugging Face Space
(Docker SDK), without touching the GitHub repo's own README.md/Dockerfile
naming or dragging in frontend/, Healthcare Doc.zip, or KT_Session_Document.docx.

File list is derived from `git ls-files "Stage 1" retrieval_layer` — the
same set already vetted safe for the public GitHub repo (secrets, venv/,
Stage 1/data/, and eval report logs are all gitignored, so none of that
reaches the Space either).

DEMO DATA: by default (unless --no-demo-data is passed), this also uploads
Stage 1/data/pdfs/ (the real source PDFs + pre-converted markdown) and the
Knowledge-Explorer-visible subset of Stage 1/data/output/ (matching
api_server.py's _TOP_LEVEL_ALLOWLIST — not the full working output/, which
has backups/logs/caches). These are gitignored on GitHub on purpose (the
PDFs are licensed product documentation), but included here so the deployed
Space demos Process/Knowledge Explorer with real data. The Space is public,
so treat this as an accepted, deliberate exposure, not a secret.

Because HF Spaces' free-tier container storage is ephemeral, this demo data
is baked into the image — running /api/reset + /api/process live on the
Space does NOT change what's in this repo, so a future redeploy (even for
an unrelated code change) will revert the running corpus back to this demo
data. Pass --no-demo-data on any redeploy meant to preserve whatever a
client has since processed live on the Space.

Usage:
    pip install -U huggingface_hub
    hf auth login                      # paste a token with write access
    python deploy_hf_space.py <username>/<space-name> [--no-demo-data]

Re-run any time backend code changes — same command, one new commit.
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ROOT = Path(__file__).parent
DEMO_PDF_DIR = REPO_ROOT / "Stage 1" / "data" / "pdfs"
DEMO_OUTPUT_DIR = REPO_ROOT / "Stage 1" / "data" / "output"

# Matches api_server.py's _TOP_LEVEL_ALLOWLIST — only what Knowledge Explorer
# actually surfaces, not Stage 1's full working output/ (eval logs, caches,
# .bak files, prompts, etc.)
_OUTPUT_ALLOWLIST = [
    "hierarchy_summaries", "topic_summaries", "delta_reports",
    "version_delta_report.md", "version_evolution_report.md",
    "enterprise_nested_topics.json", "parent_relationship_clusters.json",
    "eval_report.md",
]


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "Stage 1", "retrieval_layer"],
        cwd=REPO_ROOT, capture_output=True, check=True, text=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def demo_data_files() -> list[tuple[Path, str]]:
    """(local_path, path_in_repo) pairs for the demo PDFs + allowlisted output."""
    pairs: list[tuple[Path, str]] = []
    if DEMO_PDF_DIR.exists():
        for p in sorted(DEMO_PDF_DIR.glob("*")):
            if p.is_file() and (p.suffix == ".pdf" or p.name.endswith("_Converted.md")):
                pairs.append((p, f"Stage 1/data/pdfs/{p.name}"))
    for name in _OUTPUT_ALLOWLIST:
        src = DEMO_OUTPUT_DIR / name
        if not src.exists():
            continue
        files = [src] if src.is_file() else [f for f in src.rglob("*") if f.is_file()]
        for f in files:
            rel = f.relative_to(DEMO_OUTPUT_DIR)
            pairs.append((f, f"Stage 1/data/output/{rel}"))
    return pairs


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    include_demo_data = "--no-demo-data" not in sys.argv
    if len(args) != 1:
        sys.exit("Usage: python deploy_hf_space.py <username>/<space-name> [--no-demo-data]")
    space_id = args[0]

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    ops = [
        CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(REPO_ROOT / p))
        for p in tracked_files()
    ]
    ops.append(CommitOperationAdd(path_in_repo="Dockerfile", path_or_fileobj=str(REPO_ROOT / "Dockerfile.hf")))
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(REPO_ROOT / "README_hf.md")))

    if include_demo_data:
        for local_path, path_in_repo in demo_data_files():
            ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local_path)))

    print(f"Uploading {len(ops)} files to {space_id} ({'with' if include_demo_data else 'without'} demo data) ...")
    api.create_commit(repo_id=space_id, repo_type="space", operations=ops, commit_message="Deploy backend")
    print(f"Done: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
