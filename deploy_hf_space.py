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

Usage:
    pip install -U huggingface_hub
    hf auth login                      # paste a token with write access
    python deploy_hf_space.py <username>/<space-name>

Re-run any time backend code changes — same command, one new commit.
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ROOT = Path(__file__).parent


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "Stage 1", "retrieval_layer"],
        cwd=REPO_ROOT, capture_output=True, check=True, text=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python deploy_hf_space.py <username>/<space-name>")
    space_id = sys.argv[1]

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    ops = [
        CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(REPO_ROOT / p))
        for p in tracked_files()
    ]
    ops.append(CommitOperationAdd(path_in_repo="Dockerfile", path_or_fileobj=str(REPO_ROOT / "Dockerfile.hf")))
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(REPO_ROOT / "README_hf.md")))

    print(f"Uploading {len(ops)} files to {space_id} ...")
    api.create_commit(repo_id=space_id, repo_type="space", operations=ops, commit_message="Deploy backend")
    print(f"Done: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
