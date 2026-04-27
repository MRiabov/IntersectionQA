"""Package and optionally upload an experiment run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intersectionqa.experiments import RunArtifactManager, create_run_tarball


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tarball", type=Path)
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-repo-type", default="dataset")
    parser.add_argument("--hf-path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manager = RunArtifactManager.create(args.run_dir, resume=True)
    tarball = create_run_tarball(args.run_dir, args.tarball)
    upload_record = {
        "tarball": tarball,
        "hf_repo_id": args.hf_repo_id,
        "hf_repo_type": args.hf_repo_type,
        "hf_path": args.hf_path,
        "status": "local_only",
        "deferred_command": _deferred_command(args.run_dir, args.tarball, args.hf_repo_id, args.hf_path),
    }
    if args.hf_repo_id and not args.dry_run:
        try:
            from huggingface_hub import upload_file

            path_in_repo = args.hf_path or Path(tarball["path"]).name
            url = upload_file(
                path_or_fileobj=tarball["path"],
                path_in_repo=path_in_repo,
                repo_id=args.hf_repo_id,
                repo_type=args.hf_repo_type,
            )
            upload_record.update({"status": "uploaded", "url": url, "path_in_repo": path_in_repo})
        except Exception as exc:
            upload_record.update({"status": "failed", "error": str(exc)})
    elif args.hf_repo_id and args.dry_run:
        upload_record["status"] = "deferred"

    manager.add_artifact(kind="tarball", path=tarball["path"], role="run_archive", metadata=tarball)
    manager.add_artifact(kind="upload", path=tarball["path"], role="hf_xet", metadata=upload_record, checksum=False)
    print(json.dumps(upload_record, indent=2, sort_keys=True))
    if upload_record["status"] == "failed":
        raise SystemExit(1)


def _deferred_command(run_dir: Path, tarball: Path | None, repo_id: str | None, hf_path: str | None) -> str | None:
    if not repo_id:
        return None
    pieces = ["rtk", "uv", "run", "python", "-m", "scripts.publishing.upload_run_artifacts", str(run_dir), "--hf-repo-id", repo_id]
    if tarball:
        pieces.extend(["--tarball", str(tarball)])
    if hf_path:
        pieces.extend(["--hf-path", hf_path])
    return " ".join(pieces)


if __name__ == "__main__":
    main()

