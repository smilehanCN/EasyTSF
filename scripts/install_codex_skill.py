#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_SKILL_NAME = "migrate-model-to-easytsf"
REPO_ROOT = Path(__file__).resolve().parents[1]


def default_target_dir() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "skills"
    return Path.home() / ".codex" / "skills"


def validate_skill_dir(skill_dir: Path) -> None:
    required_paths = [
        skill_dir / "SKILL.md",
        skill_dir / "agents" / "openai.yaml",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("skill is incomplete: missing {}".format(", ".join(missing)))


def install_skill(skill_name: str, target_dir: Path, force: bool) -> Path:
    source_dir = REPO_ROOT / "skills" / skill_name
    validate_skill_dir(source_dir)

    target_dir = target_dir.expanduser().resolve()
    destination_dir = target_dir / skill_name

    if destination_dir == source_dir.resolve():
        raise ValueError("target directory resolves to the repository source directory")

    target_dir.mkdir(parents=True, exist_ok=True)

    if destination_dir.exists():
        if not force:
            raise FileExistsError("{} already exists; rerun with --force".format(destination_dir))
        if destination_dir.is_dir():
            shutil.rmtree(destination_dir)
        else:
            destination_dir.unlink()

    shutil.copytree(source_dir, destination_dir)
    return destination_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install a repository Codex skill into the local skills directory.")
    parser.add_argument(
        "--name",
        default=DEFAULT_SKILL_NAME,
        help="Skill directory name under repo/skills/.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=default_target_dir(),
        help="Target skills root directory. Defaults to ${CODEX_HOME}/skills or ~/.codex/skills.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite an existing installed copy.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    destination_dir = install_skill(args.name, args.target_dir, args.force)
    print("Installed {} to {}".format(args.name, destination_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
