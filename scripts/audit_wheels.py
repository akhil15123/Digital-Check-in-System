"""Validate the integrity and platform tags of bundled wheel files."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WHEEL_DIR = ROOT / "Dlib-Precompiled-Wheels-for-Python-on-Windows-x64-Easy-Installation-master"
NAME_PATTERN = re.compile(r"^dlib-[^-]+-(cp\d+)-[^-]+-(win_amd64)\.whl$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as wheel_file:
        for block in iter(lambda: wheel_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(path: Path) -> None:
    match = NAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected wheel filename: {path.name}")
    python_tag, platform_tag = match.groups()

    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise ValueError(f"Corrupt member in {path.name}: {corrupt_member}")
        wheel_metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/WHEEL"))
        metadata = archive.read(wheel_metadata).decode("utf-8")

    expected_tag = f"Tag: {python_tag}-"
    if expected_tag not in metadata or platform_tag not in metadata:
        raise ValueError(f"Internal tag does not match filename: {path.name}")
    print(f"ok  {path.name}  sha256:{sha256(path)}")


def main() -> None:
    wheels = sorted(WHEEL_DIR.glob("*.whl"))
    if len(wheels) != 6:
        raise SystemExit(f"Expected 6 wheel files, found {len(wheels)}")
    for wheel in wheels:
        audit(wheel)


if __name__ == "__main__":
    main()
