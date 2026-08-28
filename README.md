# dlib Wheels for Windows x64

[![Wheel audit](https://github.com/akhil15123/Digital-Check-in-System/actions/workflows/ci.yml/badge.svg)](https://github.com/akhil15123/Digital-Check-in-System/actions/workflows/ci.yml)

A compatibility archive of precompiled dlib wheels for 64-bit Windows and CPython 3.7–3.12. It is intended for older face-recognition prototypes that cannot build dlib locally.

## Available wheels

| Python | Wheel |
| --- | --- |
| 3.7 | `dlib-19.19.0-cp37-cp37m-win_amd64.whl` |
| 3.8 | `dlib-19.19.0-cp38-cp38-win_amd64.whl` |
| 3.9 | `dlib-19.22.1-cp39-cp39-win_amd64.whl` |
| 3.10 | `dlib-19.22.99-cp310-cp310-win_amd64.whl` |
| 3.11 | `dlib-19.24.1-cp311-cp311-win_amd64.whl` |
| 3.12 | `dlib-19.24.99-cp312-cp312-win_amd64.whl` |

All wheels target `win_amd64`; they do not support macOS, Linux, ARM Windows, PyPy, or a different CPython version.

## Install

Download the wheel matching `python --version`, then run:

```powershell
python -m pip install --upgrade pip
python -m pip install path\to\dlib-19.24.1-cp311-cp311-win_amd64.whl
python -m pip install face-recognition
```

Installing CMake is not required when a compatible binary wheel installs successfully.

## Audit the archive

```bash
python scripts/audit_wheels.py
```

The audit verifies each wheel as a ZIP archive, confirms its internal platform tag, and prints a SHA-256 digest that can be recorded before distribution.

## Security notice

Binary wheels execute native code. These historical artifacts are provided for reproducibility and are not official dlib releases. Prefer a current package from a trusted index or build from the official source when possible. Verify the reported digest before sharing a wheel and test it in an isolated environment.

The nested ZIP is retained as a legacy download bundle; the individual wheels are the canonical copies in this repository.
