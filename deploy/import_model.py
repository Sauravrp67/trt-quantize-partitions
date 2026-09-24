#!/usr/bin/env python3
"""Copy an ONNX model, or download one from a supplied URL with SHA-256 verification."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="local ONNX file or HTTPS URL")
    parser.add_argument("--out", required=True, type=Path, help="destination .onnx file")
    parser.add_argument("--sha256", help="expected SHA-256; required for downloads")
    args = parser.parse_args()
    remote = urllib.parse.urlparse(args.source).scheme == "https"
    if "://" in args.source and not remote:
        parser.error("downloads require HTTPS")
    if remote and not args.sha256:
        parser.error("--sha256 is required for downloads")
    if args.sha256 and (len(args.sha256) != 64 or any(c not in '0123456789abcdef' for c in args.sha256.lower())):
        parser.error("--sha256 must contain 64 hexadecimal characters")
    if args.out.suffix != ".onnx":
        parser.error("--out must end in .onnx")
    if args.out.exists():
        parser.error(f"destination exists: {args.out}; choose a new path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.out.parent, delete=False) as output:
            temp_path = Path(output.name)
            if remote:
                source = urllib.request.urlopen(args.source, timeout=60)
            else:
                source = open(args.source, "rb")
            with source:
                shutil.copyfileobj(source, output)
        digest = hashlib.sha256()
        with temp_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if not temp_path.stat().st_size:
            parser.error("source is empty")
        if args.sha256 and digest.hexdigest() != args.sha256.lower():
            parser.error("SHA-256 mismatch; destination was not written")
        temp_path.replace(args.out)
        print(f"{args.out}: SHA-256 {digest.hexdigest()}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
