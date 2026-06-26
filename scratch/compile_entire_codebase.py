from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path


DEFAULT_EXCLUDES = {
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".pyre",
    "build",
    "dist",
}


def compile_codebase(
    *,
    root_dir: Path | None = None,
    output_path: Path | None = None,
    desktop_copy: Path | None = None,
) -> Path:
    root_dir = (root_dir or Path(__file__).resolve().parents[1]).expanduser().resolve()
    output_path = output_path or (root_dir / "aura_full_codebase_audit.txt")
    desktop_copy = desktop_copy.expanduser().resolve() if desktop_copy is not None else None

    # Target directories to scan recursively for .py files
    target_dirs = [
        "core",
        "skills",
        "research",
        "proof_kernel",
        "tests",
        "utils",
        "llm",
        "interface",
        "autonomy_engine",
        "security",
        "senses",
        "infrastructure",
        "native",
        "optimizer",
        "scripts",
        "tools",
        "memory",
        "training",
        "rust_extensions",
    ]

    # Standalone python files in root
    root_files = [
        "aura_main.py",
        "main_daemon.py",
        "system_health.py",
    ]

    all_files = []

    # Scan directories
    for d in target_dirs:
        dir_path = root_dir / d
        if dir_path.exists() and dir_path.is_dir():
            found = glob.glob(str(dir_path / "**" / "*.py"), recursive=True)
            # Filter out any virtualenv, pycache, or editor/IDE configs
            for f_str in found:
                p = Path(f_str)
                parts = p.parts
                if not any(x in parts for x in DEFAULT_EXCLUDES):
                    all_files.append(f_str)

    # Scan root files
    for f in root_files:
        fp = root_dir / f
        if fp.exists():
            all_files.append(str(fp))

    # Sort for deterministic output
    all_files = sorted(list(set(all_files)))

    print(f"Found {len(all_files)} files to compile.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        outfile.write("# AURA ENTIRE CODEBASE AUDIT BUNDLE\n")
        outfile.write(f"# Root: {root_dir}\n")
        outfile.write("# Contains all Python modules, tests, research files, and skills.\n\n")

        for fp_str in all_files:
            fp = Path(fp_str)
            rel_path = fp.relative_to(root_dir)
            outfile.write("\n\n" + "=" * 80 + "\n")
            outfile.write(f"FILE: {rel_path}\n")
            outfile.write("=" * 80 + "\n\n")

            try:
                outfile.write(fp.read_text(encoding="utf-8", errors="replace"))
            except OSError as exc:
                outfile.write(f"# ERROR READING FILE: {type(exc).__name__}: {exc}\n")

    if desktop_copy is not None:
        try:
            desktop_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, desktop_copy)
            print(f"Successfully compiled codebase to {output_path} and copied to {desktop_copy}")
        except OSError as exc:
            print(f"Error copying audit bundle: {type(exc).__name__}: {exc}")
    else:
        print(f"Successfully compiled codebase to {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Aura Python source into a single audit text file.")
    parser.add_argument("--root", type=Path, default=None, help="Repository root; defaults to this script's repo.")
    parser.add_argument("--out", type=Path, default=None, help="Output text path.")
    parser.add_argument("--desktop-copy", type=Path, default=None, help="Optional copy target, e.g. ~/Desktop/audit.txt.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    compile_codebase(root_dir=args.root, output_path=args.out, desktop_copy=args.desktop_copy)
