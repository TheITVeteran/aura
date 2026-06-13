import os
import glob
from pathlib import Path

def compile_codebase():
    root_dir = Path("/Users/bryan/.aura/live-source")
    output_path = root_dir / "aura_full_codebase_audit.txt"
    desktop_path = Path("/Users/bryan/Desktop/aura_full_codebase_audit.txt")
    
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
                if not any(x in parts for x in [".venv", "venv", "env", "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".pyre", "build", "dist"]):
                    all_files.append(f_str)
            
    # Scan root files
    for f in root_files:
        fp = root_dir / f
        if fp.exists():
            all_files.append(str(fp))
            
    # Sort for deterministic output
    all_files = sorted(list(set(all_files)))
    
    print(f"Found {len(all_files)} files to compile.")
    
    with open(output_path, "w", encoding="utf-8") as outfile:
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
                with open(fp, "r", encoding="utf-8", errors="replace") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"# ERROR READING FILE: {e}\n")
                
    # Copy to Desktop
    import shutil
    try:
        shutil.copy(output_path, desktop_path)
        print(f"Successfully compiled entire codebase to {output_path} and copied to {desktop_path}")
    except Exception as e:
        print(f"Error copying to Desktop: {e}")

if __name__ == "__main__":
    compile_codebase()
