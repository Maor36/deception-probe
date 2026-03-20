"""
Colab Setup Script — Run this ONCE at the start of every Colab session.
========================================================================

Usage (paste this in the first cell of your Colab notebook):

    # ── Cell 1: Setup ──
    !pip install -q bitsandbytes>=0.46.1 scikit-learn scipy accelerate
    from google.colab import drive
    drive.mount('/content/drive')

    import os
    os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"

    # Clone or pull the repo
    if not os.path.exists("/content/deception-probe"):
        !git clone https://YOUR_GH_TOKEN@github.com/cohen-liel/deception-probe.git /content/deception-probe
    else:
        !cd /content/deception-probe && git pull

    # Run setup verification
    !cd /content/deception-probe && python colab_setup.py

    # ── Cell 2: Run experiment ──
    !cd /content/deception-probe && python experiments/04_cross_model_transfer/run.py

========================================================================
"""

import os
import sys
import importlib

def check_package(name, pip_name=None):
    """Check if a package is installed."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("  Deception Probe — Colab Environment Check")
    print("=" * 60)

    errors = []
    warnings = []

    # ── 1. Check Python packages ──────────────────────────────────
    print("\n[1/6] Checking Python packages...")
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "bitsandbytes": "bitsandbytes>=0.46.1",
        "accelerate": "accelerate",
        "numpy": "numpy",
    }
    for module, pip_name in packages.items():
        if check_package(module):
            print(f"  ✅ {module}")
        else:
            errors.append(f"Missing: pip install {pip_name}")
            print(f"  ❌ {module} — run: pip install {pip_name}")

    # ── 2. Check GPU ──────────────────────────────────────────────
    print("\n[2/6] Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  ✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            if gpu_mem < 14:
                warnings.append(f"GPU has only {gpu_mem:.1f}GB — may not fit 8B models. Use A100 or L4.")
                print(f"  ⚠️  Only {gpu_mem:.1f}GB — recommend A100 or L4 for 8B models")
        else:
            errors.append("No GPU detected! Go to Runtime → Change runtime type → GPU")
            print("  ❌ No GPU! Go to Runtime → Change runtime type → GPU")
    except Exception as e:
        errors.append(f"GPU check failed: {e}")
        print(f"  ❌ GPU check failed: {e}")

    # ── 3. Check HuggingFace token ────────────────────────────────
    print("\n[3/6] Checking HuggingFace token...")
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token and hf_token.startswith("hf_") and len(hf_token) > 10:
        print(f"  ✅ HF_TOKEN set ({hf_token[:8]}...)")
        # Verify it works
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.model_info("meta-llama/Llama-3.1-8B-Instruct")
            print("  ✅ Token has access to Llama-3.1-8B-Instruct")
        except Exception as e:
            err_str = str(e)
            if "403" in err_str or "gated" in err_str.lower():
                errors.append("HF token doesn't have Llama access. Accept license at: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
                print("  ❌ Token doesn't have Llama access!")
                print("     → Accept license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
            elif "401" in err_str:
                errors.append("HF token is invalid or expired")
                print("  ❌ Token is invalid or expired")
            else:
                warnings.append(f"Could not verify Llama access: {e}")
                print(f"  ⚠️  Could not verify: {e}")
    else:
        errors.append("HF_TOKEN not set! Add: os.environ['HF_TOKEN'] = 'hf_...'")
        print("  ❌ HF_TOKEN not set!")
        print("     → Add before running: os.environ['HF_TOKEN'] = 'hf_YOUR_TOKEN'")

    # ── 4. Check Google Drive ─────────────────────────────────────
    print("\n[4/6] Checking Google Drive...")
    drive_path = "/content/drive/MyDrive/deception-probe-results"
    if os.path.exists(drive_path):
        files = os.listdir(drive_path)
        print(f"  ✅ Drive mounted, {len(files)} files in results folder")
        # Check for key files
        expected = ["exp02c_token_hs", "exp02c_sentence_hs.npz",
                    "exp02c_responses.json", "exp02c_token_labels.json"]
        for f in expected:
            full = os.path.join(drive_path, f)
            if os.path.exists(full):
                print(f"     ✅ {f}")
            else:
                warnings.append(f"Missing from Drive: {f}")
                print(f"     ⚠️  Missing: {f}")
    elif os.path.exists("/content/drive"):
        warnings.append(f"Drive mounted but results folder not found at {drive_path}")
        print(f"  ⚠️  Drive mounted but results folder not found")
        print(f"     Expected: {drive_path}")
    else:
        warnings.append("Google Drive not mounted")
        print("  ⚠️  Drive not mounted (needed for vectors)")
        print("     → Run: from google.colab import drive; drive.mount('/content/drive')")

    # ── 5. Check repo ─────────────────────────────────────────────
    print("\n[5/6] Checking repo...")
    repo_path = "/content/deception-probe"
    if os.path.exists(repo_path):
        # Check git status
        import subprocess
        result = subprocess.run(
            ["git", "-C", repo_path, "log", "--oneline", "-1"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✅ Repo exists, latest commit: {result.stdout.strip()}")
        else:
            print(f"  ✅ Repo exists")

        # Check symlinks for exp02c/d
        hs_link = os.path.join(repo_path, "results", "exp02c_token_hs")
        if os.path.exists(hs_link):
            print(f"  ✅ Token HS symlink exists")
        else:
            warnings.append("Token HS not linked. Run: ln -s /content/drive/.../exp02c_token_hs results/")
            print(f"  ⚠️  Token HS not linked (needed for step2d)")
    else:
        errors.append("Repo not cloned!")
        print("  ❌ Repo not found at /content/deception-probe")

    # ── 6. Check git push capability ──────────────────────────────
    print("\n[6/6] Checking git push capability...")
    if os.path.exists(repo_path):
        import subprocess
        result = subprocess.run(
            ["git", "-C", repo_path, "remote", "get-url", "origin"],
            capture_output=True, text=True
        )
        url = result.stdout.strip()
        if "@github.com" in url:
            print(f"  ✅ Git remote has auth token")
        else:
            warnings.append("Git remote has no auth token — can't push results")
            print(f"  ⚠️  Git remote has no auth token — can't push")
            print(f"     → Run: git remote set-url origin https://USER:TOKEN@github.com/cohen-liel/deception-probe.git")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if errors:
        print(f"  ❌ {len(errors)} ERROR(S) — must fix before running:")
        for e in errors:
            print(f"     • {e}")
    if warnings:
        print(f"  ⚠️  {len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"     • {w}")
    if not errors and not warnings:
        print("  ✅ ALL CHECKS PASSED — Ready to run experiments!")
    elif not errors:
        print("  ✅ No errors — experiments should run (check warnings)")
    print("=" * 60)

    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️  Fix errors above before running experiments.")
        sys.exit(1)
    else:
        print("\n🚀 Ready! Run experiments with:")
        print("   !cd /content/deception-probe && python experiments/04_cross_model_transfer/run.py")
