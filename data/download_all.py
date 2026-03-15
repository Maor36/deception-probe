"""Download all datasets for DeceptionProbe experiments."""
import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

try:
    from datasets import load_dataset
except ImportError:
    install("datasets")
    from datasets import load_dataset

print("=" * 50)
print("Downloading all DeceptionProbe datasets...")
print("=" * 50)

# 1. meg-tong sycophancy-eval
print("\n[1/4] meg-tong/sycophancy-eval...")
os.makedirs("sycophancy_eval", exist_ok=True)
for split in ["answer", "are_you_sure", "feedback", "mimicry"]:
    ds = load_dataset("meg-tong/sycophancy-eval", data_files=f"{split}.jsonl", split="train")
    ds.to_json(f"sycophancy_eval/{split}.jsonl")
    print(f"  {split}: {len(ds)} rows")

# 2. Anthropic sycophancy
print("\n[2/4] Anthropic/model-written-evals (sycophancy)...")
os.makedirs("anthropic_sycophancy", exist_ok=True)
for name in ["sycophancy_on_nlp_survey", "sycophancy_on_philpapers2020", "sycophancy_on_political_typology_quiz"]:
    ds = load_dataset("Anthropic/model-written-evals", data_files=f"sycophancy/{name}.jsonl", split="train")
    ds.to_json(f"anthropic_sycophancy/{name}.jsonl")
    print(f"  {name}: {len(ds)} rows")

# 3. TruthfulQA
print("\n[3/4] TruthfulQA...")
os.makedirs("truthfulqa", exist_ok=True)
for subset in ["generation", "multiple_choice"]:
    ds = load_dataset("truthfulqa/truthful_qa", subset, split="validation")
    ds.to_json(f"truthfulqa/truthfulqa_{subset}.jsonl")
    print(f"  {subset}: {len(ds)} rows")

# 4. HaluEval
print("\n[4/4] HaluEval...")
os.makedirs("halueval", exist_ok=True)
for subset in ["qa_samples", "summarization_samples", "dialogue_samples", "general"]:
    ds = load_dataset("pminervini/HaluEval", subset, split="data")
    ds.to_json(f"halueval/halueval_{subset.replace('_samples','')}.jsonl")
    print(f"  {subset}: {len(ds)} rows")

print("\n" + "=" * 50)
print("All datasets downloaded!")
print("=" * 50)
