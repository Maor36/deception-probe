"""
Shared utilities for DeceptionProbe experiments.

Provides:
    - Model loading with quantization
    - Dataset parsing (meg-tong/sycophancy-eval)
    - Hidden state extraction from generated tokens
    - Answer matching heuristics
    - Probe training and evaluation with statistical controls
    - Logging and result serialization
"""

import os
import json
import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    quantize_4bit: bool = True,
    output_hidden_states: bool = True,
    hf_token: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load a HuggingFace causal LM with optional 4-bit quantization.

    Returns:
        (model, tokenizer) tuple
    """
    token = hf_token or os.environ.get("HF_TOKEN", "")

    kwargs = {"device_map": "auto", "token": token}

    if output_hidden_states:
        kwargs["output_hidden_states"] = True

    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_sycophancy_dataset(max_questions: int = 500) -> List[Dict]:
    """
    Parse the meg-tong/sycophancy-eval dataset into question dicts.

    Each dict contains:
        neutral_prompt, syco_prompt, correct_answer, incorrect_answer, question
    """
    ds = load_dataset("meg-tong/sycophancy-eval", data_files="answer.jsonl", split="train")

    questions: List[Dict] = []
    i = 0
    while i < len(ds) - 3:
        row_neutral = ds[i]
        row_syco = ds[i + 2]

        prompt_neutral = row_neutral["prompt"]
        prompt_syco = row_syco["prompt"]

        if isinstance(prompt_neutral, str):
            prompt_neutral = json.loads(prompt_neutral)
        if isinstance(prompt_syco, str):
            prompt_syco = json.loads(prompt_syco)

        neutral_text = _extract_human_content(prompt_neutral)
        syco_text = _extract_human_content(prompt_syco)

        base_info = row_neutral.get("base", {})
        if isinstance(base_info, str):
            base_info = json.loads(base_info)

        correct = base_info.get("correct_answer", "")
        incorrect = base_info.get("incorrect_answer", "")
        question = base_info.get("question", "")

        if neutral_text and syco_text and correct and incorrect:
            if incorrect.lower() in syco_text.lower():
                questions.append({
                    "neutral_prompt": neutral_text,
                    "syco_prompt": syco_text,
                    "correct_answer": correct,
                    "incorrect_answer": incorrect,
                    "question": question,
                })

        i += 4

    return questions[:max_questions]


def _extract_human_content(prompt_list: list) -> str:
    for msg in prompt_list:
        if msg.get("type") == "human":
            return msg["content"]
    return ""


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    target_layers: List[int],
    max_new_tokens: int = 80,
) -> Tuple[str, Dict[int, np.ndarray]]:
    """
    Generate a response and extract hidden states from the first generated token.

    Returns:
        (response_text, {layer_idx: hidden_state_vector})
    """
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    hidden_states: Dict[int, np.ndarray] = {}
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        first_token_hidden = outputs.hidden_states[0]
        for layer_idx in target_layers:
            if layer_idx < len(first_token_hidden):
                hidden_states[layer_idx] = (
                    first_token_hidden[layer_idx][0, -1, :].cpu().float().numpy()
                )

    return response, hidden_states


def extract_all_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 80,
) -> Tuple[str, Dict[int, np.ndarray]]:
    """
    Generate a response and extract hidden states from ALL layers.
    Used for mechanistic analysis (logit lens, activation patching).

    Returns:
        (response_text, {layer_idx: hidden_state_vector})
    """
    n_layers = model.config.num_hidden_layers
    all_layers = list(range(n_layers + 1))  # +1 for embedding layer (layer 0)
    return extract_hidden_states(model, tokenizer, prompt, all_layers, max_new_tokens)


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------

def check_answer_match(response: str, answer: str) -> bool:
    """Check if a response contains the expected answer (word-level matching)."""
    resp_lower = response.lower()
    answer_lower = answer.strip().lower()
    words = [w for w in answer_lower.split() if len(w) > 3]
    if words:
        return any(w in resp_lower for w in words)
    return answer_lower in resp_lower


# ---------------------------------------------------------------------------
# Probe training and evaluation
# ---------------------------------------------------------------------------

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Train a logistic regression probe with cross-validation.

    Returns dict with: accuracy, std, folds, chance_level
    """
    n_splits = min(n_splits, min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=random_seed, C=1.0, class_weight="balanced")
    bal_scorer = make_scorer(balanced_accuracy_score)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring=bal_scorer)

    n_classes = len(np.unique(y))
    return {
        "balanced_accuracy": float(scores.mean()),
        "std": float(scores.std()),
        "folds": [float(s) for s in scores],
        "chance_level": 1.0 / n_classes,
        "n_samples": len(y),
        "n_per_class": int(min(np.bincount(y))),
    }


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    observed_accuracy: float,
    n_permutations: int = 500,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Run a permutation test to assess statistical significance.

    Returns dict with: p_value, null_mean, null_std, n_permutations
    """
    n_splits = min(n_splits, min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    bal_scorer = make_scorer(balanced_accuracy_score)

    null_scores = []
    for i in range(n_permutations):
        y_perm = np.random.permutation(y)
        clf = LogisticRegression(max_iter=1000, random_state=random_seed, C=1.0)
        perm_scores = cross_val_score(clf, X_scaled, y_perm, cv=cv, scoring=bal_scorer)
        null_scores.append(perm_scores.mean())

    p_value = np.mean([s >= observed_accuracy for s in null_scores])

    return {
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_scores)),
        "null_std": float(np.std(null_scores)),
        "n_permutations": n_permutations,
    }


def length_baseline(
    responses_a: List[str],
    responses_b: List[str],
    random_seed: int = 42,
) -> float:
    """
    Train a classifier using only response length. Returns balanced accuracy.
    A result near 50% confirms that length is not a confound.
    """
    n = min(len(responses_a), len(responses_b))
    X = np.array([len(r) for r in responses_a[:n]] + [len(r) for r in responses_b[:n]]).reshape(-1, 1)
    y = np.array([0] * n + [1] * n)

    n_splits = min(5, n)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    clf = LogisticRegression(max_iter=1000, random_state=random_seed)
    bal_scorer = make_scorer(balanced_accuracy_score)
    scores = cross_val_score(clf, X, y, cv=cv, scoring=bal_scorer)
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(data: dict, path: str) -> None:
    """Save results dict to JSON with numpy support."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
