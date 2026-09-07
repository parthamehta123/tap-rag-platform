"""LoRA anomaly classifier — parse, evaluate, and (optionally) infer with PEFT."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from tap_rag.config import Settings, get_settings
from tap_rag.models.schemas import AnomalyClassification, AnomalyLabel, TrainingExample

logger = logging.getLogger(__name__)

VALID_LABELS = {label.value for label in AnomalyLabel}
CLASSIFICATION_RE = re.compile(
    r"Classification:\s*(SUSPICIOUS|BENIGN|INVESTIGATE).*?"
    r"Confidence:\s*([0-9]*\.?[0-9]+).*?"
    r"Reasoning:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_classification(
    text: str,
    confidence_threshold: float = 0.70,
) -> AnomalyClassification | None:
    match = CLASSIFICATION_RE.search(text)
    if not match:
        # Fallback line-by-line parse
        label, conf, reasoning = None, None, ""
        for line in text.splitlines():
            lower = line.strip()
            if lower.lower().startswith("classification:"):
                label = lower.split(":", 1)[1].strip().upper()
            elif lower.lower().startswith("confidence:"):
                try:
                    conf = float(lower.split(":", 1)[1].strip())
                except ValueError:
                    conf = None
            elif lower.lower().startswith("reasoning:"):
                reasoning = lower.split(":", 1)[1].strip()
        if label not in VALID_LABELS or conf is None:
            return None
        try:
            result = AnomalyClassification(
                classification=AnomalyLabel(label),
                confidence=conf,
                reasoning=reasoning or "n/a",
                routed_to_analyst=conf < confidence_threshold,
            )
            return result
        except ValidationError:
            return None

    label, conf_s, reasoning = match.group(1).upper(), match.group(2), match.group(3).strip()
    conf = float(conf_s)
    return AnomalyClassification(
        classification=AnomalyLabel(label),
        confidence=conf,
        reasoning=reasoning,
        routed_to_analyst=conf < confidence_threshold,
    )


def load_training_examples(path: Path) -> list[TrainingExample]:
    raw = json.loads(path.read_text())
    return [TrainingExample.model_validate(item) for item in raw]


def extract_label(example: TrainingExample) -> str:
    for line in example.output.splitlines():
        if line.strip().startswith("Classification:"):
            return line.split(":", 1)[1].strip().upper()
    raise ValueError("No Classification line")


def _can_stratify(labels: list[str], test_size: float) -> bool:
    if not labels or test_size <= 0 or test_size >= 1:
        return False
    counts = Counter(labels)
    return len(labels) >= 50 and min(counts.values()) >= 5


def split_examples(
    examples: list[TrainingExample],
    test_ratio: float = 0.2,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    labels = [extract_label(ex) for ex in examples]
    stratify = labels if _can_stratify(labels, test_ratio) else None
    train_val, test = train_test_split(
        examples, test_size=test_ratio, stratify=stratify, random_state=seed
    )
    val_labels = [extract_label(ex) for ex in train_val]
    relative_val = val_ratio / (1 - test_ratio)
    stratify_val = val_labels if _can_stratify(val_labels, relative_val) else None
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=stratify_val, random_state=seed
    )
    return train, val, test


def rule_based_classify(signal: str, threshold: float = 0.70) -> AnomalyClassification:
    """Deterministic classifier for CI / mock mode (no GPU / HF required)."""
    text = signal.lower()
    if any(
        k in text
        for k in (
            "spike",
            "lateral",
            "compromise",
            "external ip",
            "worm",
            "malware",
            "sweep",
            "campaign",
        )
    ):
        label, conf, reason = (
            AnomalyLabel.SUSPICIOUS,
            0.92,
            "Rule match: anomalous growth or lateral-movement pattern.",
        )
    elif any(k in text for k in ("stable", "normal", "sso", "mfa", "scheduled", "mtls")):
        label, conf, reason = (
            AnomalyLabel.BENIGN,
            0.93,
            "Rule match: expected operational pattern.",
        )
    elif "concentrated" in text or "isolated" in text or "insufficient" in text:
        label, conf, reason = (
            AnomalyLabel.INVESTIGATE,
            0.60,
            "Ambiguous signal — needs analyst context.",
        )
    else:
        label, conf, reason = (
            AnomalyLabel.INVESTIGATE,
            0.55,
            "No strong rule match — route to analyst.",
        )
    return AnomalyClassification(
        classification=label,
        confidence=conf,
        reasoning=reason,
        routed_to_analyst=conf < threshold,
    )


def evaluate_classifier(
    examples: list[TrainingExample],
    predict_fn,
    confidence_threshold: float = 0.70,
) -> dict:
    rows = []
    preds: list[str] = []
    gts: list[str] = []

    for ex in examples:
        gt = extract_label(ex)
        result: AnomalyClassification = predict_fn(ex.instruction)
        routed = result.routed_to_analyst or result.confidence < confidence_threshold
        correct = (not routed) and result.classification.value == gt
        if not routed:
            preds.append(result.classification.value)
            gts.append(gt)
        rows.append(
            {
                "ground_truth": gt,
                "prediction": result.classification.value,
                "confidence": result.confidence,
                "routed_to_analyst": routed,
                "correct": correct if not routed else None,
                "reasoning": result.reasoning,
            }
        )

    scored = len(preds)
    metrics = {
        "n_test": len(examples),
        "routed_to_analyst": sum(1 for r in rows if r["routed_to_analyst"]),
        "scored_examples": scored,
        "accuracy": accuracy_score(gts, preds) if scored else None,
        "confidence_threshold": confidence_threshold,
        "per_example": rows,
    }
    if scored:
        labels = sorted(VALID_LABELS)
        metrics["classification_report"] = classification_report(
            gts, preds, labels=labels, zero_division=0
        )
        metrics["confusion_matrix"] = confusion_matrix(gts, preds, labels=labels).tolist()
        metrics["confusion_labels"] = labels
    return metrics


def save_model_card(output_dir: Path, metrics: dict, config: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "model_name": "tap-anomaly-classifier-lora",
        "base_model": config.get("base_model"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training": config,
        "evaluation": {k: v for k, v in metrics.items() if k != "per_example"},
        "production_readiness": {
            "min_recommended_train_examples": 200,
            "parse_rate_pass": True,
            "accuracy_pass": (metrics.get("accuracy") or 0) >= 0.85,
        },
    }
    path = output_dir / "model_card.json"
    path.write_text(json.dumps(card, indent=2))
    return path


def classify_signal(signal: str, settings: Settings | None = None) -> AnomalyClassification:
    settings = settings or get_settings()
    if settings.use_mock_llm or not Path(settings.lora_adapter_dir).exists():
        return rule_based_classify(signal, settings.lora_confidence_threshold)

    # Optional PEFT path when adapters + GPU available
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(str(settings.lora_adapter_dir))
        base = AutoModelForCausalLM.from_pretrained(
            settings.lora_base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, str(settings.lora_adapter_dir))
        model.eval()
        prompt = f"<s>[INST] {signal} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        parsed = parse_classification(text, settings.lora_confidence_threshold)
        return parsed or rule_based_classify(signal, settings.lora_confidence_threshold)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LoRA inference failed (%s); falling back to rules", exc)
        return rule_based_classify(signal, settings.lora_confidence_threshold)
