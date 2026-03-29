#!/usr/bin/env python3
"""
Evaluate the fine-tuned model against the FLT benchmark tasks.

Uses flt_tasks.json as questions and flt_task_answers.json as gold answers.
Reports per-task scores and overall accuracy.

Usage:
    python scripts/evaluate.py \
        --model checkpoints/flt-qwen2.5-7b/final \
        --tasks dataset/flt_tasks.json \
        --answers dataset/flt_task_answers.json \
        --output eval_results.json
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "You are a formal learning theory expert. You answer questions about "
    "PAC learning, online learning, Gold's model, VC dimension, complexity "
    "measures, and their relationships. You cite theorems precisely, "
    "including separation witnesses and non-implications. When a relationship "
    "does NOT hold, you state so explicitly with the known counterexample."
)


def load_model(model_path: str, base_model: str | None = None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if base_model:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.1, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def score_set_match(response: str, expected: list[str]) -> dict:
    """Score set_match tasks: what fraction of expected items appear in response."""
    response_lower = response.lower()
    found = [item for item in expected if item.lower().replace("_", " ") in response_lower
             or item.lower() in response_lower]
    precision = len(found) / max(len(expected), 1)
    return {"score": precision, "found": found, "missing": [x for x in expected if x not in found]}


def score_chain_match(response: str, expected: list[str]) -> dict:
    """Score chain_match tasks: are items in the correct order?"""
    response_lower = response.lower()
    positions = []
    for item in expected:
        name = item.lower().replace("_", " ")
        pos = response_lower.find(name)
        if pos == -1:
            pos = response_lower.find(item.lower())
        positions.append(pos)

    found_positions = [(item, pos) for item, pos in zip(expected, positions) if pos >= 0]
    if len(found_positions) < 2:
        return {"score": len(found_positions) / len(expected), "order_correct": None}

    in_order = all(found_positions[i][1] <= found_positions[i+1][1]
                   for i in range(len(found_positions)-1))
    coverage = len(found_positions) / len(expected)
    score = coverage * (1.0 if in_order else 0.5)
    return {"score": score, "order_correct": in_order, "coverage": coverage}


def score_structured(response: str, expected: dict) -> dict:
    """Score structured_explanation tasks: check required elements."""
    response_lower = response.lower()
    required = expected.get("required_elements", {})
    matched = 0
    details = {}

    for key, value in required.items():
        if isinstance(value, bool):
            # Check if the concept is mentioned
            concept = key.replace("_", " ").lower()
            present = concept in response_lower or key.lower() in response_lower
            details[key] = present
            if present:
                matched += 1
        elif isinstance(value, str):
            present = value.lower() in response_lower
            details[key] = present
            if present:
                matched += 1
        elif isinstance(value, (int, float)):
            present = str(value) in response
            details[key] = present
            if present:
                matched += 1
        elif isinstance(value, list):
            found = sum(1 for v in value if v.lower().replace("_", " ") in response_lower
                        or v.lower() in response_lower)
            details[key] = f"{found}/{len(value)}"
            matched += found / max(len(value), 1)

    score = matched / max(len(required), 1)
    return {"score": score, "details": details}


def evaluate_task(response: str, answer: dict) -> dict:
    scoring = answer.get("scoring", answer.get("type", ""))
    expected = answer.get("expected", {})

    if scoring == "set_match":
        return score_set_match(response, expected)
    elif scoring == "chain_match":
        return score_chain_match(response, expected)
    elif scoring == "structured_explanation":
        return score_structured(response, expected)
    else:
        return {"score": 0, "error": f"Unknown scoring type: {scoring}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/flt-qwen2.5-7b/final")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--tasks", default="dataset/flt_tasks.json")
    parser.add_argument("--answers", default="dataset/flt_task_answers.json")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks_data = json.load(f)
    with open(args.answers) as f:
        answers_data = json.load(f)

    tasks = tasks_data["tasks"]
    answers = answers_data["answers"]

    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, args.base_model)

    results = []
    total_score = 0

    for task in tasks:
        tid = task["id"]
        question = task["question"]
        answer = answers.get(tid, {})

        print(f"\n{'='*60}")
        print(f"Task {tid}: {task['type']} (difficulty: {task['difficulty']})")
        print(f"Q: {question}")

        response = generate(model, tokenizer, question)
        print(f"A: {response[:200]}...")

        result = evaluate_task(response, answer)
        result["task_id"] = tid
        result["task_type"] = task["type"]
        result["difficulty"] = task["difficulty"]
        result["response"] = response
        results.append(result)

        score = result.get("score", 0)
        total_score += score
        print(f"Score: {score:.2f}")

    avg_score = total_score / len(tasks)
    summary = {
        "model": args.model,
        "total_tasks": len(tasks),
        "average_score": round(avg_score, 3),
        "by_difficulty": {},
        "results": results,
    }

    # Score by difficulty
    for diff in ("easy", "medium", "hard"):
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            avg = sum(r["score"] for r in diff_results) / len(diff_results)
            summary["by_difficulty"][diff] = {
                "count": len(diff_results),
                "average_score": round(avg, 3),
            }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"OVERALL: {avg_score:.3f} ({len(tasks)} tasks)")
    for diff, stats in summary["by_difficulty"].items():
        print(f"  {diff}: {stats['average_score']:.3f} ({stats['count']} tasks)")
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
