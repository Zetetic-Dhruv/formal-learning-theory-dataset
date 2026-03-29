#!/usr/bin/env python3
"""
Generate fine-tuning training data from the FLT concept graph.

Produces JSONL in chat format (system/user/assistant) suitable for
Qwen 2.5 fine-tuning. Covers both positive and negative space:
  - Positive: definitions, characterizations, bounds, proof dependencies
  - Negative: non-implications, separations, obstructions, scope boundaries

Usage:
    python scripts/generate_training_data.py \
        --graph dataset/flt_concept_graph.json \
        --output data/train.jsonl \
        [--held-out dataset/flt_tasks.json]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a formal learning theory expert. You answer questions about "
    "PAC learning, online learning, Gold's model, VC dimension, complexity "
    "measures, and their relationships. You cite theorems precisely, "
    "including separation witnesses and non-implications. When a relationship "
    "does NOT hold, you state so explicitly with the known counterexample."
)

# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_indices(graph: dict) -> dict:
    """Build lookup indices over nodes and edges."""
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    edges_by_relation = defaultdict(list)
    edges_from = defaultdict(list)
    edges_to = defaultdict(list)

    for e in graph["edges"]:
        edges_by_relation[e["relation"]].append(e)
        edges_from[e["source"]].append(e)
        edges_to[e["target"]].append(e)

    return {
        "nodes": nodes_by_id,
        "edges": graph["edges"],
        "by_relation": edges_by_relation,
        "from": edges_from,
        "to": edges_to,
    }


def node_name(idx: dict, nid: str) -> str:
    n = idx["nodes"].get(nid)
    return n["name"] if n else nid


# ---------------------------------------------------------------------------
# Chat message helpers
# ---------------------------------------------------------------------------

def make_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }


# ---------------------------------------------------------------------------
# Generators — Positive Space
# ---------------------------------------------------------------------------

def gen_definition_qa(idx: dict) -> list[dict]:
    """What is X? → formal definition."""
    examples = []
    for nid, node in idx["nodes"].items():
        if node.get("status") in ("deferred",):
            continue
        if node.get("category") == "scope_boundary":
            continue
        desc = node.get("description", "")
        formal = node.get("formal_definition", "")
        if not formal:
            continue
        name = node["name"]

        # Variant 1: direct definition
        q = f"What is the formal definition of {name}?"
        a = formal
        if node.get("provenance"):
            prov = node["provenance"]
            if "introduced_by" in prov:
                a += f"\n\nIntroduced by {prov['introduced_by']}"
                if "year" in prov:
                    a += f" ({prov['year']})"
                a += "."
        examples.append(make_example(q, a))

        # Variant 2: role question
        if desc and desc != formal:
            q2 = f"What role does {name} play in learning theory?"
            examples.append(make_example(q2, desc))

    return examples


def gen_prerequisite_qa(idx: dict) -> list[dict]:
    """What must be defined before X?"""
    examples = []
    for nid, node in idx["nodes"].items():
        deps = [e["target"] for e in idx["from"].get(nid, [])
                if e["relation"] == "defined_using"]
        if not deps:
            continue
        name = node["name"]
        dep_names = [node_name(idx, d) for d in deps]
        q = f"What concepts must be defined before {name} can be stated?"
        a = f"{name} is defined using: {', '.join(dep_names)}."
        examples.append(make_example(q, a))
    return examples


def gen_characterization_qa(idx: dict) -> list[dict]:
    """X characterizes Y — equivalence."""
    examples = []
    for e in idx["by_relation"].get("characterizes", []):
        src = node_name(idx, e["source"])
        tgt = node_name(idx, e["target"])
        note = e.get("note", "")
        citation = e.get("citation", "")

        q = f"Does {src} characterize {tgt}?"
        a = f"Yes. {src} characterizes {tgt} (equivalence)."
        if note:
            a += f" {note}"
        if citation:
            a += f" [{citation}]"
        examples.append(make_example(q, a))

        # Reverse direction
        q2 = f"What characterizes {tgt}?"
        chars = [edge for edge in idx["to"].get(e["target"], [])
                 if edge["relation"] == "characterizes"]
        char_names = [node_name(idx, c["source"]) for c in chars]
        a2 = f"{tgt} is characterized by: {', '.join(char_names)}."
        examples.append(make_example(q2, a2))
    return examples


def gen_bounds_qa(idx: dict) -> list[dict]:
    """Upper/lower bound relationships."""
    examples = []
    for rel in ("upper_bounds", "lower_bounds"):
        direction = "upper" if "upper" in rel else "lower"
        for e in idx["by_relation"].get(rel, []):
            src = node_name(idx, e["source"])
            tgt = node_name(idx, e["target"])
            note = e.get("note", "")
            citation = e.get("citation", "")

            q = f"What is the relationship between {src} and {tgt}?"
            a = f"{src} provides a {direction} bound on {tgt}."
            if note:
                a += f" {note}"
            if citation:
                a += f" [{citation}]"
            examples.append(make_example(q, a))
    return examples


def gen_hierarchy_qa(idx: dict) -> list[dict]:
    """Strictly stronger / restricts / extends_grammar chains."""
    examples = []
    for rel in ("strictly_stronger", "restricts", "extends_grammar"):
        for e in idx["by_relation"].get(rel, []):
            src = node_name(idx, e["source"])
            tgt = node_name(idx, e["target"])
            witness = e.get("witness", "")
            note = e.get("note", "")
            citation = e.get("citation", "")

            if rel == "strictly_stronger":
                q = f"Is {src} strictly stronger than {tgt}?"
                a = f"Yes. {src} ⊋ {tgt}."
                if witness:
                    a += f" Witness: {witness}."
            elif rel == "restricts":
                q = f"How does {src} relate to {tgt} as a generalization?"
                a = f"{src} generalizes {tgt} by removing a constraint (no new grammar needed)."
            else:
                q = f"How does {src} extend {tgt}?"
                a = f"{src} extends the grammar of {tgt} by introducing new formal primitives."

            if note:
                a += f" {note}"
            if citation:
                a += f" [{citation}]"
            examples.append(make_example(q, a))
    return examples


def gen_proof_dependency_qa(idx: dict) -> list[dict]:
    """used_in_proof edges."""
    examples = []
    for e in idx["by_relation"].get("used_in_proof", []):
        src = node_name(idx, e["source"])
        tgt = node_name(idx, e["target"])
        note = e.get("note", "")
        citation = e.get("citation", "")

        q = f"What role does {tgt} play in proving {src}?"
        a = f"The proof of {src} uses {tgt}."
        if note:
            a += f" {note}"
        if citation:
            a += f" [{citation}]"
        examples.append(make_example(q, a))
    return examples


def gen_instance_qa(idx: dict) -> list[dict]:
    """instance_of edges."""
    examples = []
    for e in idx["by_relation"].get("instance_of", []):
        src = node_name(idx, e["source"])
        tgt = node_name(idx, e["target"])

        q = f"Is {src} an instance of {tgt}?"
        a = f"Yes. {src} is a specific instance/specialization of {tgt}."
        examples.append(make_example(q, a))
    return examples


def gen_measures_qa(idx: dict) -> list[dict]:
    """measures edges."""
    examples = []
    for e in idx["by_relation"].get("measures", []):
        src = node_name(idx, e["source"])
        tgt = node_name(idx, e["target"])

        q = f"What does {src} measure?"
        a = f"{src} is a complexity measure that quantifies a property of {tgt}."
        examples.append(make_example(q, a))
    return examples


# ---------------------------------------------------------------------------
# Generators — Negative Space
# ---------------------------------------------------------------------------

def gen_non_implication_qa(idx: dict) -> list[dict]:
    """does_not_imply edges — the most valuable for proof discovery."""
    examples = []
    for e in idx["by_relation"].get("does_not_imply", []):
        src = node_name(idx, e["source"])
        tgt = node_name(idx, e["target"])
        witness = e.get("witness", "")
        note = e.get("note", "")
        citation = e.get("citation", "")

        # Direct question
        q = f"Does {src} imply {tgt}?"
        a = f"NO. {src} does NOT imply {tgt}."
        if witness:
            a += f"\n\nWitness/counterexample: {witness}."
        if note:
            a += f"\n\n{note}"
        if citation:
            a += f" [{citation}]"
        examples.append(make_example(q, a))

        # Variant: "Can we conclude X from Y?"
        q2 = f"If a concept class satisfies {src}, can we conclude {tgt}?"
        examples.append(make_example(q2, a))

        # Witness-focused variant
        if witness:
            q3 = f"What is the separation witness between {src} and {tgt}?"
            a3 = f"The witness separating {src} from {tgt} is: {witness}."
            if citation:
                a3 += f" [{citation}]"
            examples.append(make_example(q3, a3))

    return examples


def gen_obstruction_qa(idx: dict) -> list[dict]:
    """Analogy edges with obstructions — what blocks formal upgrade."""
    examples = []
    for e in idx["by_relation"].get("analogy", []):
        obstruction = e.get("obstruction", "")
        if not obstruction:
            continue
        src = node_name(idx, e["source"])
        tgt = node_name(idx, e["target"])
        obs_type = e.get("obstruction_type", "unknown")

        q = f"Why is the analogy between {src} and {tgt} not a formal theorem?"
        a = f"The analogy between {src} and {tgt} is structural but not formally provable."
        a += f"\n\nObstruction ({obs_type}): {obstruction}"
        examples.append(make_example(q, a))

        # Variant: can we formalize this?
        q2 = f"Can the relationship between {src} and {tgt} be made into a theorem?"
        a2 = f"Not directly. There is a structural analogy, but it is blocked by: {obstruction}"
        examples.append(make_example(q2, a2))

    return examples


def gen_scope_boundary_qa(idx: dict) -> list[dict]:
    """Scope boundaries — what is explicitly excluded and why."""
    examples = []
    for nid, node in idx["nodes"].items():
        if node.get("category") != "scope_boundary":
            continue
        name = node["name"]
        desc = node.get("description", "")

        q = f"Is {name} covered in formal learning theory?"
        a = f"Explicitly excluded. {desc}"
        examples.append(make_example(q, a))
    return examples


def gen_requires_assumption_qa(idx: dict) -> list[dict]:
    """Conditional results — what assumptions are needed."""
    examples = []
    for e in idx["by_relation"].get("requires_assumption", []):
        src = node_name(idx, e["source"])
        tgt = node_name(idx, e["target"])
        note = e.get("note", "")
        citation = e.get("citation", "")

        q = f"What assumptions does {src} require?"
        a = f"{src} requires an assumption related to {tgt}."
        if note:
            a += f" {note}"
        if citation:
            a += f" [{citation}]"
        examples.append(make_example(q, a))
    return examples


# ---------------------------------------------------------------------------
# Generators — Multi-hop reasoning chains
# ---------------------------------------------------------------------------

def gen_multihop_prereq_chains(idx: dict) -> list[dict]:
    """Recursive prerequisite chains via BFS on defined_using."""
    examples = []
    for nid, node in idx["nodes"].items():
        if node.get("category") == "scope_boundary":
            continue
        # BFS
        visited = []
        queue = [nid]
        seen = {nid}
        while queue:
            current = queue.pop(0)
            deps = [e["target"] for e in idx["from"].get(current, [])
                    if e["relation"] == "defined_using"]
            for d in deps:
                if d not in seen:
                    seen.add(d)
                    visited.append(d)
                    queue.append(d)
        if len(visited) < 2:
            continue

        name = node["name"]
        chain_names = [node_name(idx, v) for v in visited]
        q = f"What is the full dependency chain for defining {name}?"
        a = f"To define {name}, you need (in dependency order): {' → '.join(chain_names)}."
        a += f"\n\nThis chain has {len(visited)} prerequisite concepts."
        examples.append(make_example(q, a))
    return examples


def gen_paradigm_comparison(idx: dict) -> list[dict]:
    """Cross-paradigm questions combining positive and negative edges."""
    examples = []

    # Find all does_not_imply and strictly_stronger for paradigm comparisons
    separations = idx["by_relation"].get("does_not_imply", []) + \
                  idx["by_relation"].get("strictly_stronger", [])

    for e in separations:
        src = node_name(idx, e["source"])
        tgt = node_name(idx, e["target"])
        rel = e["relation"]
        witness = e.get("witness", "")

        # What CAN we say?
        bounds_from_src = [b for b in idx["from"].get(e["source"], [])
                          if b["relation"] in ("upper_bounds", "characterizes")]
        if not bounds_from_src:
            continue

        q = f"Compare {src} and {tgt}: what holds and what doesn't?"
        a = f"Relationship between {src} and {tgt}:\n\n"

        if rel == "does_not_imply":
            a += f"DOES NOT HOLD: {src} does NOT imply {tgt}."
            if witness:
                a += f" Witness: {witness}."
        else:
            a += f"STRICT HIERARCHY: {src} ⊋ {tgt}."
            if witness:
                a += f" Witness: {witness}."

        a += "\n\nWhat DOES hold for " + src + ":\n"
        for b in bounds_from_src[:3]:
            btgt = node_name(idx, b["target"])
            a += f"- {src} {b['relation']} {btgt}\n"

        examples.append(make_example(q, a))

    return examples


def gen_negative_probe_qa(idx: dict) -> list[dict]:
    """Generate questions where the answer is 'no' from absence of edges.

    Key for proof discovery: the model must learn to say 'there is no known
    relationship' when the graph has no edge, rather than hallucinating one.
    """
    examples = []
    all_node_ids = [nid for nid, n in idx["nodes"].items()
                    if n.get("category") != "scope_boundary"
                    and n.get("status") != "deferred"]

    # Build set of existing edges for fast lookup
    existing = set()
    for e in idx["edges"]:
        existing.add((e["source"], e["target"], e["relation"]))

    # Sample pairs with no characterizes edge
    rng = random.Random(42)
    char_sources = {e["source"] for e in idx["by_relation"].get("characterizes", [])}
    char_targets = {e["target"] for e in idx["by_relation"].get("characterizes", [])}

    for _ in range(50):
        a_id = rng.choice(all_node_ids)
        b_id = rng.choice(all_node_ids)
        if a_id == b_id:
            continue
        if (a_id, b_id, "characterizes") in existing:
            continue
        if (a_id, b_id, "does_not_imply") in existing:
            continue

        a_name = node_name(idx, a_id)
        b_name = node_name(idx, b_id)
        q = f"Does {a_name} characterize {b_name}?"
        a = f"No. There is no known characterization equivalence between {a_name} and {b_name} in the formal learning theory literature."

        # Check if there IS some other relationship
        other_edges = [e for e in idx["from"].get(a_id, [])
                       if e["target"] == b_id]
        if other_edges:
            e0 = other_edges[0]
            a += f"\n\nHowever, {a_name} {e0['relation']} {b_name}."

        examples.append(make_example(q, a))

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate FLT training data")
    parser.add_argument("--graph", default="dataset/flt_concept_graph.json")
    parser.add_argument("--output", default="data/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    graph = load_graph(args.graph)
    idx = build_indices(graph)

    generators = {
        # Positive space
        "definition": gen_definition_qa,
        "prerequisite": gen_prerequisite_qa,
        "characterization": gen_characterization_qa,
        "bounds": gen_bounds_qa,
        "hierarchy": gen_hierarchy_qa,
        "proof_dependency": gen_proof_dependency_qa,
        "instance": gen_instance_qa,
        "measures": gen_measures_qa,
        # Negative space
        "non_implication": gen_non_implication_qa,
        "obstruction": gen_obstruction_qa,
        "scope_boundary": gen_scope_boundary_qa,
        "requires_assumption": gen_requires_assumption_qa,
        # Multi-hop / composite
        "multihop_prereq": gen_multihop_prereq_chains,
        "paradigm_comparison": gen_paradigm_comparison,
        "negative_probe": gen_negative_probe_qa,
    }

    all_examples = []
    counts = {}
    for name, gen_fn in generators.items():
        examples = gen_fn(idx)
        counts[name] = len(examples)
        all_examples.extend(examples)

    random.shuffle(all_examples)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary
    print(f"Generated {len(all_examples)} training examples:")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        pos_neg = "negative" if name in (
            "non_implication", "obstruction", "scope_boundary",
            "requires_assumption", "negative_probe"
        ) else "positive" if name not in (
            "multihop_prereq", "paradigm_comparison"
        ) else "composite"
        print(f"  {name:25s} {count:4d}  [{pos_neg}]")

    neg_count = sum(counts[k] for k in (
        "non_implication", "obstruction", "scope_boundary",
        "requires_assumption", "negative_probe"
    ))
    print(f"\nPositive/composite: {len(all_examples) - neg_count}")
    print(f"Negative space:     {neg_count}")
    print(f"Total:              {len(all_examples)}")
    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
