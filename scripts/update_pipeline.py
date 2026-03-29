#!/usr/bin/env python3
"""
Update pipeline: regenerate training data and retrain when the knowledge base changes.

Designed for the workflow where:
  1. The Lean 4 formalization repo produces new theorems / synthetic discoveries
  2. Those get merged into flt_concept_graph.json (new nodes/edges)
  3. This script detects changes, regenerates training data, and triggers retraining

Usage:
    # Full update: regenerate data + retrain
    python scripts/update_pipeline.py --full

    # Data only: regenerate training data (no retraining)
    python scripts/update_pipeline.py --data-only

    # Diff mode: show what changed since last training
    python scripts/update_pipeline.py --diff

    # Incremental: only generate examples for NEW nodes/edges
    python scripts/update_pipeline.py --incremental
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


STATE_FILE = Path("data/update_state.json")
GRAPH_PATH = Path("dataset/flt_concept_graph.json")
TRAIN_FILE = Path("data/train.jsonl")
INCREMENTAL_FILE = Path("data/incremental.jsonl")


def graph_hash(path: Path) -> str:
    """SHA256 of the graph file for change detection."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_graph(path: Path) -> dict:
    return json.loads(path.read_text())


def diff_graphs(old_graph: dict | None, new_graph: dict) -> dict:
    """Compute structural diff between two graph versions."""
    if old_graph is None:
        return {
            "new_nodes": [n["id"] for n in new_graph["nodes"]],
            "new_edges": list(range(len(new_graph["edges"]))),
            "removed_nodes": [],
            "removed_edges": [],
            "modified_nodes": [],
        }

    old_node_ids = {n["id"] for n in old_graph["nodes"]}
    new_node_ids = {n["id"] for n in new_graph["nodes"]}

    old_edge_keys = {
        (e["source"], e["target"], e["relation"])
        for e in old_graph["edges"]
    }
    new_edge_keys = set()
    new_edge_indices = []
    for i, e in enumerate(new_graph["edges"]):
        key = (e["source"], e["target"], e["relation"])
        new_edge_keys.add(key)
        if key not in old_edge_keys:
            new_edge_indices.append(i)

    # Check for modified nodes (same id, different content)
    old_nodes_by_id = {n["id"]: n for n in old_graph["nodes"]}
    modified = []
    for n in new_graph["nodes"]:
        if n["id"] in old_nodes_by_id:
            if json.dumps(n, sort_keys=True) != json.dumps(old_nodes_by_id[n["id"]], sort_keys=True):
                modified.append(n["id"])

    return {
        "new_nodes": sorted(new_node_ids - old_node_ids),
        "removed_nodes": sorted(old_node_ids - new_node_ids),
        "new_edges": new_edge_indices,
        "removed_edges": [
            (s, t, r) for s, t, r in old_edge_keys if (s, t, r) not in new_edge_keys
        ],
        "modified_nodes": modified,
    }


def run_data_generation(incremental_ids: list[str] | None = None):
    """Run the data generation script."""
    cmd = [sys.executable, "scripts/generate_training_data.py",
           "--graph", str(GRAPH_PATH), "--output", str(TRAIN_FILE)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def run_training():
    """Run the training script."""
    cmd = [sys.executable, "scripts/train.py", "--config", "config/training_config.yaml"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Training failed!", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="FLT update pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full", action="store_true",
                       help="Full update: regenerate data + retrain")
    group.add_argument("--data-only", action="store_true",
                       help="Only regenerate training data")
    group.add_argument("--diff", action="store_true",
                       help="Show diff since last training")
    group.add_argument("--incremental", action="store_true",
                       help="Generate examples only for new content")
    args = parser.parse_args()

    state = load_state()
    current_hash = graph_hash(GRAPH_PATH)
    new_graph = load_graph(GRAPH_PATH)

    # Load previous graph snapshot if available
    old_graph = None
    snapshot_path = Path("data/last_trained_graph.json")
    if snapshot_path.exists():
        old_graph = load_graph(snapshot_path)

    diff = diff_graphs(old_graph, new_graph)
    has_changes = (
        diff["new_nodes"] or diff["removed_nodes"] or
        diff["new_edges"] or diff["removed_edges"] or
        diff["modified_nodes"]
    )

    if args.diff:
        print("=== Graph Diff ===")
        if not has_changes:
            print("No changes since last training.")
        else:
            if diff["new_nodes"]:
                print(f"\nNew nodes ({len(diff['new_nodes'])}):")
                for nid in diff["new_nodes"]:
                    print(f"  + {nid}")
            if diff["removed_nodes"]:
                print(f"\nRemoved nodes ({len(diff['removed_nodes'])}):")
                for nid in diff["removed_nodes"]:
                    print(f"  - {nid}")
            if diff["new_edges"]:
                print(f"\nNew edges ({len(diff['new_edges'])}):")
                for i in diff["new_edges"][:20]:
                    e = new_graph["edges"][i]
                    print(f"  + {e['source']} --[{e['relation']}]--> {e['target']}")
                if len(diff["new_edges"]) > 20:
                    print(f"  ... and {len(diff['new_edges'])-20} more")
            if diff["modified_nodes"]:
                print(f"\nModified nodes ({len(diff['modified_nodes'])}):")
                for nid in diff["modified_nodes"]:
                    print(f"  ~ {nid}")
        return

    if not has_changes and state.get("graph_hash") == current_hash:
        print("No changes detected. Skipping update.")
        if not args.full:
            return
        print("--full flag set, proceeding anyway.")

    # Regenerate data
    print(f"\n{'='*60}")
    print(f"UPDATE PIPELINE — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}")
    print(f"Graph hash: {current_hash[:16]}...")
    print(f"New nodes: {len(diff['new_nodes'])}")
    print(f"New edges: {len(diff['new_edges'])}")
    print(f"Modified:  {len(diff['modified_nodes'])}")
    print()

    run_data_generation()

    if args.data_only or args.incremental:
        print("\nData regenerated. Skipping training (--data-only / --incremental).")
    elif args.full:
        print("\nStarting training...")
        run_training()

    # Save state
    state["graph_hash"] = current_hash
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    state["last_diff"] = {
        "new_nodes": len(diff["new_nodes"]),
        "new_edges": len(diff["new_edges"]),
        "removed_nodes": len(diff["removed_nodes"]),
        "modified_nodes": len(diff["modified_nodes"]),
    }
    save_state(state)

    # Save graph snapshot for next diff
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(new_graph))

    print("\nUpdate complete.")
    print(f"State saved to {STATE_FILE}")


if __name__ == "__main__":
    main()
