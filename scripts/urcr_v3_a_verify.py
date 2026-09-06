#!/usr/bin/env python3
"""Bounded A1 replay, immutable launch identity, and one-time smoke calibration."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.ppo.plan06_checkpointing import require_complete_checkpoint
from verl.trainer.ppo.urcr_v3_focus import FOCUS_PREAMBLE, build_visible_focus, validate_focus_config
from verl.trainer.ppo.urcr_v3_focus_loss import calibrate_focus_coefficient

REPO = Path(__file__).resolve().parents[1]
FAMILY = Path("/data0/kongmu/agentic-RL/runtime/urcr_v3_a_visible_focus")
V2_CONFIG = Path("/data2/kongmu/agentic-RL-runtime/urcr_v2_fixed_support_agam/formal_4gpu/segment_localmax0p01_resume_s150_010/resolved_config.json")
# User 2026-09-06: reuse the V2 rolling slot after retaining only S200/S300 inference exports.
ROLLING_ROOT = Path("/data0/kongmu/agentic-RL/runtime/urcr_v2_fixed_support_agam/rolling_checkpoints")
TRAINING_FILES = (
    "verl/trainer/main_evisd.py", "verl/trainer/ppo/evisd_ray_trainer.py",
    "verl/trainer/ppo/urcr_v3_focus.py", "verl/trainer/ppo/urcr_v3_focus_loss.py",
    "verl/workers/actor/dp_actor.py", "verl/workers/fsdp_workers.py",
    "examples/urcr_online/configs/urcr_v3_a_visible_focus.yaml",
    "examples/urcr_online/configs/plan07_answer_agam_core.yaml",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def config():
    with initialize_config_dir(config_dir=str(REPO / "examples/urcr_online/configs"), version_base=None):
        return compose(config_name="urcr_v3_a_visible_focus")


def verify_resume(checkpoint, original_config):
    checkpoint = Path(checkpoint).resolve()
    if FAMILY not in checkpoint.parents and checkpoint.parent != ROLLING_ROOT:
        raise ValueError("A1 resume must belong to the current isolated run family")
    step = int(checkpoint.name.removeprefix("global_step_"))
    require_complete_checkpoint(checkpoint, global_step=step, expected_world_size=4)
    saved = json.loads((checkpoint / "resolved_config.json").read_text())
    expected = OmegaConf.to_container(original_config, resolve=True)
    for key in ("actor_rollout_ref", "algorithm", "data", "env", "reward_model"):
        if saved[key] != expected[key]:
            raise ValueError(f"A1 resume configuration mismatch in {key}")
    if saved["trainer"]["project_name"] != "urcr_v3_a_visible_focus":
        raise ValueError("Not an A1 checkpoint")
    return step


def prepare(args):
    phase = args.phase
    run = Path(args.run_dir).resolve()
    if run.parent != FAMILY or run.exists():
        raise ValueError("Use a new direct child of the isolated A1 runtime family")
    c = config()
    previous = json.loads(V2_CONFIG.read_text())
    for key in ("actor_rollout_ref", "data", "env", "ray_init", "reward_model"):
        if OmegaConf.to_container(c[key], resolve=True) != previous[key]:
            raise ValueError(f"Base V3/V2 scientific configuration drift in {key}")
    if args.actor_micro not in (1, 2, 4, 8, 16) or args.teacher_micro not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Only A1-authorized memory microbatch sizes are supported")
    c.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = args.actor_micro
    c.algorithm.urcr_v3_focus.teacher_micro_batch_size = args.teacher_micro
    c.algorithm.urcr_v3_focus.output_dir = str(run)
    c.trainer.default_local_dir = str(ROLLING_ROOT if phase == "formal" else run / "checkpoints/rolling")
    c.trainer.experiment_name = f"urcr_v3_a_visible_focus_A1_{run.name}"
    hashes = {name: digest(REPO / name) for name in TRAINING_FILES}
    if phase == "smoke":
        c.trainer.stop_after_steps = 2
        c.algorithm.urcr_v3_focus.audit_all_minibatches = True
        c.algorithm.urcr_v3_focus.view_audit_steps = [1, 2]
    elif phase == "formal":
        calibration = json.loads((FAMILY / "calibration.json").read_text())
        if calibration["training_file_hashes"] != hashes:
            raise ValueError("Training implementation changed after smoke calibration")
        c.algorithm.urcr_v3_focus.coefficient_max = calibration["coefficient_max"]
        if (args.actor_micro, args.teacher_micro) != (calibration["actor_micro"], calibration["teacher_micro"]):
            raise ValueError("Formal microbatches must match the successful smoke")
    else:
        smoke = FAMILY / args.smoke_name
        original = OmegaConf.load(smoke / "launch_config.yaml")
        checkpoint = smoke / "checkpoints/rolling/global_step_2"
        verify_resume(checkpoint, original)
        manifest = json.loads((smoke / "run_manifest.json").read_text())
        if manifest["training_file_hashes"] != hashes:
            raise ValueError("Training implementation changed before load-only verification")
        c = original
        c.trainer.resume_mode = "resume_path"
        c.trainer.resume_from_path = str(checkpoint)
        c.trainer.require_complete_checkpoint = True
        c.trainer.exit_after_load = True
        c.algorithm.urcr_v3_focus.output_dir = str(run)
    validate_focus_config(c)
    # Existing V2 rolling checkpoint guard: 38 GiB covers one observed ~36 GiB full resume.
    checkpoint_bytes = 38 * 1024**3
    export_bytes = sum(p.stat().st_size for p in Path(c.actor_rollout_ref.model.path).glob("*.safetensors"))
    needed = max(38 * 1024**3, checkpoint_bytes)
    if phase == "formal":
        needed = checkpoint_bytes + export_bytes
    free = shutil.disk_usage(FAMILY.parent).free
    if phase != "resume_check" and free < needed:
        raise RuntimeError(f"A1 storage budget not met: free={free}, required={needed}")
    run.mkdir(parents=True)
    OmegaConf.save(c, run / "launch_config.yaml")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "flash-attn", "ray", "tensordict", "torchdata")}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "--", "EviSD-URCR"], cwd=REPO.parent)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", "EviSD-URCR"], cwd=REPO.parent, text=True).splitlines()
    for name in untracked:
        diff += subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", name], cwd=REPO.parent, capture_output=True).stdout
    (run / "implementation.diff").write_bytes(diff)
    manifest = {
        "run_family": "urcr_v3_a_visible_focus", "scientific_run": "A1", "phase": phase,
        "source_commit": head, "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "training_file_hashes": hashes, "base_model": c.actor_rollout_ref.model.path,
        "base_config_sha256": digest(Path(c.actor_rollout_ref.model.path) / "config.json"),
        "config_sha256": digest(run / "launch_config.yaml"),
        "prompt_template_sha256": hashlib.sha256(FOCUS_PREAMBLE.encode()).hexdigest(),
        "environments": versions,
        "train_data_sha256": digest(c.data.train_files), "test_data_sha256": digest(c.data.val_files),
        "upstreams": {"EviSD": "e72922f891f7e66d773eeb53fa84435f08f8e495", "SDAR": "f0461dd6fad4d828a05035f9f327d76616859912"},
        "upstream_licenses": {"EviSD": "no root LICENSE in local upstream snapshot; retained file headers", "SDAR": "Apache-2.0"},
    }
    write_json(run / "run_manifest.json", manifest)
    write_json(run / "method_contract.json", {**validate_focus_config(c), "student_input": "unchanged visible original token IDs", "teacher": "same pre-update FSDP actor; no grad", "teacher_target_cache": "CPU int32 top64 IDs, FP32 full-normalized logp and tail", "distributed_scale": "lambda * DP_world / original_trajectories; no accumulation divisor", "microbatch": args.actor_micro, "world_size": 4})
    print(json.dumps({"phase": phase, "run_dir": str(run), "versions": versions, "config_sha256": manifest["config_sha256"]}))


def replay(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from verl import DataProto
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from verl.trainer.ppo.urcr_text import extract_supporting_facts
    from verl.trainer.ppo.urcr_v3_focus import object_array

    c = config()
    tok = AutoTokenizer.from_pretrained(c.actor_rollout_ref.model.path, local_files_only=True)
    collector = TrajectoryCollector(c, tok, None)
    cfg = validate_focus_config(c)
    source = Path("/data2/kongmu/agentic-RL-runtime/urcr_v2_fixed_support_agam/formal_4gpu/segment_localmax0p01_resume_s150_010/turn_components_parts/step_000300.parquet")
    rows = pq.read_table(source, columns=["data_source", "question_uid", "traj_uid", "turn_step", "turn_context_text", "response_text", "action_type"]).to_pylist()
    rows = [r for r in rows if r["data_source"] == "hotpotqa" and r["turn_step"] > 0 and r["action_type"] in ("search", "answer")]
    needed = {int(r["question_uid"].split(":")[-1]) for r in rows}
    table = pq.read_table(c.data.train_files, columns=["extra_info", "metadata"])
    metadata = {}
    for i, extra in enumerate(table["extra_info"].to_pylist()):
        if extra["index"] in needed:
            metadata[extra["index"]] = table["metadata"][i].as_py()
    gen = DataProto.from_dict(tensors={}, non_tensors={"raw_prompt": object_array([[{"role": "user", "content": "unused"}]]), "data_source": object_array(["hotpotqa"])})
    buckets = {}
    for row in rows:
        original = collector.preprocess_single_sample(0, gen, {"text": [row["turn_context_text"]]})
        ids = original["input_ids"][original["attention_mask"].bool()].tolist()
        facts = extract_supporting_facts(metadata[int(row["question_uid"].split(":")[-1])])
        view = build_visible_focus(ids, [(f.title, f.sentence) for f in facts], tok, cfg, current_turn=int(row["turn_step"]))
        if view is None:
            continue
        match = "fact_visible" if any(e["match_type"] == "fact_visible" for e in view.excerpts) else "doc_only"
        key = (row["action_type"], match, len(ids) == 4096)
        if len(buckets.setdefault(key, [])) >= 4:
            continue
        response = tok.encode(row["response_text"], add_special_tokens=False)
        assert all(e["source_observation_turn"] < row["turn_step"] for e in view.excerpts)
        record = {**asdict(view), "question_uid": row["question_uid"], "traj_uid": row["traj_uid"], "turn_step": row["turn_step"], "action_type": row["action_type"], "response_token_ids": response, "provenance": "V2 retained context replay through current actual preprocessing; historical raw token IDs were NOT saved", "source": str(source), "left_truncated": len(ids) == 4096}
        buckets[key].append(record)
        if sum(map(len, buckets.values())) >= 32:
            break
    # Round-robin across observed types; deterministic and independent of rollout RNG.
    chosen = [items[i] for i in range(4) for _, items in sorted(buckets.items()) if len(items) > i][:16]
    if len(chosen) != 16:
        raise RuntimeError(f"Only {len(chosen)} legal retained-view replays found")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "replayed_views.jsonl").open("x") as handle:
        for item in chosen:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {"view_pairs": len(chosen), "action_types": dict(Counter(r["action_type"] for r in chosen)), "match_types": dict(Counter(e["match_type"] for r in chosen for e in r["excerpts"])), "left_truncated": sum(r["left_truncated"] for r in chosen), "source_sha256": digest(source), "historical_original_token_ids_available": False}
    write_json(out / "replay_summary.json", summary)
    print(json.dumps(summary))


def calibrate(args):
    smoke = FAMILY / args.smoke_name
    check = FAMILY / args.load_name
    if "global_step=2," not in (check / "train.log").read_text():
        raise RuntimeError("Successful actual S2 checkpoint load-only receipt is required")
    c = OmegaConf.load(smoke / "launch_config.yaml")
    verify_resume(smoke / "checkpoints/rolling/global_step_2", c)
    steps = [json.loads((smoke / "step_metrics" / f"step_{s:06d}.json").read_text()) for s in (1, 2)]
    ratios = []
    for m in steps:
        if m["evisd/teacher_scored_row_count"] != 0:
            raise RuntimeError("Legacy EviSD teacher must not score any rows in A1")
        for key in ("prompt_mismatch", "response_mismatch", "future_leak_count", "duplicate_target_count", "teacher_grad_nonzero_count", "parameter_nonfinite_count"):
            if m[f"v3_focus/{key}"] != 0:
                raise RuntimeError(f"Smoke contract failure: {key}")
        if m["v3_focus/selected_rows"] and m["v3_focus/unscaled_focus_logit_l2"] <= 0:
            raise RuntimeError("Selected focus targets produced zero gradient; debug before fallback")
        if m["v3_focus/pg_logit_l2"] > 0 and m["v3_focus/unscaled_focus_logit_l2"] > 0:
            ratios.append(m["v3_focus/unscaled_focus_over_pg_logit_l2"])
        if not all(math.isfinite(m[k]) for k in ("actor/pg_loss", "actor/grad_norm", "v3_focus/scaled_loss")):
            raise RuntimeError("Nonfinite smoke loss/gradient")
    if not sum(m["v3_focus/selected_rows"] for m in steps):
        raise RuntimeError("No targets in smoke; run the plan's known-legal small batch before calibration")
    manifest = json.loads((smoke / "run_manifest.json").read_text())
    view_count = 0
    for path in sorted((smoke / "focus_audit").glob("step_*.jsonl")):
        for line in path.read_text().splitlines():
            view = json.loads(line)
            start, length = view["insertion_start"], len(view["inserted_token_ids"])
            if view["teacher_prompt_token_ids"][:start] + view["teacher_prompt_token_ids"][start + length:] != view["student_prompt_token_ids"]:
                raise RuntimeError("Actual smoke focus view changed original prompt IDs")
            if any(e["source_observation_turn"] >= view["turn_step"] for e in view["excerpts"]):
                raise RuntimeError("Actual smoke focus view violates next-decision timing")
            view_count += 1
    if view_count < 16:
        raise RuntimeError("At least 16 actual smoke focus pairs are required")
    result = {**calibrate_focus_coefficient(ratios), "ratios": ratios,
              "verified_actual_view_pairs": view_count,
              "actor_micro": c.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
              "teacher_micro": c.algorithm.urcr_v3_focus.teacher_micro_batch_size,
              "training_file_hashes": manifest["training_file_hashes"], "smoke": str(smoke),
              "load_check": str(check), "raw_step_metrics": steps}
    write_json(FAMILY / "calibration.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("raw_step_metrics", "training_file_hashes")}))


def export_model(args):
    run = Path(args.run_dir).resolve()
    if run != FAMILY / "formal":
        raise ValueError("Only the formal A1 S200/S300 exports are in scope")
    step = args.step
    c = OmegaConf.load(run / "launch_config.yaml")
    checkpoint = Path(c.trainer.default_local_dir) / f"global_step_{step}"
    verify_resume(checkpoint, c)
    output = run / "exports" / f"S{step}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an A1 export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    for action, extra in (("merge", ["--target_dir", str(output)]), ("test", ["--test_hf_dir", str(output)])):
        subprocess.run([sys.executable, str(REPO / "scripts/model_merger.py"), action, "--backend", "fsdp", "--local_dir", str(checkpoint / "actor"), *extra], check=True, cwd=REPO)
    write_json(output / "export_manifest.json", {
        "method": "urcr_v3_a_visible_focus", "scientific_run": "A1", "global_step": step,
        "source_checkpoint": str(checkpoint), "source_config_sha256": digest(checkpoint / "resolved_config.json"),
        "precision": "bfloat16", "model_merger_fsdp_test": "passed",
        "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    })


def direct_success_rates(log, datasets):
    # pprint wraps the *string representation* of the metrics across quoted lines.
    # Strip only Ray's prefix and joining quotes; never use rounded console metrics.
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log)
    clean = re.sub(r"(?m)^\(EviSDTaskRunner pid=\d+\)\s*", "", clean)
    clean = re.sub(r'"\s*\n\s*"', "", clean)
    values = {}
    for dataset in datasets:
        hits = re.findall(rf"['\"]val/{dataset}_success_rate['\"]:\s*([0-9.eE+-]+)", clean)
        if not hits:
            raise RuntimeError(f"Full-precision directly emitted success_rate missing for {dataset}")
        values[dataset] = float(hits[-1])
    return values


def evaluate_summary(args):
    import pyarrow.parquet as pq
    from scripts.score_raw_qa_em_f1 import score

    output = Path(args.run_dir) / "evaluations" / f"S{args.step}"
    expected = {"nq": 3610, "triviaqa": 11313, "popqa": 14267, "hotpotqa": 7405, "2wikimultihopqa": 12576, "musique": 2417, "bamboogle": 125}
    log = (output / "eval.log").read_text()
    direct = direct_success_rates(log, expected)
    raw = score(output)
    if raw["duplicate_dataset_question_rows"] or {d: v["count"] for d, v in raw["per_dataset"].items()} != expected:
        raise RuntimeError("Seven-dataset raw outputs contain duplicates, omissions, or unexpected counts")
    call_sums, token_sums = Counter(), Counter()
    for path in sorted((output / "raw_trajectories").glob("batch_*.parquet")):
        for row in pq.read_table(path, columns=["dataset", "tool_call_count", "response_token_count"]).to_pylist():
            call_sums[row["dataset"]] += row["tool_call_count"]
            token_sums[row["dataset"]] += row["response_token_count"]
    costs = {d: {"calls_per_question": call_sums[d] / n, "response_tokens_per_question": token_sums[d] / n} for d, n in expected.items()}
    write_json(output / "seven_dataset_results.json", {
        "global_step": args.step, "world_size": 4, "temperature": 0, "trajectories_per_question": 1,
        "direct_success_rate": direct, "direct_unweighted_average": sum(direct.values()) / 7,
        "raw_standard_em_f1": raw,
        "costs_per_dataset": costs,
        "cost_macro": {key: sum(v[key] for v in costs.values()) / 7 for key in next(iter(costs.values()))},
        "cost_micro": {"calls_per_question": sum(call_sums.values()) / 51713, "response_tokens_per_question": sum(token_sums.values()) / 51713},
    })
    lines = ["# A1 S%d evaluation" % args.step, "", "Canonical upstream direct success_rate; raw standard EM/F1 are separate.", "", "| Dataset | Direct success | Raw EM | Raw F1 |", "|---|---:|---:|---:|"]
    for d in expected:
        r = raw["per_dataset"][d]
        lines.append(f"| {d} | {100 * direct[d]:.2f} | {100 * r['standard_em']:.2f} | {100 * r['standard_f1']:.2f} |")
    macro, micro = raw["unweighted_dataset_average"], raw["micro_average"]
    lines.extend([f"| Macro | {100 * sum(direct.values()) / 7:.2f} | {100 * macro['standard_em']:.2f} | {100 * macro['standard_f1']:.2f} |", "", f"Raw micro EM/F1: {100 * micro['standard_em']:.2f}/{100 * micro['standard_f1']:.2f}; n=51713, no duplicate dataset/question IDs."])
    lines += ["", "| Dataset | Calls/question | Response tokens/question |", "|---|---:|---:|"]
    lines += [f"| {d} | {costs[d]['calls_per_question']:.3f} | {costs[d]['response_tokens_per_question']:.2f} |" for d in expected]
    with (output / "results.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--phase", choices=["smoke", "formal", "resume_check"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--actor-micro", type=int, default=16)
    p.add_argument("--teacher-micro", type=int, default=32)
    p.add_argument("--smoke-name", default="smoke_001")
    p.set_defaults(func=prepare)
    p = sub.add_parser("replay")
    p.add_argument("--output", required=True)
    p.set_defaults(func=replay)
    p = sub.add_parser("calibrate")
    p.add_argument("--smoke-name", default="smoke_001")
    p.add_argument("--load-name", default="resume_check_001")
    p.set_defaults(func=calibrate)
    for name, func in (("export", export_model), ("evaluate_summary", evaluate_summary)):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--step", type=int, choices=[200, 300], required=True)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
