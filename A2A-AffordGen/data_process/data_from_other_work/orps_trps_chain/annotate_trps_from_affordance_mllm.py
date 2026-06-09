import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODEL_PATH = "data/robot/Qwen3-VL-32B-Instruct"


TRPS_TEMPLATE = (
    "You are writing TRPS task instructions. "
    "Write one short natural imperative that describes the user goal without naming the object part directly. "
    "The instruction must clearly reflect the action verb phrase and must not collapse into a generic sentence. "
    "The model should infer which part to localize from the task. "
    "For style reference, good examples are 'Flush the toilet' and 'Turn on the faucet'. "
    "Bad examples are 'Press the toilet handle', 'Lift the faucet handle', and generic outputs like 'Use the toilet'. "
    "Now write one instruction for object={OBJECT}, action={ACTION}, affordance={AFFORDANCE}, and hidden target part={PART}. "
    "The instruction should be centered on the given action phrase. "
    "If action is 'light up', write something like 'Light up the lamp'. "
    "If action is 'sit on', write something like 'Sit on the chair'. "
    "If action is 'drink from', write something like 'Drink from the cup'. "
    "Do not mention the part term or close lexical variants. "
    "Do not use vague verbs like 'use', 'interact with', or 'handle' unless the action itself is exactly that verb. "
    "Keep it concise and natural, avoid long explanations. "
    "Return only JSON wrapped by <answer></answer>: <answer>{\"instruction\":\"\"}</answer>"
)


AFFORDANCE_TASK_FALLBACK: Dict[str, str] = {
    "Open": "Open the {obj}",
    "Press": "Turn on the {obj}",
    "Push": "Move the {obj} forward",
    "Pull": "Pull the {obj}",
    "Pour": "Pour with the {obj}",
    "Contain": "Put something in the {obj}",
    "Display": "Check the {obj}",
    "Listen": "Listen with the {obj}",
    "Wear": "Wear the {obj}",
    "Grasp": "Pick up the {obj}",
    "Lift": "Lift the {obj}",
    "Move": "Move the {obj}",
    "Sit": "Sit on the {obj}",
    "Support": "Use the {obj} for support",
    "Lay": "Place something on the {obj}",
    "Wrap": "Wrap with the {obj}",
    "Cut": "Cut with the {obj}",
    "Stab": "Pierce with the {obj}",
}


def parse_args():
    p = argparse.ArgumentParser(description="Annotate TRPS instructions from affordance+part using MLLM")
    p.add_argument("--gpu", type=str, default="0,1", help="CUDA_VISIBLE_DEVICES")
    p.add_argument(
        "--input_glob",
        type=str,
        default="data/test/top10k_quality_objectset_balanced_v2_part*_mllm.jsonl",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        default="data/test/trps_affordance_annotated.jsonl",
    )
    p.add_argument(
        "--processed_json",
        type=str,
        default="data/test/trps_affordance_annotated_processed.json",
    )
    p.add_argument(
        "--summary_json",
        type=str,
        default="data/test/trps_affordance_annotated_summary.json",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_items", type=int, default=0, help="0 means all")
    p.add_argument("--skip_whole_object", action="store_true", help="Skip part=='whole_object'")
    return p.parse_args()


def load_json_list(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)
        return x if isinstance(x, list) else []
    except Exception:
        return []


def dump_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, rows: List[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def clean_str(x) -> str:
    if not isinstance(x, str):
        return ""
    return x.strip()


def normalize(s: str) -> str:
    s = clean_str(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def split_parts(parts) -> List[str]:
    if isinstance(parts, str):
        parts = [parts]
    if not isinstance(parts, list):
        return []
    out = []
    for p in parts:
        p = clean_str(p)
        if p:
            out.append(p)
    return out


def make_uid(item: dict) -> str:
    payload = json.dumps(
        {
            "image_path": item.get("image_path", ""),
            "object": item.get("object", ""),
            "part": item.get("part", ""),
            "affordance": item.get("affordance", ""),
            "source": item.get("source", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def iter_base_items(files: List[Path], skip_whole_object: bool, max_items: int = 0) -> Iterable[dict]:
    n = 0
    for fp in files:
        for ln, rec in enumerate(iter_jsonl(fp), 1):
            flat_obj = clean_str(rec.get("object"))
            flat_part = clean_str(rec.get("part"))
            flat_aff = clean_str(rec.get("affordance"))
            flat_image_path = clean_str(rec.get("image_path"))
            if flat_obj and flat_part and flat_aff and flat_image_path:
                if skip_whole_object and normalize(flat_part) == "whole_object":
                    continue
                item = {
                    "source": clean_str(rec.get("source")) or f"{fp.name}:{ln}",
                    "image_path": flat_image_path,
                    "object": flat_obj,
                    "part": flat_part,
                    "affordance": flat_aff,
                    "bucket_id": rec.get("bucket_id"),
                    "action": clean_str(rec.get("action")),
                }
                yield item
                n += 1
                if max_items > 0 and n >= max_items:
                    return
                continue

            if not rec.get("ok", False):
                continue
            result = rec.get("result", {})
            if not isinstance(result, dict):
                continue
            obj = clean_str(rec.get("object"))
            image_path = clean_str(rec.get("image_path"))

            selected = result.get("selected_affordances", [])
            if not isinstance(selected, list):
                selected = []

            if not selected:
                all_j = result.get("all_judgements", [])
                if isinstance(all_j, list):
                    for x in all_j:
                        if isinstance(x, dict) and bool(x.get("exists", False)):
                            selected.append(
                                {
                                    "affordance": x.get("affordance", ""),
                                    "parts": x.get("parts", []),
                                    "exists": True,
                                }
                            )

            for ai, aff_item in enumerate(selected):
                if not isinstance(aff_item, dict):
                    continue
                if not bool(aff_item.get("exists", True)):
                    continue
                aff = clean_str(aff_item.get("affordance"))
                if not obj or not aff:
                    continue
                for pi, part in enumerate(split_parts(aff_item.get("parts", []))):
                    if skip_whole_object and normalize(part) == "whole_object":
                        continue
                    item = {
                        "source": f"{fp.name}:{ln}:a{ai}:p{pi}",
                        "image_path": image_path,
                        "object": obj,
                        "part": part,
                        "affordance": aff,
                        "bucket_id": rec.get("bucket_id"),
                        "action": clean_str(rec.get("action")),
                    }
                    yield item
                    n += 1
                    if max_items > 0 and n >= max_items:
                        return


def build_prompt(item: dict) -> str:
    return (
        TRPS_TEMPLATE.replace("{OBJECT}", item["object"])
        .replace("{ACTION}", item.get("action", ""))
        .replace("{PART}", item["part"])
        .replace("{AFFORDANCE}", item["affordance"])
    )


def extract_answer_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    if "<answer>" in t:
        try:
            inner = t.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
            return json.loads(inner)
        except Exception:
            pass
    try:
        l = t.find("{")
        r = t.rfind("}")
        if l >= 0 and r > l:
            return json.loads(t[l : r + 1])
    except Exception:
        return None
    return None


def sanitize_text(s: str) -> str:
    s = clean_str(s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" ,.;:")
    return s


def fallback_trps_instruction(obj: str, aff: str, action: str = "") -> str:
    action = clean_str(action)
    if action:
        action = re.sub(r"\s+", " ", action).strip()
        if action:
            return f"{action[:1].upper() + action[1:]} the {obj}"
    t = AFFORDANCE_TASK_FALLBACK.get(aff, "Use the {obj}")
    return t.format(obj=obj)


def contains_part_tokens(part: str, text: str) -> bool:
    p = normalize(part)
    t = normalize(text)
    if not p:
        return False
    if p in t:
        return True
    for tok in p.split():
        if len(tok) >= 3 and tok in t:
            return True
    return False


def starts_with_generic_use(text: str, action: str) -> bool:
    t = normalize(text)
    a = normalize(action)
    if not t.startswith("use the "):
        return False
    return a != "use"


def validate_trps_instruction(obj: str, part: str, text: str, action: str = "") -> Optional[str]:
    t = normalize(text)
    if len(t.split()) < 2:
        return "too_short"
    if len(t.split()) > 16:
        return "too_long"
    if contains_part_tokens(part, text):
        return "contains_part"
    if " of " in f" {t} ":
        return "contains_of_pattern"
    if normalize(obj) not in t:
        return "missing_object"
    if starts_with_generic_use(text, action):
        return "generic_use"
    return None


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_MM_LIMIT_PER_PROMPT_VIDEO"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    files = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not files:
        p = Path(args.input_glob)
        if p.exists():
            files = [p]
    if not files:
        raise FileNotFoundError(f"no input matched: {args.input_glob}")

    processed_set = set(load_json_list(args.processed_json))
    all_items = []
    for item in iter_base_items(files, skip_whole_object=args.skip_whole_object, max_items=args.max_items):
        uid = make_uid(item)
        if uid in processed_set:
            continue
        item["id"] = uid
        all_items.append(item)

    print(f"[INFO] matched_files={len(files)} todo_items={len(all_items)} already_processed={len(processed_set)}")
    if not all_items:
        summary = {
            "matched_files": len(files),
            "todo_items": 0,
            "already_processed": len(processed_set),
            "message": "no new items",
        }
        dump_json(summary, args.summary_json)
        return

    _ = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=max(1, len(args.gpu.split(","))),
        mm_encoder_tp_mode="data",
        gpu_memory_utilization=0.85,
        max_model_len=32768,
        enforce_eager=True,
        trust_remote_code=True,
        enable_expert_parallel=False,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=384,
        top_k=-1,
        stop_token_ids=[],
    )

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.processed_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)

    stat = {"ok": 0, "parse_fail": 0, "repaired": 0}

    for start in range(0, len(all_items), args.batch_size):
        batch = all_items[start : start + args.batch_size]
        prompts = [build_prompt(it) for it in batch]
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        rows = []
        for it, out in zip(batch, outputs):
            uid = it["id"]
            raw = out.outputs[0].text if out.outputs else ""
            ans = extract_answer_json(raw)
            processed_set.add(uid)

            if not isinstance(ans, dict):
                stat["parse_fail"] += 1
                rows.append(
                    {
                        "id": uid,
                        "ok": False,
                        "error_code": 1,
                        "raw_text": raw[:2000],
                        "source": it["source"],
                    }
                )
                continue

            ins = sanitize_text(clean_str(ans.get("instruction")))
            err = validate_trps_instruction(it["object"], it["part"], ins, it.get("action", ""))
            repaired = False
            if err is not None:
                ins = fallback_trps_instruction(it["object"], it["affordance"], it.get("action", ""))
                err2 = validate_trps_instruction(it["object"], it["part"], ins, it.get("action", ""))
                if err2 is not None:
                    rows.append(
                        {
                            "id": uid,
                            "ok": False,
                            "error_code": 2,
                            "error_msg": f"{err}|fallback:{err2}",
                            "raw_text": raw[:2000],
                            "source": it["source"],
                        }
                    )
                    continue
                repaired = True
                stat["repaired"] += 1

            stat["ok"] += 1
            rows.append(
                {
                    "id": uid,
                    "ok": True,
                    "task_type": "TRPS",
                    "image_path": it["image_path"],
                    "object": it["object"],
                    "part": it["part"],
                    "affordance": it["affordance"],
                    "bucket_id": it.get("bucket_id"),
                    "action": it.get("action", ""),
                    "instruction": ins,
                    "source": it["source"],
                    "repaired": repaired,
                }
            )

        append_jsonl(args.output_jsonl, rows)
        dump_json(sorted(processed_set), args.processed_json)
        print(f"[INFO] saved {start}-{start + len(batch)}")

    summary = {
        "input_glob": args.input_glob,
        "matched_files": [str(p) for p in files],
        "output_jsonl": args.output_jsonl,
        "processed_json": args.processed_json,
        "summary_json": args.summary_json,
        "todo_items": len(all_items),
        "stats": stat,
        "config": {
            "gpu": args.gpu,
            "batch_size": args.batch_size,
            "max_items": args.max_items,
            "skip_whole_object": args.skip_whole_object,
        },
    }
    dump_json(summary, args.summary_json)
    print(f"[DONE] output: {args.output_jsonl}")
    print(f"[DONE] processed: {args.processed_json}")
    print(f"[DONE] summary: {args.summary_json}")


if __name__ == "__main__":
    main()
