import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODEL_PATH = "data/robot/Qwen3-VL-32B-Instruct"
BATCH_SIZE = 256
GPU_MEMORY_UTIL = 0.85

# Keep these aligned with infer_NP.py
SAMPLING_TEMPERATURE = 0.0
SAMPLING_TOP_K = -1
SAMPLING_MAX_TOKENS = 3200

ROUND3_TAG = "count_object_only_v2"

COUNT_JUDGE_TEMPLATE_V2 = (
    "You are a careful and consistent visual object counter. "
    "Your task is to estimate how many visible real-world instances of one specific object class appear in one image.\n\n"
    "Injected fields:\n"
    "- object_name: \"{OBJECT}\"\n"
    "\n"
    "Counting rules:\n"
    "1) Count only distinct visible instances of object_name.\n"
    "2) Count only real objects physically present in the scene.\n"
    "3) Do NOT count reflections, shadows, posters, paintings, printed photos, screen content, drawings, logos, icons, or text mentions.\n"
    "4) Do NOT count object parts as separate instances.\n"
    "5) Count a partially visible or image-boundary-cut object only if there is enough visible evidence to identify it as one distinct instance.\n"
    "6) Ignore tiny or heavily occluded regions that cannot be reliably identified.\n"
    "7) Count each instance once, even if multiple visible regions belong to the same object.\n"
    "8) Do not assume any expected number. Base the answer only on visible evidence in the image.\n"
    "9) When uncertain between two counts, choose the most likely count supported by visible evidence, not the smallest possible count.\n"
    "10) The output estimated_count must be exactly one of: 0, 1, 2, 3, 4, 5, more_than_5.\n"
    "11) Use confidence='high' when instances are clear and well separated; "
    "use 'medium' when there is mild occlusion, small size, or limited ambiguity; "
    "use 'low' when heavy occlusion, blur, dense overlap, or class ambiguity makes counting unreliable.\n"
    "12) reason must be short English text with no more than 25 words, and should briefly justify the chosen count.\n\n"
    "Return ONLY one JSON object wrapped by <answer> and </answer>:\n"
    "<answer>{\"object\":\"{OBJECT}\",\"estimated_count\":\"0\","
    "\"confidence\":\"low\",\"reason\":\"\"}</answer>"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="MLLM judge: count how many instances of the target object are visible in image"
    )
    p.add_argument("--gpu", type=str, required=True, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 5,6")
    p.add_argument("--startidx", type=int, required=True, help="start patch id")
    p.add_argument("--endidx", type=int, required=True, help="end patch id (inclusive)")
    p.add_argument(
        "--input_root",
        type=str,
        default="data/keep",
        help="Root with bucket folders 1..5",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="data/count_judge",
    )
    p.add_argument(
        "--input_name_fmt",
        type=str,
        default="object365_patch{patch}_round1_aff_select_merged_v1_round2_affordance_verify_v1.jsonl",
    )
    p.add_argument(
        "--buckets",
        type=str,
        default="1,2,3,4,5",
        help="comma separated bucket ids",
    )
    return p.parse_args()


def load_json_list(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if not txt:
                return []
            x = json.loads(txt)
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
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                yield rec


def safe_check_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_single_input_qwen(image_path: str, processor, question: str):
    if not safe_check_image(image_path):
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]

    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
    except Exception:
        return None

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


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


def clean_str(x) -> str:
    return str(x).strip() if x is not None else ""


def extract_object_and_source_count(rec: dict, bucket_id: int) -> Tuple[str, int]:
    obj = clean_str(rec.get("keep_target_object", ""))
    if not obj:
        res = rec.get("result", {}) if isinstance(rec.get("result", {}), dict) else {}
        evals = res.get("evaluations", []) if isinstance(res.get("evaluations", []), list) else []
        for e in evals:
            if isinstance(e, dict) and clean_str(e.get("object", "")):
                obj = clean_str(e.get("object", ""))
                break

    n = rec.get("keep_target_bbox_count", None)
    if n is None:
        n = bucket_id
    try:
        n = int(n)
    except Exception:
        n = int(bucket_id)
    return obj, n


def make_uid(image_path: str, obj: str) -> str:
    payload = json.dumps(
        {
            "image_path": image_path,
            "object": obj,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def build_question(obj: str) -> str:
    return COUNT_JUDGE_TEMPLATE_V2.replace("{OBJECT}", obj)


def validate_count_answer(ans: dict, obj: str) -> Tuple[bool, Optional[str]]:
    if not isinstance(ans, dict):
        return False, "answer_not_dict"

    est = clean_str(ans.get("estimated_count", "")).lower()
    conf = clean_str(ans.get("confidence", "")).lower()
    reason = clean_str(ans.get("reason", ""))

    allowed_est = {"0", "1", "2", "3", "4", "5", "more_than_5"}
    if est not in allowed_est:
        return False, "estimated_count_invalid"
    if conf not in {"low", "medium", "high"}:
        return False, "confidence_invalid"
    if not reason:
        return False, "reason_empty"

    return True, None


def process_patch_bucket(
    patch_id: int,
    bucket_id: int,
    in_path: Path,
    out_root: Path,
    llm: LLM,
    processor,
    sampling_params: SamplingParams,
):
    if not in_path.exists():
        print(f"[WARN] missing input: {in_path}")
        return

    out_dir = out_root / str(bucket_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / f"object365_patch{patch_id}_{ROUND3_TAG}.jsonl"
    processed_json = out_dir / f"object365_patch{patch_id}_{ROUND3_TAG}_processed.json"

    processed_set = set(load_json_list(str(processed_json)))

    todo = []
    local_seen = set()
    for rec in iter_jsonl(in_path):
        image_path = clean_str(rec.get("image_path", ""))
        if not image_path:
            continue

        obj, source_n = extract_object_and_source_count(rec, bucket_id=bucket_id)
        if not obj:
            continue

        uid = make_uid(image_path=image_path, obj=obj)
        if uid in processed_set or uid in local_seen:
            continue

        local_seen.add(uid)
        todo.append(
            {
                "id": uid,
                "image_path": image_path,
                "object": obj,
                "source_count": int(source_n),
                "source_file": str(in_path),
            }
        )

    print(
        f"[INFO] patch={patch_id} bucket={bucket_id} todo={len(todo)} processed={len(processed_set)}"
    )
    if not todo:
        return

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start : start + BATCH_SIZE]

        batch_inputs = []
        valid_items = []
        for it in batch:
            q = build_question(it["object"])
            mm_item = prepare_single_input_qwen(it["image_path"], processor, q)
            if mm_item is None:
                processed_set.add(it["id"])
                continue
            batch_inputs.append(mm_item)
            valid_items.append(it)

        if not batch_inputs:
            dump_json(sorted(processed_set), str(processed_json))
            continue

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        rows = []
        for it, out in zip(valid_items, outputs):
            raw_text = out.outputs[0].text if out.outputs else ""
            ans = extract_answer_json(raw_text)
            processed_set.add(it["id"])

            if not isinstance(ans, dict):
                rows.append(
                    {
                        "id": it["id"],
                        "ok": False,
                        "error_code": 1,
                        "error_msg": "parse_fail",
                        "image_path": it["image_path"],
                        "object": it["object"],
                        "source_count": it["source_count"],
                        "raw_text": raw_text[:4000],
                        "source_file": it["source_file"],
                    }
                )
                continue

            valid, err = validate_count_answer(ans, it["object"])
            if not valid:
                rows.append(
                    {
                        "id": it["id"],
                        "ok": False,
                        "error_code": 2,
                        "error_msg": err,
                        "image_path": it["image_path"],
                        "object": it["object"],
                        "source_count": it["source_count"],
                        "raw_text": raw_text[:4000],
                        "answer": ans,
                        "source_file": it["source_file"],
                    }
                )
                continue

            rows.append(
                {
                    "id": it["id"],
                    "ok": True,
                    "error_code": 0,
                    "image_path": it["image_path"],
                    "object": it["object"],
                    "source_count": it["source_count"],
                    "result": {
                        "object": clean_str(ans.get("object", it["object"])),
                        "estimated_count": clean_str(ans.get("estimated_count", "")).lower(),
                        "confidence": clean_str(ans.get("confidence", "")).lower(),
                        "reason": clean_str(ans.get("reason", "")),
                    },
                    "source_file": it["source_file"],
                }
            )

        append_jsonl(str(out_jsonl), rows)
        dump_json(sorted(processed_set), str(processed_json))

        print(
            f"[INFO] patch={patch_id} bucket={bucket_id} saved {start}-{start + len(batch)} processed={len(processed_set)}"
        )


def main():
    args = parse_args()
    assert args.startidx <= args.endidx, "startidx must be <= endidx"

    bucket_ids = [int(x.strip()) for x in args.buckets.split(",") if x.strip()]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_MM_LIMIT_PER_PROMPT_VIDEO"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch  # noqa

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"patch range: {args.startidx} -> {args.endidx}")
    print(f"buckets: {bucket_ids}")
    print(f"ROUND3_TAG={ROUND3_TAG}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    tp = torch.cuda.device_count()
    if tp <= 0:
        raise RuntimeError("No visible CUDA devices. Check CUDA_VISIBLE_DEVICES.")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp,
        mm_encoder_tp_mode="data",
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=32768,
        enforce_eager=True,
        trust_remote_code=True,
        enable_expert_parallel=False,
    )

    # Keep the same sampling values as infer_NP.py
    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        max_tokens=SAMPLING_MAX_TOKENS,
        top_k=SAMPLING_TOP_K,
        stop_token_ids=[],
    )

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    for patch_id in range(args.startidx, args.endidx + 1):
        in_name = args.input_name_fmt.format(patch=patch_id)
        for bucket_id in bucket_ids:
            in_path = input_root / str(bucket_id) / in_name
            process_patch_bucket(
                patch_id=patch_id,
                bucket_id=bucket_id,
                in_path=in_path,
                out_root=output_root,
                llm=llm,
                processor=processor,
                sampling_params=sampling_params,
            )


if __name__ == "__main__":
    main()
