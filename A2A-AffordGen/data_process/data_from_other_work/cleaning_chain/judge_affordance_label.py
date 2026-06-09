import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


# =========================
# User-configurable paths
# =========================
OUT_ROOT = "data/data00"
MODEL_PATH = "data/robot/Qwen3-VL-32B-Instruct"
MANIFEST_JSONL = "data/data00/obj365_yes_ann_train/manifest.jsonl"

MAX_ANNOS_IN_PROMPT = 12
BATCH_SIZE = 256
GPU_MEMORY_UTIL = 0.85


AFFORDANCES = [
    "Push", "Drink_with", "Take_photo", "Brush_with", "Swing", "Talk_on", "Beat", "Ride", "Wash", "Pour",
    "Open", "Cut", "Eat", "Look_out", "Lie_on", "Stir", "Boxing", "Hit", "Pick_up", "Cut_with",
    "Throw", "Catch", "Text_on", "Sit_on", "Pack", "Drag", "Hold", "Write", "Kick", "Peel",
    "Lift", "Stick", "Type_on", "Sip", "Carry", "Jump",
]
AFFORDANCE_ID_TEXT = "\n".join([f"- {i}: {a}" for i, a in enumerate(AFFORDANCES)])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen-32B Object365 Round-2 (manifest.jsonl + numeric-only output + resumable)"
    )
    parser.add_argument("--gpu", type=str, required=True, help="GPU ids, e.g. 0 or 5,6")
    parser.add_argument("--startidx", type=int, required=True, help="start patch id")
    parser.add_argument("--endidx", type=int, required=True, help="end patch id (inclusive)")
    return parser.parse_args()


def load_json_list(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: str, records: List[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
            inside = t.split("<answer>", 1)[1]
            inside = inside.split("</answer>", 1)[0].strip()
            return json.loads(inside)
        except Exception:
            pass

    try:
        l = t.find("{")
        r = t.rfind("}")
        if l >= 0 and r > l:
            return json.loads(t[l:r + 1])
    except Exception:
        return None

    return None


def load_manifest_index(manifest_jsonl: str) -> Dict[str, List[dict]]:
    if not os.path.exists(manifest_jsonl):
        raise FileNotFoundError(f"manifest not found: {manifest_jsonl}")

    idx: Dict[str, List[dict]] = {}
    bad = 0
    total = 0

    with open(manifest_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
                ip = rec.get("image_path", "")
                labels = rec.get("labels", [])
                if not ip:
                    bad += 1
                    continue
                if not isinstance(labels, list):
                    labels = []
                labels = sorted(labels, key=lambda x: float(x.get("area", 0.0)), reverse=True)
                idx[ip] = labels
            except Exception:
                bad += 1

    print(f"[INFO] manifest loaded: lines={total}, indexed={len(idx)}, bad_lines={bad}")
    return idx


# =========================
# PROMPT injection helpers (CHANGED)
# =========================
def _sanitize_name(name: str) -> str:
    # Keep it minimal; do not leak extra info; ensure consistent "object_number" tokens.
    # Replace whitespace with underscore, strip.
    if not isinstance(name, str) or not name:
        return "object"
    return "_".join(name.strip().split())


def format_annos_text(labels: List[dict], topk: int) -> Dict[str, str]:
    """
    Return injection strings:
      - N_INSTANCES: total number of labeled instances in this image (len(labels))
      - LABEL_NAMES: unique object names from top-K (by area) labels, one per line
      - LABEL_COUNTS: counts of object names within top-K, one per line: "name: count"
    """
    n_instances = len(labels) if isinstance(labels, list) else 0
    if not labels:
        return {
            "N_INSTANCES": str(n_instances),
            "LABEL_NAMES": "(none)",
            "LABEL_COUNTS": "(none)",
        }

    top = labels[:topk] if topk > 0 else labels

    # unique names (preserve order)
    seen = set()
    uniq = []
    for a in top:
        nm = str(a.get("name", "")).strip()
        if not nm:
            continue
        if nm not in seen:
            uniq.append(nm)
            seen.add(nm)

    # counts in top-K
    from collections import Counter
    cnt = Counter([str(a.get("name", "")).strip() for a in top if str(a.get("name", "")).strip()])

    label_names_text = "\n".join(uniq) if uniq else "(none)"
    label_counts_text = "\n".join([f"{k}: {v}" for k, v in cnt.most_common()]) if cnt else "(none)"

    return {
        "N_INSTANCES": str(n_instances),
        "LABEL_NAMES": label_names_text,
        "LABEL_COUNTS": label_counts_text,
    }

# =========================
# Injected-label affordance prompt (CHANGED)
# =========================
ROUND2_TEMPLATE = (
    "You are an AI vision assistant capable of analyzing a single image. You will receive an image, along with the object detection label information for that image. "
    "The label information only contains object category names; it does not contain bounding box coordinates, nor does it contain object part annotations.\n\n"
    "This image has a total of {N_INSTANCES} object instances annotated.\n\n"
    "Below are the candidate object category names selected from the annotations (after taking the top K instances by box area from largest to smallest, then deduplicating), one name per line. "
    "These names are the only set of objects you are allowed to reference: you must use them verbatim. Adding new names, rewriting names, or outputting any object name that does not appear in the list is forbidden.\n"
    "\"\"\"\n"
    "{LABEL_NAMES}\n"
    "\"\"\"\n\n"
    "The instance counts for objects of the same category are listed below (also based on the top K instances). This count is only used to hint that multiple objects of the same category may exist, but you still must not invent numbering, "
    "nor may you output forms such as cup_1 or cup_2.\n"
    "\"\"\"\n"
    "{LABEL_COUNTS}\n"
    "\"\"\"\n\n"
    "Your task is: from the set of object names above, select the object categories most suitable to serve as affordance data samples, and for each selected object give one most appropriate affordance, "
    "as well as the names of the key parts (parts) that this affordance depends on for that object.\n\n"
    "You may select at most 3 objects, and you may select fewer than 3. If there is no suitable object, output an empty array.\n"
    "You must verify against the image content whether these objects are truly clearly visible and suitable to serve as the main subject sample. If a name is in the list but is not clear in the image, or is obviously unsuitable as a main subject, then do not select it.\n\n"
    "Selecting person or animals is forbidden.\n"
    "Selecting environmental structures or natural terrain is forbidden, for example floor, wall, stairs, pillar, rock, snow, ground, etc.\n\n"
    "You may only select one affordance from the affordance word list below as output, it must match verbatim, and outputting any affordance outside the word list is forbidden:\n"
    "{AFFORDANCE_LIST}\n\n"
    "The output requirements for parts are as follows:\n"
    "parts must be a list of English strings, with no length restriction.\n"
    "parts is used to describe the key parts that this affordance depends on. Prefer outputting part names directly related to this affordance, for example handle, blade, lid, cap, spout, button, screen, keyboard, wheel, seat, base, trigger, etc.\n"
    "If the affordance depends on the object as a whole rather than a specific part, you may output only a special marker \"whole_object\" to indicate dependence on the object itself.\n"
    "You must use \"whole_object\" cautiously: it is only allowed when the affordance, by common sense, truly depends mainly on the object as a whole, rather than on some obvious key part.\n\n"
    "Output format requirements:\n"
    "You must output only one JSON, and it must be wrapped with <answer> and </answer>. Outputting any other content besides the JSON is not allowed.\n"
    "All strings in the JSON must be in English.\n"
    "object must come strictly from the object name list above, matching verbatim.\n"
    "affordance must come strictly from the affordance word list above, matching verbatim.\n\n"
    "<answer>{\n"
    "  \"selected\": [\n"
    "    {\n"
    "      \"object\": \"\",\n"
    "      \"affordance\": \"\",\n"
    "      \"parts\": [\"whole_object\"]\n"
    "    }\n"
    "  ]\n"
    "}</answer>\n"
)



def build_round2_question(labels: List[dict]) -> str:
    inj = format_annos_text(labels, MAX_ANNOS_IN_PROMPT)
    affordance_list = ", ".join(AFFORDANCES)

    q = ROUND2_TEMPLATE
    q = q.replace("{N_INSTANCES}", inj["N_INSTANCES"])
    q = q.replace("{LABEL_NAMES}", inj["LABEL_NAMES"])
    q = q.replace("{LABEL_COUNTS}", inj["LABEL_COUNTS"])
    q = q.replace("{AFFORDANCE_LIST}", affordance_list)
    return q


def process_one_patch_round2(
    X: int,
    llm: LLM,
    processor,
    sampling_params: SamplingParams,
    yes_json: str,
    out_jsonl: str,
    processed_json: str,
    manifest_index: Dict[str, List[dict]],
    batch_size: int,
):
    yes_list = load_json_list(yes_json)
    if not isinstance(yes_list, list) or len(yes_list) == 0:
        print(f"[patch{X}] empty yes_json: {yes_json}")
        return

    processed_set = set(load_json_list(processed_json))
    remaining = [p for p in yes_list if p not in processed_set]
    print(f"[patch{X}] yes={len(yes_list)} remaining={len(remaining)}")

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(remaining), batch_size):
        batch_paths = remaining[start:start + batch_size]

        batch_inputs = []
        valid_paths = []
        meta_for_paths = {}

        for img_path in batch_paths:
            if not safe_check_image(img_path):
                processed_set.add(img_path)
                continue

            labels = manifest_index.get(img_path, [])
            question = build_round2_question(labels)

            item = prepare_single_input_qwen(img_path, processor, question)
            if item is None:
                processed_set.add(img_path)
                continue

            batch_inputs.append(item)
            valid_paths.append(img_path)
            meta_for_paths[img_path] = {"labels_count": len(labels)}

        if not batch_inputs:
            dump_json(sorted(processed_set), processed_json)
            continue

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        to_write = []
        for img_path, output in zip(valid_paths, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            ans = extract_answer_json(raw_text)

            processed_set.add(img_path)

            if ans is None:
                to_write.append({
                    "image_path": img_path,
                    "ok": False,
                    "error_code": 1,          # 1=parse_failed
                    "raw_text": raw_text[:4000],
                    "meta": meta_for_paths.get(img_path, {}),
                })
                continue

            to_write.append({
                "image_path": img_path,
                "ok": True,
                "error_code": 0,
                "result": ans,
                "meta": meta_for_paths.get(img_path, {}),
            })

        append_jsonl(out_jsonl, to_write)
        dump_json(sorted(processed_set), processed_json)

        print(f"[patch{X}] saved batch {start}-{start + len(batch_paths)} processed={len(processed_set)}")

    print(f"[patch{X}] done round2")


def main():
    args = parse_args()
    assert args.startidx <= args.endidx, "startidx must be <= endidx"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_MM_LIMIT_PER_PROMPT_VIDEO"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch  # noqa

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"patch range: {args.startidx} -> {args.endidx}")
    print(f"manifest: {MANIFEST_JSONL}")

    manifest_index = load_manifest_index(MANIFEST_JSONL)

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

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3200,   # numeric-only JSON, reduce tokens for speed
        top_k=-1,
        stop_token_ids=[],
    )

    for X in range(args.startidx, args.endidx + 1):
        yes_json = f"{OUT_ROOT}/object365_patch{X}_yes.json"
        out_jsonl = f"{OUT_ROOT}/object365_patch{X}_round1_aff_select.jsonl"
        processed_json = f"{OUT_ROOT}/object365_patch{X}_round1_aff_select_processed.json"

        process_one_patch_round2(
            X=X,
            llm=llm,
            processor=processor,
            sampling_params=sampling_params,
            yes_json=yes_json,
            out_jsonl=out_jsonl,
            processed_json=processed_json,
            manifest_index=manifest_index,
            batch_size=BATCH_SIZE,
        )


if __name__ == "__main__":
    main()