import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


# =========================
# Fixed configuration (does not use argparse)
# =========================
OUT_ROOT = "data/data00"
MODEL_PATH = "data/robot/Qwen3-VL-32B-Instruct"

# Read only the first-round results after merging
ROUND1_GLOB = "object365_patch*_round1_aff_select_merged_v1.jsonl"

# Second-round output
OUT_JSONL_SUFFIX = "_round2_affordance_verify_v1.jsonl"
PROCESSED_JSON_SUFFIX = "_round2_affordance_verify_v1_processed.json"

# Maximum number of candidate records to inject per image
MAX_CANDS_IN_PROMPT = 24

# batch
BATCH_SIZE = 256
GPU_MEMORY_UTIL = 0.85

# bbox injection format precision (must be stable, to make it easier for the VLM to understand)
COORD_DECIMALS = 2

def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Round-2 verify (read merged_v1, resumable)"
    )
    parser.add_argument("--gpu", type=str, required=True, help="GPU ids, e.g. 0 or 5,6")
    parser.add_argument("--startidx", type=int, required=True, help="start patch id")
    parser.add_argument("--endidx", type=int, required=True, help="end patch id (inclusive)")
    return parser.parse_args()

# =========================
# General I/O
# =========================
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


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def safe_check_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


# =========================
# Qwen-VL input
# =========================
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


# =========================
# The prompt you approved (copied verbatim, only placeholders kept)
# =========================
ROUND2_TEMPLATE = (
    "You are a visual assistant capable of analyzing a single image. You will receive an image, the image width and height, and several candidate records. Each candidate record contains an object name, the bbox coordinates of that object, an affordance, and the list of key part names (parts) corresponding to that affordance. Your task is to perform strict verification and scoring using the image and the candidate records. You are only allowed to evaluate the object, bbox, affordance, and parts given in the candidate records; you are not allowed to add new objects, not allowed to add new bounding boxes, not allowed to add new affordances, not allowed to add new part names, and not allowed to rewrite any input string. You may only output one JSON, wrapped in <answer> and </answer>, and do not output any explanatory text.\n"
    "Image width {IMG_W}\n"
    "Image height {IMG_H}\n"
    "Candidate records (object, affordance, parts must be used exactly as given; bbox is only used for verification and scoring):\n"
    "\"\"\"\n"
    "{LABELS}\n"
    "\"\"\"\n"
    "bw_yellow_veto rules\n"
    "If any condition is met, then bw_yellow_veto=1, otherwise bw_yellow_veto=0.\n"
    "Condition 1 Black-and-white image: the entire image has almost no valid color information and is mainly composed of grayscale\n"
    "Condition 2 Yellowed old photo: the entire image shows a clearly overall yellow/brownish old-photo color tone, accompanied by an obvious aged-look filter or paper texture that causes distortion of details and colors\n"
    "image_quality scoring rules (0 to 5)\n"
    "Measured dimensions: sharpness, focus, motion blur, noise, compression artifacts, whether the resolution is sufficient, whether the target's texture and edges are distinguishable, whether it is overexposed/underexposed, and whether severe occlusion causes loss of detail.\n"
    "If bw_yellow_veto=1, then image_quality must be 0 or 1.\n"
    "5 Very clear, the target's edges and texture details are sufficient, noise and compression artifacts are minimal, exposure is normal\n"
    "4 Clear, details are basically sufficient, there is slight noise/artifacts/slight blur but it does not affect learning\n"
    "3 Average, there is visible blur or noise or compression artifacts, details are distinguishable but not sharp enough\n"
    "2 Poor, obvious blur/noise/artifacts or low resolution, insufficient details will affect learning\n"
    "1 Very poor, strong blur or severe noise/artifacts/overexposure or underexposure, the target details are almost unusable\n"
    "0 Unusable, severely damaged or the main subject and key details are almost invisible\n"
    "clutter scoring rules (0 to 5)\n"
    "Definition: the degree to which the clutter of the scene interferes with 'candidate affordance learning', mainly based on background complexity, occlusion and overlap, the number of irrelevant objects, whether the main subject is prominent, and whether attention is distracted.\n"
    "5 Extremely cluttered, the main subject is not prominent, occlusion and overlap are severe, strongly interferes with learning\n"
    "4 Very cluttered, interference is large, the subject is frequently occluded or unclear\n"
    "3 Clearly cluttered, interference is moderate, the subject is still usable but learning difficulty increases\n"
    "2 There is some interference but it is manageable, the subject is fairly prominent\n"
    "1 Relatively clean, interference is very weak\n"
    "0 Very clean, the subject is prominent, almost no interference\n"
    "Per-candidate-record scoring rules\n"
    "For each candidate record i, you must output four scores: object_visible, bbox_quality, parts_visibility, affordance_quality. The output scores should reflect the differences in quality and the strength of evidence between different candidate samples; it is not allowed to generally give all candidates the same high score or the same low score.\n"
    "object_visible scoring rules (0 to 3)\n"
    "Only verify whether this object actually appears within this bbox and can be confirmed.\n"
    "3 The object is clearly visible within the bbox, its shape and appearance features are clear, almost no ambiguity\n"
    "2 The object is visible within the bbox and basically confirmable, but there is slight occlusion/angle effect/partial truncation\n"
    "1 The content within the bbox may be this object but it is uncertain, or only a part is seen and key features are lacking\n"
    "0 The object cannot be seen within the bbox or it is clearly mislabeled (the bbox points at another object/background)\n"
    "bbox_quality scoring rules (0 to 3)\n"
    "Only evaluate whether the bbox is favorable for learning and verification: whether the position is correct, whether the coverage is complete, whether it is too large and contains a lot of background, whether it is too small and misses key regions, whether it severely truncates key regions, and whether the scale is sufficient.\n"
    "Let bw = x2-x1, bh = y2-y1, area = bw*bh.\n"
    "3 The box is tight and completely covers the main region, the background proportion is small, the scale is sufficient for learning details\n"
    "2 The box is basically correct and covers the main region, but there is slight offset/slightly loose/slightly tight/slight truncation or slightly more background\n"
    "1 The box is clearly offset or too loose or too tight, or the scale is too small leading to insufficient details, making verification difficult\n"
    "0 The box is severely wrong (points to the wrong region) or extremely too small/key regions severely missing, cannot support learning\n"
    "parts_visibility scoring rules (0 to 3)\n"
    "Only evaluate whether the key parts listed in the input parts list are visible in the image and correspond to this object. parts may be the object itself or the names of the object's parts, in all cases taking the input as authoritative.\n"
    "3 Most of the key parts in the given parts are clearly visible and clearly correspond to this object\n"
    "2 The parts are visible but not clear enough, or there is partial occlusion/angle that makes the details incomplete, but they are still recognizable\n"
    "1 The parts can only be seen vaguely or are highly uncertain, the evidence for the parts is insufficient\n"
    "0 The parts are basically invisible or cannot be matched to this object\n"
    "affordance_quality scoring rules (0 to 5)\n"
    "Goal: whether this image is suitable as a training sample for this object-this affordance, with strict requirements and discriminative power.\n"
    "Must comprehensively consider:\n"
    "1 Whether object_visible is sufficient (if object_visible=0, then affordance_quality must be 0)\n"
    "2 Whether bbox_quality is sufficient to support learning (the lower bbox_quality is, the lower the upper bound)\n"
    "3 Whether parts_visibility supports this affordance (if parts_visibility=0 and this affordance depends on parts, then the upper bound is significantly lowered)\n"
    "4 Whether the affordance holds in this scene and has visual evidence (for example, its usability/operability/functionality can be inferred to hold, and it matches the object's state)\n"
    "Meaning of the scoring levels:\n"
    "5 Strong evidence and very typical: the object and key parts are clear, the state/pose is very typical for this affordance, almost no ambiguity\n"
    "4 Very strong evidence: overall clear and holds, only minor issues (slight occlusion, slight background interference, or slight box issues)\n"
    "3 Moderate evidence: usable and holds, but not typical enough or the evidence is not strong enough (average angle, average occlusion, average state)\n"
    "2 Weak evidence: barely holds or limited usability (insufficient details, suboptimal state, more occlusion, weak box)\n"
    "1 Very weak evidence: almost unusable (strong occlusion/very poor details/state mismatch), but it still cannot be asserted to completely not hold\n"
    "0 Clearly does not hold or there is an obvious contradiction (wrong object, wrong box, or the scene/state clearly does not support this affordance)\n"
    "Output requirements\n"
    "You may only output one JSON, wrapped in <answer> and </answer>. Do not output any extra text.\n"
    "All string values in the output must be in English (including object, affordance, parts).\n"
    "evaluations must correspond one-to-one with the input candidate records, the order must be consistent, object, affordance, parts must be copied exactly as given and must not be rewritten. Do not output bbox_xyxy.\n"
    "<answer>{ \"bw_yellow_veto\": 0, \"image_quality\": 0, \"clutter\": 0, \"evaluations\": [ { \"object\": \"\", \"affordance\": \"\", \"parts\": [], \"object_visible\": 0, \"bbox_quality\": 0, \"parts_visibility\": 0, \"affordance_quality\": 0 } ] }</answer>"
)


def build_round2_question(img_w: int, img_h: int, labels_text: str) -> str:
    q = ROUND2_TEMPLATE
    q = q.replace("{IMG_W}", str(int(img_w)))
    q = q.replace("{IMG_H}", str(int(img_h)))
    q = q.replace("{LABELS}", labels_text if labels_text else "")
    return q


# =========================
# Candidate injection: strictly use text lines, not JSON injection
# Candidate line format: "{obj}; bbox [x1,y1,x2,y2]; affordance {aff}; parts {parts};"
# =========================
def _fmt_xyxy(xyxy: List[float], nd: int = 2) -> str:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    s = f"{{:.{nd}f}}"
    return f"[{s.format(x1)},{s.format(y1)},{s.format(x2)},{s.format(y2)}]"


def _format_candidate_line(obj: str, bbox_xyxy: List[float], aff: str, parts: List[str]) -> str:
    parts_text = ", ".join(parts) if isinstance(parts, list) and len(parts) > 0 else "whole_object"
    bbox_text = _fmt_xyxy(bbox_xyxy, nd=COORD_DECIMALS)
    return f"{obj}; bbox {bbox_text}; affordance {aff}; parts {parts_text};"


def build_candidates_for_image_from_merged(rec: dict, max_cands: int) -> Optional[Dict[str, Any]]:
    """
    Read only merged_v1:
      rec["result"]["image_width"], rec["result"]["image_height"]
      rec["result"]["selected"][k]["object"/"affordance"/"parts"/"instances_xyxy"]
    Expand instances_xyxy -> multiple candidate records (each with a bbox)
    """
    if not isinstance(rec, dict) or not rec.get("ok", False):
        return None

    ip = rec.get("image_path", "")
    if not isinstance(ip, str) or not ip:
        return None

    result = rec.get("result", {})
    if not isinstance(result, dict):
        return None

    img_w = int(result.get("image_width", 0) or 0)
    img_h = int(result.get("image_height", 0) or 0)
    if img_w <= 0 or img_h <= 0:
        return None

    selected = result.get("selected", [])
    if not isinstance(selected, list) or len(selected) == 0:
        return None

    candidates: List[Dict[str, Any]] = []
    lines: List[str] = []

    for item in selected:
        if not isinstance(item, dict):
            continue

        obj = item.get("object", "")
        aff = item.get("affordance", "")
        parts = item.get("parts", [])

        if not (isinstance(obj, str) and obj):
            continue
        if not (isinstance(aff, str) and aff):
            continue
        if not isinstance(parts, list):
            parts = []

        inst_xyxy = item.get("instances_xyxy", [])
        if not isinstance(inst_xyxy, list) or len(inst_xyxy) == 0:
            continue

        for xyxy in inst_xyxy:
            if not (isinstance(xyxy, list) and len(xyxy) == 4):
                continue

            xyxy_rounded = [round(float(v), COORD_DECIMALS) for v in xyxy]

            candidates.append({
                "object": obj,
                "bbox": xyxy_rounded,
                "affordance": aff,
                "parts": parts,
            })
            lines.append(_format_candidate_line(obj, xyxy_rounded, aff, parts))

            if len(candidates) >= max_cands:
                break
        if len(candidates) >= max_cands:
            break

    if len(candidates) == 0:
        return None

    return {
        "image_path": ip,
        "img_w": img_w,
        "img_h": img_h,
        "labels_text": "\n".join(lines),
        "cand_count": len(candidates),
        "candidates": candidates,  # only used for your post-processing alignment, will not be put into the prompt, nor will the VLM output it
    }


# =========================
# Batch-process one merged_v1 file -> round2 output
# =========================
def process_round1_file(round1_path: Path, llm: LLM, processor, sampling_params: SamplingParams):
    out_jsonl = round1_path.with_name(round1_path.stem + OUT_JSONL_SUFFIX)
    processed_json = round1_path.with_name(round1_path.stem + PROCESSED_JSON_SUFFIX)

    processed_set = set(load_json_list(str(processed_json)))

    todo: List[Dict[str, Any]] = []
    for line in iter_jsonl(str(round1_path)):
        try:
            rec = json.loads(line)
        except Exception:
            continue

        ip = rec.get("image_path", "")
        if not ip or ip in processed_set:
            continue

        if not rec.get("ok", False):
            processed_set.add(ip)
            continue

        pack = build_candidates_for_image_from_merged(rec, max_cands=MAX_CANDS_IN_PROMPT)
        if pack is None:
            processed_set.add(ip)
            continue

        todo.append(pack)

    print(f"[INFO] {round1_path.name} todo={len(todo)} already_processed={len(processed_set)}")
    if not todo:
        dump_json(sorted(processed_set), str(processed_json))
        return

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(todo), BATCH_SIZE):
        batch_items = todo[start:start + BATCH_SIZE]

        batch_inputs = []
        valid_items = []
        meta_for_paths = {}

        for pack in batch_items:
            ip = pack["image_path"]

            if not safe_check_image(ip):
                processed_set.add(ip)
                continue

            question = build_round2_question(pack["img_w"], pack["img_h"], pack["labels_text"])
            item = prepare_single_input_qwen(ip, processor, question)
            if item is None:
                processed_set.add(ip)
                continue

            batch_inputs.append(item)
            valid_items.append(pack)
            meta_for_paths[ip] = {
                "img_w": int(pack["img_w"]),
                "img_h": int(pack["img_h"]),
                "cand_count": int(pack["cand_count"]),
            }

        if not batch_inputs:
            dump_json(sorted(processed_set), str(processed_json))
            continue

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        to_write = []
        for pack, output in zip(valid_items, outputs):
            ip = pack["image_path"]
            raw_text = output.outputs[0].text if output.outputs else ""
            ans = extract_answer_json(raw_text)

            processed_set.add(ip)

            if ans is None:
                to_write.append({
                    "image_path": ip,
                    "ok": False,
                    "error_code": 1,
                    "raw_text": raw_text[:4000],
                    "meta": meta_for_paths.get(ip, {}),
                })
                continue

            to_write.append({
                "image_path": ip,
                "ok": True,
                "error_code": 0,
                "result": ans,
                "meta": meta_for_paths.get(ip, {}),
            })

        append_jsonl(str(out_jsonl), to_write)
        dump_json(sorted(processed_set), str(processed_json))

        print(f"[INFO] {round1_path.name} saved {start}-{start + len(batch_items)} processed={len(processed_set)}")

    print(f"[DONE] round2 output: {out_jsonl}")
    print(f"[DONE] round2 processed: {processed_json}")


def main():
    args = parse_args()
    assert args.startidx <= args.endidx, "startidx must be <= endidx"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["VLLM_MM_LIMIT_PER_PROMPT_VIDEO"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch  # noqa

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"patch range: {args.startidx} -> {args.endidx}")
    print(f"round1 file pattern: object365_patchX_round1_aff_select_merged_v1.jsonl")

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
        max_tokens=3200,
        top_k=-1,
        stop_token_ids=[],
    )

    # Run patch by patch (read only the merged_v1 round1 files)
    for X in range(args.startidx, args.endidx + 1):
        round1_path = Path(OUT_ROOT) / f"object365_patch{X}_round1_aff_select_merged_v1.jsonl"
        if not round1_path.exists():
            print(f"[WARN] missing file: {round1_path}")
            continue

        process_round1_file(
            round1_path=round1_path,
            llm=llm,
            processor=processor,
            sampling_params=sampling_params,
        )


if __name__ == "__main__":
    main()

