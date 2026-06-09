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
# 固定配置（不使用 argparse）
# =========================
OUT_ROOT = "data/data00"
MODEL_PATH = "data/robot/Qwen3-VL-32B-Instruct"

# 只读 merge 后的第一轮结果
ROUND1_GLOB = "object365_patch*_round1_aff_select_merged_v1.jsonl"

# 第二轮输出
OUT_JSONL_SUFFIX = "_round2_affordance_verify_v1.jsonl"
PROCESSED_JSON_SUFFIX = "_round2_affordance_verify_v1_processed.json"

# 每张图最多注入多少条候选记录
MAX_CANDS_IN_PROMPT = 24

# batch
BATCH_SIZE = 256
GPU_MEMORY_UTIL = 0.85

# bbox 注入格式精度（必须稳定，便于 VLM 理解）
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
# 通用 I/O
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
# 你认可的 prompt（严格照抄，仅保留占位符）
# =========================
ROUND2_TEMPLATE = (
    "你是一个能够分析单张图片的视觉助手。你会收到一张图片、图片宽高、以及若干条候选记录。每条候选记录包含 object 名称、该 object 的 bbox 坐标、一个 affordance、以及该 affordance 对应的关键部件 parts 名称列表。你的任务是利用图片与候选记录进行严格核验与打分。你只允许评估候选记录中给出的 object、bbox、affordance、parts，不允许新增物体，不允许新增边界框，不允许新增 affordance，不允许新增部件名称，不允许改写任何输入字符串。你只能输出一个 JSON，并用 <answer> 与 </answer> 包裹，不要输出任何解释文字。\n"
    "图片宽度 {IMG_W}\n"
    "图片高度 {IMG_H}\n"
    "候选记录（object、affordance、parts 必须原样使用；bbox 仅用于核验与评分）:\n"
    "\"\"\"\n"
    "{LABELS}\n"
    "\"\"\"\n"
    "bw_yellow_veto 规则\n"
    "如果满足任一条件则 bw_yellow_veto=1，否则 bw_yellow_veto=0。\n"
    "条件1 黑白图片：整张图几乎没有有效色彩信息，主要由灰度构成\n"
    "条件2 泛黄老照片：整张图出现明显整体偏黄/褐色的老照片色调，且伴随明显年代感滤镜或纸质感导致细节与颜色失真\n"
    "image_quality 评分规则（0 到 5）\n"
    "衡量维度：清晰度、对焦、运动模糊、噪声、压缩伪影、分辨率是否足够、目标纹理与边缘是否可辨、是否过曝/欠曝、是否严重遮挡导致细节丢失。\n"
    "若 bw_yellow_veto=1，则 image_quality 必须为 0 或 1。\n"
    "5 非常清晰，目标边缘与纹理细节充足，噪声和压缩伪影极少，曝光正常\n"
    "4 清晰，细节基本充足，存在轻微噪声/伪影/轻微模糊但不影响学习\n"
    "3 一般，存在可见模糊或噪声或压缩伪影，细节可辨但不够锐利\n"
    "2 较差，明显模糊/噪声/伪影或分辨率偏低，细节不足会影响学习\n"
    "1 极差，强模糊或严重噪声/伪影/过曝欠曝，目标细节几乎不可用\n"
    "0 不可用，严重损坏或几乎看不清主体与关键细节\n"
    "clutter 评分规则（0 到 5）\n"
    "定义：画面杂乱程度对“候选 affordance 学习”的干扰强度，主要看背景复杂度、遮挡与重叠、无关物体数量、主体是否突出、视线是否被分散。\n"
    "5 极度杂乱，主体不突出，遮挡重叠严重，强干扰学习\n"
    "4 很杂乱，干扰很大，经常遮挡或主体不清晰\n"
    "3 杂乱明显，干扰中等，主体仍可用但学习难度上升\n"
    "2 有一定干扰但可控，主体较突出\n"
    "1 比较干净，干扰很弱\n"
    "0 非常干净，主体突出，几乎无干扰\n"
    "候选记录逐条评分规则\n"
    "对每条候选记录 i，你必须输出 object_visible、bbox_quality、parts_visibility、affordance_quality 四项分数。输出的分数应当反映不同候选样本之间的质量差异与证据强弱，不允许所有候选普遍给同一高分或同一低分。\n"
    "object_visible 评分规则（0 到 3）\n"
    "只核验该 object 是否真实出现在该 bbox 中且可确认。\n"
    "3 bbox 内该物体清晰可见，形状与外观特征明确，几乎无歧义\n"
    "2 bbox 内该物体可见且基本可确认，但存在轻微遮挡/角度影响/部分截断\n"
    "1 bbox 内可能是该物体但不确定，或只见到局部且缺乏关键特征\n"
    "0 bbox 内看不到该物体或明显错标（bbox 对着其他物体/背景）\n"
    "bbox_quality 评分规则（0 到 3）\n"
    "只评价 bbox 是否有利于学习与核验：位置是否对、覆盖是否完整、是否过大包含大量背景、是否过小缺失关键区域、是否严重截断关键区域、尺度是否足够。\n"
    "令 bw = x2-x1，bh = y2-y1，area = bw*bh。\n"
    "3 框紧致且完整覆盖主要区域，背景占比小，尺度足够用于学习细节\n"
    "2 框基本正确，覆盖主要区域，但存在轻微偏移/略松/略紧/轻微截断或背景稍多\n"
    "1 框偏移明显或过松过紧，或尺度偏小导致细节不足，核验困难\n"
    "0 框严重错误（指向错误区域）或极度过小/关键区域严重缺失，无法支持学习\n"
    "parts_visibility 评分规则（0 到 3）\n"
    "只评价输入 parts 列表中所列关键部件是否在图中可见且与该 object 对应。parts 可能是物体本身或物体的部件名，均以输入为准。\n"
    "3 给定 parts 中的关键部件大多清晰可见，且与该 object 对应明确\n"
    "2 部件可见但不够清晰，或有部分遮挡/角度导致细节不完整，但仍可辨认\n"
    "1 只能模糊看到部件或高度不确定，部件证据不足\n"
    "0 部件基本不可见或无法对应到该 object\n"
    "affordance_quality 评分规则（0 到 5）\n"
    "目标：该图片是否适合作为该 object-该 affordance 的训练样本，要求严格且具有区分度。\n"
    "必须综合考虑：\n"
    "1 object_visible 是否足够（若 object_visible=0，则 affordance_quality 必须为 0）\n"
    "2 bbox_quality 是否足够承载学习（bbox_quality 越低，上限越低）\n"
    "3 parts_visibility 是否支撑该 affordance（若 parts_visibility=0 且该 affordance 依赖部件，则上限显著降低）\n"
    "4 affordance 是否在该场景成立且有视觉证据（例如可推断其可用性/可操作性/功能成立，且与物体状态匹配）\n"
    "打分等级含义：\n"
    "5 强证据且非常典型：物体与关键部件清晰，状态/姿态对该 affordance 很典型，几乎无歧义\n"
    "4 证据很强：整体清晰且成立，只有轻微问题（轻微遮挡、轻微背景干扰或轻微框问题）\n"
    "3 证据中等：可用且成立，但不够典型或证据不够强（角度一般、遮挡一般、状态一般）\n"
    "2 证据偏弱：勉强成立或可用性有限（细节不足、状态不理想、遮挡偏多、框偏弱）\n"
    "1 证据很弱：几乎不可用（强遮挡/细节很差/状态不匹配），但仍不能断言完全不成立\n"
    "0 明确不成立或明显矛盾（错物体、错框、或场景/状态明显不支持该 affordance）\n"
    "输出要求\n"
    "你只能输出一个 JSON，并用 <answer> 与 </answer> 包裹。不要输出任何额外文字。\n"
    "输出中所有字符串值必须是英文（包含 object、affordance、parts）。\n"
    "evaluations 必须与输入候选记录一一对应，顺序必须一致，object、affordance、parts 必须原样拷贝，不允许改写。不要输出 bbox_xyxy。\n"
    "<answer>{ \"bw_yellow_veto\": 0, \"image_quality\": 0, \"clutter\": 0, \"evaluations\": [ { \"object\": \"\", \"affordance\": \"\", \"parts\": [], \"object_visible\": 0, \"bbox_quality\": 0, \"parts_visibility\": 0, \"affordance_quality\": 0 } ] }</answer>"
)


def build_round2_question(img_w: int, img_h: int, labels_text: str) -> str:
    q = ROUND2_TEMPLATE
    q = q.replace("{IMG_W}", str(int(img_w)))
    q = q.replace("{IMG_H}", str(int(img_h)))
    q = q.replace("{LABELS}", labels_text if labels_text else "")
    return q


# =========================
# 候选注入：严格用文本行，不用 JSON 注入
# 候选行格式："{obj}; bbox [x1,y1,x2,y2]; affordance {aff}; parts {parts};"
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
    只读取 merged_v1：
      rec["result"]["image_width"], rec["result"]["image_height"]
      rec["result"]["selected"][k]["object"/"affordance"/"parts"/"instances_xyxy"]
    展开 instances_xyxy -> 多条候选记录（每条带 bbox）
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
        "candidates": candidates,  # 仅用于你后处理对齐，不会塞进 prompt，也不会让 VLM 输出
    }


# =========================
# 批量处理一个 merged_v1 文件 -> round2 输出
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

    # 逐 patch 跑（只读 merged_v1 的 round1 文件）
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

