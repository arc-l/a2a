import os
import json
import argparse
from pathlib import Path
from PIL import Image

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen-32B Object365 Affordance 预筛选（patch 范围 + 断点续跑）"
    )
    parser.add_argument(
        "--gpu", type=str, required=True,
        help="使用的 GPU 编号，例如 0 或 5,6（建议命令行设置 CUDA_VISIBLE_DEVICES）"
    )
    parser.add_argument(
        "--startidx", type=int, required=True,
        help="起始 patch 编号（例如 0）"
    )
    parser.add_argument(
        "--endidx", type=int, required=True,
        help="结束 patch 编号（例如 25，闭区间）"
    )
    return parser.parse_args()


def safe_check_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_single_input_qwen(image_path: str, processor, question: str):
    """
    与你给的 Qwen 脚本一致：
    - messages: image + text
    - apply_chat_template
    - process_vision_info -> mm_data + video_kwargs
    - 返回 vLLM 所需 dict
    """
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
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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


def extract_final_answer(text):
    clean = text.replace(" ", "").replace("\n", "")
    if "<answer>" in clean:
        try:
            return clean.split("<answer>")[1].split("<")[0]
        except:
            return clean.split("<answer>")[1]
    tail = clean[-5:]
    if "是" in tail:
        return "是"
    if "否" in tail:
        return "否"
    return "否"


def load_json_list(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def process_one_patch(
    X: int,
    llm: LLM,
    processor,
    sampling_params: SamplingParams,
    question: str,
    image_root: str,
    yes_json: str,
    processed_json: str,
    batch_size: int = 256,
):
    if not os.path.exists(image_root):
        print(f"[patch{X}] missing: {image_root}")
        return

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_images = [
        str(p) for p in Path(image_root).rglob("*")
        if p.suffix.lower() in exts
    ]
    all_images.sort()

    yes_results = load_json_list(yes_json)
    processed_set = set(load_json_list(processed_json))

    remaining = [p for p in all_images if p not in processed_set]
    print(f"[patch{X}] total={len(all_images)} remaining={len(remaining)}")

    for start in range(0, len(remaining), batch_size):
        batch_paths = remaining[start:start + batch_size]

        batch_inputs = []
        valid_paths = []

        for img_path in batch_paths:
            item = prepare_single_input_qwen(img_path, processor, question)
            if item is None:
                processed_set.add(img_path)
                continue
            batch_inputs.append(item)
            valid_paths.append(img_path)

        if not batch_inputs:
            dump_json(yes_results, yes_json)
            dump_json(sorted(processed_set), processed_json)
            continue

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        for img_path, output in zip(valid_paths, outputs):
            raw_text = output.outputs[0].text
            ans = extract_final_answer(raw_text)

            processed_set.add(img_path)
            if ans == "是":
                yes_results.append(img_path)

        # 与 GLM 参考一致：batch 级实时写盘，保证断点安全
        dump_json(yes_results, yes_json)
        dump_json(sorted(processed_set), processed_json)

        print(f"[patch{X}] saved yes={len(yes_results)} processed={len(processed_set)}")

    print(f"[patch{X}] done")


def main():
    args = parse_args()
    assert args.startidx <= args.endidx, "startidx 必须 <= endidx"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.environ["VLLM_MM_LIMIT_PER_PROMPT_VIDEO"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch  # noqa

    DATA_ROOT = "data/data00/object365/Objects365/data/train"
    OUT_ROOT = "data/data00"
    MODEL_PATH = "data/robot/Qwen3-VL-32B-Instruct"

    QUESTION = (
        "你现在作为一个用于“筛选 Affordance 图像”的模型，需要判断给定图像中是否存在至少一个"
        "“以物体为中心、主体明确，且在日常生活中具有典型人类可执行 Affordance 的目标物体”。\n"
        "本任务的目标是：只保留“典型的、以可交互日常物体为主体”的图像，而过滤掉依靠想象或引申才成立的 Affordance"
        "（例如柱子、石墩、台阶、雪堆、随机平台等被当成可坐/可跳的情况）。\n"
        "你必须严格按照以下步骤判断，只能回答：“是” 或 “否”，不要输出任何其它内容。\n\n"
        "第 1 步：判断是否为“物体为中心、主体明确”的图像\n"
        "只有在同时满足以下条件时，才认为图像是“物体为中心、主体明确”的：\n"
        "1. 图像的视觉注意力集中在一个主要物体上，或少数几件强相关的物体上，而不是：\n"
        "   - 宽广的室外 / 室内大场景（街道、广场、海滩、风景、房间全景等）；\n"
        "   - 拥挤的人群、复杂街景、商场等多目标混杂场景；\n"
        "   - 以人物或动物为主体，而物体只是小配件或背景。\n"
        "2. 该主要物体位于图像中心或显著区域，并且：\n"
        "   - 在画面中占据明显面积（不是远处很小的点状目标）；\n"
        "   - 轮廓和形状清晰可见，没有严重遮挡或大面积裁切。\n"
        "3. 主要物体应该是独立的、可识别的实体物体，而不是：\n"
        "   - 地面、墙壁、天花板、楼梯、台阶、栏杆、柱子、石块、岩石、雪堆、土堆、台子等环境结构或自然地形；\n"
        "   - 抽象几何体、纹理、图案、标志牌、路面标线、云彩、风景等。\n"
        "如果图像不满足以上任意一条（例如：大场景、主体不清晰、物体太小、只有环境结构等），请直接回答：“否”。\n\n"
        "第 2 步：判断主要物体是否是“典型日常可交互物体”，并具有人类可执行的 Affordance\n"
        "仅当通过第 1 步后，才继续第 2 步。\n"
        "1. 主要物体必须是典型的日常人造物体或常见可食用物体，类别应类似于下列这一类物品：\n"
        "   - 家具与支撑物：chair, couch, bench, bed 等；\n"
        "   - 餐具与容器：cup, bottle, bowl, plate, spoon, fork, knife, wine_glass 等；\n"
        "   - 厨房 / 家电：microwave, oven, refrigerator 等；\n"
        "   - 手持工具与文具：pen, scissors, hammer, axe, toothbrush 等；\n"
        "   - 电子设备：cell_phone, camera, laptop, keyboard 等；\n"
        "   - 载具：bicycle, motorcycle 等；\n"
        "   - 体育器材与球类：soccer_ball, basketball, baseball, tennis_racket, golf_clubs, baseball_bat, badminton, frisbee, "
        "javelin, discus, skis, snowboard, skateboard, punching_bag 等；\n"
        "   - 行李与容器：suitcase 等；\n"
        "   - 常见食物：apple, banana, orange, carrot, broccoli, hot_dog 等。\n"
        "   如果主要物体明显是环境结构、自然地形或难以命名的抽象形状"
        "（例如：柱子、石墩、台阶、楼梯、雪堆、岩石、随机台子、地板、墙面等），"
        "即使它“看上去可以坐/可以跳/可以踢”，也要视为不符合本任务要求，直接回答：“否”。\n"
        "2. 在主要物体属于上述“典型日常物体”时，再判断它在正常生活情境下，是否自然地支持下列任一人类操作"
        "（Affordance 标签以英文记述）：\n"
        "Push, Drink_with, Take_photo, Brush_with, Swing, Talk_on, Beat, Ride, Wash, Pour, Open, Cut, Eat, Look_out, Lie_on, "
        "Stir, Boxing, Hit, Pick_up, Cut_with, Throw, Catch, Text_on, Sit_on, Pack, Drag, Hold, Write, Kick, Peel, Lift, Stick, "
        "Type_on, Sip, Carry, Jump\n"
        "判断原则：\n"
        "   - 只考虑该物体在日常、常规用法下是否会被人类这样使用；\n"
        "   - 不要依赖夸张、危险或极端的用法，也不要通过纯粹想象“看起来也可以”来推断；\n"
        "   - 例如：椅子用来 Sit_on，杯子/酒杯/勺子用来 Drink_with / Sip，手机用来 Take_photo / Talk_on / Text_on，"
        "键盘和笔记本电脑用来 Type_on，刀/剪刀用来 Cut_with，球类用来 Kick / Throw / Catch 等，都是合理的典型情况。\n\n"
        "若同时满足：\n"
        "(1) 图像以一个主要物体为视觉中心、主体清晰（第 1 步为“是”）；\n"
        "(2) 该物体属于上述“典型日常可交互物体”，且在正常生活中自然地支持至少一种上述 Affordance（第 2 步为“是”）；\n"
        "则请只回答：是\n"
        "其他任何情况（包括：主体是柱子、石头、台阶、雪堆等环境结构；主体无法清晰归类到常见日常物体；"
        "或者只能通过联想/引申才认为可以坐/踢/跳），请只回答：否\n"
        "不要输出任何其他内容（包括解释、符号、置信度或多余文字）。"
    )

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"patch range: {args.startidx} -> {args.endidx}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    tp = torch.cuda.device_count()
    if tp <= 0:
        raise RuntimeError("No visible CUDA devices. Check CUDA_VISIBLE_DEVICES.")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=tp,
        mm_encoder_tp_mode="data",
        gpu_memory_utilization=0.85,  # C
        max_model_len=32768,          # C: >16384
        enforce_eager=True,
        trust_remote_code=True,
        enable_expert_parallel=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3200,   # D 保持不变
        top_k=-1,
        stop_token_ids=[],
    )

    batch_size = 256  # B

    for X in range(args.startidx, args.endidx + 1):
        image_root = f"{DATA_ROOT}/patch{X}"
        yes_json = f"{OUT_ROOT}/object365_patch{X}_yes.json"
        processed_json = f"{OUT_ROOT}/object365_patch{X}_processed.json"

        process_one_patch(
            X=X,
            llm=llm,
            processor=processor,
            sampling_params=sampling_params,
            question=QUESTION,
            image_root=image_root,
            yes_json=yes_json,
            processed_json=processed_json,
            batch_size=batch_size,
        )


if __name__ == "__main__":
    main()