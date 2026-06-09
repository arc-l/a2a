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
        description="Qwen-32B Object365 Affordance pre-filtering (patch range + resume from checkpoint)"
    )
    parser.add_argument(
        "--gpu", type=str, required=True,
        help="GPU id(s) to use, e.g. 0 or 5,6 (it is recommended to set CUDA_VISIBLE_DEVICES on the command line)"
    )
    parser.add_argument(
        "--startidx", type=int, required=True,
        help="Start patch index (e.g. 0)"
    )
    parser.add_argument(
        "--endidx", type=int, required=True,
        help="End patch index (e.g. 25, inclusive)"
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
    Consistent with the Qwen script you provided:
    - messages: image + text
    - apply_chat_template
    - process_vision_info -> mm_data + video_kwargs
    - return the dict required by vLLM
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
    tail = clean[-5:].lower()
    if "yes" in tail:
        return "yes"
    if "no" in tail:
        return "no"
    return "no"


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
            if ans == "yes":
                yes_results.append(img_path)

        # Consistent with the GLM reference: flush to disk in real time at the batch level to keep checkpoints safe
        dump_json(yes_results, yes_json)
        dump_json(sorted(processed_set), processed_json)

        print(f"[patch{X}] saved yes={len(yes_results)} processed={len(processed_set)}")

    print(f"[patch{X}] done")


def main():
    args = parse_args()
    assert args.startidx <= args.endidx, "startidx must be <= endidx"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.environ["VLLM_MM_LIMIT_PER_PROMPT_VIDEO"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch  # noqa

    DATA_ROOT = "data/data00/object365/Objects365/data/train"
    OUT_ROOT = "data/data00"
    MODEL_PATH = "data/robot/Qwen3-VL-32B-Instruct"

    QUESTION = (
        "You are now acting as a model for \"filtering Affordance images\", and you need to judge whether the given image contains at least one "
        "\"object-centric target object with a clear subject that, in everyday life, has a typical human-executable Affordance\".\n"
        "The goal of this task is: keep only images that are \"typical and centered on an interactable everyday object\", and filter out Affordances that hold only through imagination or extrapolation "
        "(for example, cases where pillars, stone blocks, steps, snow piles, random platforms, etc. are treated as sittable/jumpable).\n"
        "You must strictly follow the steps below to judge, and may only answer: \"yes\" or \"no\", do not output any other content.\n\n"
        "Step 1: Judge whether the image is \"object-centric with a clear subject\"\n"
        "Only when all of the following conditions are satisfied at the same time is the image considered \"object-centric with a clear subject\":\n"
        "1. The visual attention of the image is focused on a single main object, or on a few strongly related objects, rather than:\n"
        "   - A broad outdoor / indoor large scene (street, plaza, beach, scenery, full room panorama, etc.);\n"
        "   - A crowded crowd, a complex street scene, a shopping mall, or other multi-target mixed scenes;\n"
        "   - A scene centered on a person or animal, where the object is only a small accessory or background.\n"
        "2. The main object is located at the center or a prominent region of the image, and:\n"
        "   - It occupies a noticeable area in the frame (not a tiny dot-like target in the distance);\n"
        "   - Its outline and shape are clearly visible, with no severe occlusion or large-area cropping.\n"
        "3. The main object should be an independent, recognizable physical object, rather than:\n"
        "   - The ground, walls, ceiling, stairs, steps, railings, pillars, stone blocks, rocks, snow piles, dirt mounds, platforms, or other environmental structures or natural terrain;\n"
        "   - Abstract geometric shapes, textures, patterns, signboards, road markings, clouds, scenery, etc.\n"
        "If the image fails to satisfy any one of the above (for example: a large scene, an unclear subject, an object that is too small, only environmental structures, etc.), please answer directly: \"no\".\n\n"
        "Step 2: Judge whether the main object is a \"typical everyday interactable object\" and has a human-executable Affordance\n"
        "Only proceed to Step 2 after passing Step 1.\n"
        "1. The main object must be a typical everyday man-made object or a common edible object, with a category similar to the following kinds of items:\n"
        "   - Furniture and supports: chair, couch, bench, bed, etc.;\n"
        "   - Tableware and containers: cup, bottle, bowl, plate, spoon, fork, knife, wine_glass, etc.;\n"
        "   - Kitchen / home appliances: microwave, oven, refrigerator, etc.;\n"
        "   - Handheld tools and stationery: pen, scissors, hammer, axe, toothbrush, etc.;\n"
        "   - Electronic devices: cell_phone, camera, laptop, keyboard, etc.;\n"
        "   - Vehicles: bicycle, motorcycle, etc.;\n"
        "   - Sports equipment and balls: soccer_ball, basketball, baseball, tennis_racket, golf_clubs, baseball_bat, badminton, frisbee, "
        "javelin, discus, skis, snowboard, skateboard, punching_bag, etc.;\n"
        "   - Luggage and containers: suitcase, etc.;\n"
        "   - Common foods: apple, banana, orange, carrot, broccoli, hot_dog, etc.\n"
        "   If the main object is clearly an environmental structure, natural terrain, or a hard-to-name abstract shape "
        "(for example: pillars, stone blocks, steps, stairs, snow piles, rocks, random platforms, floors, walls, etc.), "
        "then even if it \"looks like it can be sat on / jumped on / kicked\", it should be regarded as not meeting this task's requirement, and you should answer directly: \"no\".\n"
        "2. When the main object belongs to the above \"typical everyday objects\", further judge whether, in a normal life context, it naturally supports any one of the following human actions "
        "(the Affordance labels are written in English):\n"
        "Push, Drink_with, Take_photo, Brush_with, Swing, Talk_on, Beat, Ride, Wash, Pour, Open, Cut, Eat, Look_out, Lie_on, "
        "Stir, Boxing, Hit, Pick_up, Cut_with, Throw, Catch, Text_on, Sit_on, Pack, Drag, Hold, Write, Kick, Peel, Lift, Stick, "
        "Type_on, Sip, Carry, Jump\n"
        "Judgment principles:\n"
        "   - Only consider whether, under everyday, normal usage, the object would be used by humans in this way;\n"
        "   - Do not rely on exaggerated, dangerous, or extreme usages, and do not infer through pure imagination that \"it looks like it could also\";\n"
        "   - For example: a chair is used for Sit_on, a cup/wine glass/spoon is used for Drink_with / Sip, a phone is used for Take_photo / Talk_on / Text_on, "
        "a keyboard and a laptop are used for Type_on, a knife/scissors is used for Cut_with, a ball is used for Kick / Throw / Catch, etc., which are all reasonable typical cases.\n\n"
        "If all of the following are satisfied at the same time:\n"
        "(1) The image is visually centered on a single main object with a clear subject (Step 1 is \"yes\");\n"
        "(2) The object belongs to the above \"typical everyday interactable objects\" and, in normal life, naturally supports at least one of the above Affordances (Step 2 is \"yes\");\n"
        "then please answer only: yes\n"
        "In any other case (including: the subject is a pillar, stone, step, snow pile, or other environmental structure; the subject cannot be clearly classified into a common everyday object; "
        "or it can be considered sittable/kickable/jumpable only through association/extrapolation), please answer only: no\n"
        "Do not output any other content (including explanations, symbols, confidence scores, or extra text)."
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
        max_tokens=3200,   # D: keep unchanged
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