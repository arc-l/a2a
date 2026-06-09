# Agent-assisted expansion from external datasets

Imports masks from external affordance datasets (RAGNet, GraspNet, HANDAL, 3DOI,
InstructPart, …) and turns them into A2A-style **one-to-many** part-affordance
annotations using a local Qwen3-VL labeler. Run order:

1. **`start_vllm_qwen3.sh`** — start the labeler VLM (OpenAI-compatible, `:8000`).
2. **`filter_and_unify_other_data.py`** — unify source masks to 0/1 and get a
   first pass of part name + up to 5 affordance task descriptions per mask.
3. **`cleaning_chain/`** — LLM/VLM quality control, applied in this order:
   `judge_quality` → `judge_affordance_label` → `judge_object_count_mllm` →
   `select_adaptive_quality_from_count` → `extract_yes_annotations` →
   `build_keep_quality_only_by_object_bbox` → `merge`.
   (`infer_qwen32B.py` is the shared batched-inference helper.)
4. **`orps_trps_chain/`** — produce the two instruction styles and merge:
   `build_annotation_input_from_count_filtered` →
   `annotate_orps_from_affordance_mllm` (**ORPS** = Object-Referring Part
   Segmentation, e.g. *"the handle of the mug"*) +
   `annotate_trps_from_affordance_mllm` (**TRPS** = Task-Referring Part
   Segmentation, e.g. *"the part you grasp to pour"*) →
   `merge_orps_trps_to_part_list` → `split_final_annotations_by_count`.
   `remap_error_matches_to_remaining_affordance` is an optional fixup.
5. **`instruct_part_label/`** — the same idea specialized for InstructPart-style
   inputs.

Every script is a small single-purpose tool with its own `argparse` flags — read
the top of each file for its exact inputs and outputs.
