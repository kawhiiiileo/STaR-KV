import torch
import ast
import json
import time
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
import copy
import itertools
from multiprocessing import freeze_support
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info
from opencua_utils import opencua_parse_action, analyze_vision_tokens_opencua_multi_images
from ui_tars_utils import (
    parse_action_to_structure_output, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR,
    analyze_vision_tokens_multi_images
)
from attention_helpers import (
    add_flash_attention_2_argument,
    require_flash_attention_2,
    replace_qwen2_5_vl,
    replace_opencua,
    set_attention_implementation,
    configure_accelerate_skip_attention,
    set_kv_cache_budget,
    set_vision_start_idx,
    apply_vision_analysis_to_model,
    set_vision_end_idx,
    set_alpha,
    set_window_size,
    set_move_attention_to_cpu,
    set_temperature,
    collect_temporal_diagnostics_from_model,
    collect_original_mi_diagnostics_from_model,
    reset_gpu_memory_stats,
    collect_gpu_memory_stats,
    reset_starkv_per_sample_state,
    reset_kv_cache_stats,
    set_starkv_group_config,
    is_starkv_kv,
    is_compressed_kv,
    needs_vision_spans,
    set_kv_cache_mode,
    collect_kv_cache_stats,
    compute_kv_cache_memory_summary,
    collect_mi_diagnostics_from_model,
    collect_cvi_diagnostics_from_model,
    collect_group_specialization_from_model,
    timed_generate,
    reset_runtime_metrics,
    collect_runtime_metrics,
    RUNTIME_METRICS,
)
from attention_helpers import add_starkv_kv_arguments, finalize_starkv_args
from eval_paths import (
    add_model_path_argument,
    add_results_dir_argument,
    add_screenspot_pro_dataset_args,
    validate_required_paths,
)


logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)


GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""

# Device selection utility
def get_device():
    """Dynamically select the best available device"""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU device")
    return device


# --- ScreenSpot Pro grading (merged from screenspotpro_utils.py) ---

def _ssp_collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    filtered_results = []
    for sample in results:
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)
    return filtered_results


def _ssp_make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    unique_values = {
        "platform": set(), "group": set(), "application": set(), "language": set(),
        "gt_type": set(), "instruction_style": set(), "ui_type": set(),
    }
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []
    combinations = []
    for combination in itertools.product(*filtered_values.values()):
        combinations.append(dict(zip(filtered_values.keys(), combination)))
    return combinations


def _ssp_calc_metric_for_result_list(results):
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")
    text_results = _ssp_collect_results_to_eval(results, ui_type="text")
    icon_results = _ssp_collect_results_to_eval(results, ui_type="icon")
    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    return {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0,
    }


def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    click_point = response["point"]
    if click_point is None:
        return "wrong_format"
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    return "wrong"


def _ssp_evaluate_fine_grained(results):
    evaluation_result = {}
    for combo in _ssp_make_combinations(results, platform=True, application=True, instruction_style=True, gt_type=True):
        filtered_results = _ssp_collect_results_to_eval(
            results=results,
            platform=combo.get("platform"),
            application=combo.get("application"),
            instruction_style=combo.get("instruction_style"),
            gt_type=combo.get("gt_type"),
        )
        metrics = _ssp_calc_metric_for_result_list(filtered_results)
        if metrics["num_total"] == 0:
            continue
        key = f"plat:{combo.get('platform')} app:{combo.get('application')} inst_style:{combo.get('instruction_style')} gt_type:{combo.get('gt_type')}"
        evaluation_result[key] = metrics
    return evaluation_result


def _ssp_evaluate_seeclick_paper_style(results):
    evaluation_result = {}
    for combo in _ssp_make_combinations(results, platform=True, instruction_style=True, gt_type=True):
        filtered_results = _ssp_collect_results_to_eval(
            results=results,
            platform=combo.get("platform"),
            instruction_style=combo.get("instruction_style"),
            gt_type=combo.get("gt_type"),
        )
        metrics = _ssp_calc_metric_for_result_list(filtered_results)
        if metrics["num_total"] == 0:
            continue
        key = f"plat:{combo.get('platform')} inst_style:{combo.get('instruction_style')} gt_type:{combo.get('gt_type')}"
        evaluation_result[key] = metrics
    return evaluation_result


def _ssp_evaluate_leaderboard_detailed_style(results):
    evaluation_result = {}
    for combo in _ssp_make_combinations(results, application=True):
        filtered_results = _ssp_collect_results_to_eval(results=results, application=combo.get("application"))
        metrics = _ssp_calc_metric_for_result_list(filtered_results)
        if metrics["num_total"] == 0:
            continue
        evaluation_result[f"app:{combo.get('application')}"] = metrics
    return evaluation_result


def _ssp_evaluate_leaderboard_simple_style(results):
    evaluation_result = {}
    for combo in _ssp_make_combinations(results, group=True):
        filtered_results = _ssp_collect_results_to_eval(results=results, group=combo.get("group"))
        metrics = _ssp_calc_metric_for_result_list(filtered_results)
        if metrics["num_total"] == 0:
            continue
        evaluation_result[f"group:{combo.get('group')}"] = metrics
    return evaluation_result


def screenspotpro_evaluate(results):
    """Collect SSP results and calculate metrics."""
    return {
        "details": results,
        "metrics": {
            "fine_grained": _ssp_evaluate_fine_grained(results),
            "seeclick_style": _ssp_evaluate_seeclick_paper_style(results),
            "leaderboard_simple_style": _ssp_evaluate_leaderboard_simple_style(results),
            "leaderboard_detailed_style": _ssp_evaluate_leaderboard_detailed_style(results),
            "overall": _ssp_calc_metric_for_result_list(results),
        },
    }



def _aggregate_cvi_recon_from_results(results):
    """Aggregate per-sample / per-layer CVI recon diagnostics."""
    import math

    layer_errs_base = []
    layer_errs_act = []
    layer_delta = []
    layer_improved = 0
    layer_total = 0
    spearman = []
    jaccard = []
    replaced = []
    scopes = {}
    fallbacks = {}
    for r in results:
        by = r.get("cvi_recon_by_layer") or {}
        if not by and isinstance(r.get("cvi_recon"), dict):
            by = {"0": r["cvi_recon"]}
        for st in by.values():
            if not isinstance(st, dict):
                continue
            eb = st.get("recon_error_base")
            ea = st.get("recon_error_active")
            if isinstance(eb, (int, float)) and math.isfinite(eb):
                layer_errs_base.append(float(eb))
            if isinstance(ea, (int, float)) and math.isfinite(ea):
                layer_errs_act.append(float(ea))
            if isinstance(eb, (int, float)) and isinstance(ea, (int, float)):
                d = float(ea) - float(eb)
                layer_delta.append(d)
                layer_total += 1
                if float(ea) < float(eb) - 1e-12:
                    layer_improved += 1
            sp = st.get("spearman_utility_vs_base")
            if isinstance(sp, (int, float)) and math.isfinite(sp):
                spearman.append(float(sp))
            jj = st.get("topk_jaccard_base_vs_active")
            if isinstance(jj, (int, float)) and math.isfinite(jj):
                jaccard.append(float(jj))
            rc = st.get("topk_replaced_count")
            if isinstance(rc, (int, float)):
                replaced.append(float(rc))
            sc = st.get("query_scope_resolved")
            if sc:
                scopes[str(sc)] = scopes.get(str(sc), 0) + 1
            fb = st.get("query_scope_fallback_reason")
            if fb:
                fallbacks[str(fb)] = fallbacks.get(str(fb), 0) + 1

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n_samples": len(results),
        "n_layer_obs": layer_total,
        "recon_error_base_mean": _mean(layer_errs_base),
        "recon_error_active_mean": _mean(layer_errs_act),
        "recon_error_delta_mean": _mean(layer_delta),
        "frac_layers_recon_improved": (layer_improved / layer_total) if layer_total else None,
        "spearman_utility_vs_base_mean": _mean(spearman),
        "topk_jaccard_mean": _mean(jaccard),
        "topk_replaced_mean": _mean(replaced),
        "query_scope_resolved_counts": scopes,
        "query_scope_fallback_counts": fallbacks,
    }


def _write_group_specialization_reports(results_dir, results, summary_results):
    """No-op: group specialization diagnostics removed."""
    return


if __name__ == '__main__':
    freeze_support()
    
    parser = argparse.ArgumentParser()
    add_model_path_argument(parser)
    add_screenspot_pro_dataset_args(parser)
    parser.add_argument('--task', type=str, required=True, choices=["all"])
    parser.add_argument('--debug', default=None, type=int)
    parser.add_argument('--sample_offset', default=0, type=int,
                        help='Skip the first N deterministic samples (held-out calibration).')
    parser.add_argument('--num_chunks', type=int, default=1, help='Number of chunks to split data across GPUs.')
    parser.add_argument('--chunk_id', type=int, default=0, help='Which chunk (0-indexed) this process handles.')
    parser.add_argument('--max_new_tokens', type=int, default=400)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/mps/cpu). If not specified, will auto-detect best available device.')
    parser.add_argument('--model_dtype', type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help='Data type to use (auto/bfloat16/float16/float32).')
    add_flash_attention_2_argument(parser)
    add_starkv_kv_arguments(
        parser,
        kv_cache_default="original",
        kv_cache_budget_default=100,
        include_max_samples=True,
    )
    add_results_dir_argument(parser)
    parser.add_argument('--max_pixels', type=int, default=MAX_PIXELS, help='Optional max pixels for processor.')
    parser.add_argument('--min_pixels', type=int, default=MIN_PIXELS, help='Optional min pixels for processor.')
    args = parser.parse_args()
    validate_required_paths(args, ("model_path", "screenspot_imgs", "screenspot_test"))
    finalize_starkv_args(args, disable_starkv_extras_for_original=True)

    if args.num_chunks < 1:
        raise SystemExit("--num_chunks must be >= 1")
    if args.num_chunks > 1 and not (0 <= args.chunk_id < args.num_chunks):
        raise SystemExit(
            f"--chunk_id must satisfy 0 <= chunk_id < num_chunks; got chunk_id={args.chunk_id}, num_chunks={args.num_chunks}"
        )

    if args.device:
        device = args.device
        print(f"Using user-specified device: {device}")
    else:
        device = get_device()
    
    print(f"Selected device: {device}")
    print(f"Number of CUDA devices available: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    try:
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = "cpu"
    except Exception as e:
        print(f"Device validation error: {e}. Using CPU as fallback.")
        device = "cpu"

    require_flash_attention_2(args, device)

    model_path = args.model_path
    print("model_path: ", model_path)

    # Processor from local model_path only (no HuggingFace hub).
    # Keep parse min/max aligned with local preprocessor_config (same as androidcontrol).
    if "UI-TARS" in args.model_path:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True, use_fast=False
        )
        ip = getattr(processor, "image_processor", None)
        # Explicit CLI override only when user changes away from ui_tars_utils defaults.
        if ip is not None:
            if int(args.min_pixels) != int(MIN_PIXELS):
                ip.min_pixels = int(args.min_pixels)
            if int(args.max_pixels) != int(MAX_PIXELS):
                ip.max_pixels = int(args.max_pixels)
            if getattr(ip, "min_pixels", None) is not None:
                MIN_PIXELS = int(ip.min_pixels)
            if getattr(ip, "max_pixels", None) is not None:
                MAX_PIXELS = int(ip.max_pixels)
        print(
            f"UI-TARS processor from {model_path}; "
            f"min_pixels={MIN_PIXELS}, max_pixels={MAX_PIXELS}"
        )
    elif "opencua" in args.model_path.lower():
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True, use_fast=False
        )
        tokenizer = processor.tokenizer
    else:
        raise NotImplementedError("Model not supported")
    
    if args.model_dtype == "float32":
        model_dtype = torch.float32
    elif args.model_dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif args.model_dtype == "float16":
        model_dtype = torch.float16
    elif args.model_dtype == "auto":
        model_dtype = "auto"
    else:
        raise ValueError(f"Invalid model dtype: {args.model_dtype}")

    if "UI-TARS" in args.model_path:
        replace_qwen2_5_vl(kv_cache_mode=args.kv_cache)
    elif "opencua" in args.model_path.lower():
        replace_opencua(kv_cache_mode=args.kv_cache)
    else:
        raise NotImplementedError("Model not supported")

    if device == "cpu":
        if "UI-TARS" in args.model_path:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu", 
                attn_implementation=args.attention_implementation,
            )
        elif "opencua" in args.model_path.lower():
            model = AutoModel.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu",
                attn_implementation=args.attention_implementation,
                trust_remote_code=True
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if is_compressed_kv(args.kv_cache):
            set_kv_cache_mode(model, args)
        if is_starkv_kv(args.kv_cache):
            set_starkv_group_config(model, args)
        reset_kv_cache_stats()
    else:
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            device_map = {"": "cuda:0"}
        if "UI-TARS" in args.model_path:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map, 
                attn_implementation=args.attention_implementation,
            )
        elif "opencua" in args.model_path.lower():
            model = AutoModel.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map,
                attn_implementation=args.attention_implementation,
                trust_remote_code=True
            )
        else:
            raise NotImplementedError("Model not supported")
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if is_compressed_kv(args.kv_cache):
            set_kv_cache_mode(model, args)
        if is_starkv_kv(args.kv_cache):
            set_starkv_group_config(model, args)
        reset_gpu_memory_stats()
        reset_kv_cache_stats()
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            configure_accelerate_skip_attention(model)

    print("Load Success")

    if args.task == "all":
        task_filenames = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ])
        print("task_filenames: ", len(task_filenames))
    else:
        raise NotImplementedError("Task not implemented")
    
    
    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
            task_data = json.load(f)
        gt_types = ["positive"]
        inst_styles = ["instruction"]
        languages = ["en"]
        # Create the list of tasks to run, one item as an instance. Tasks may be reused.
        for inst_style in inst_styles:  # Expand tasks based on user configurations
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang
                        if lang == "cn":
                            if inst_style!= 'instruction' or gt_type != 'positive':
                                raise AttributeError(
                                    "Chinese ScreenSpot-Pro: only positive samples with instruction style "
                                    "'instruction' are supported in this release."
                                )
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        elif lang == "en":
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                        tasks_to_run.append(task_instance)
        print(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    print(f"Total tasks: {len(tasks_to_run)}")

    sample_offset = max(0, int(getattr(args, "sample_offset", 0) or 0))
    if sample_offset:
        tasks_to_run = tasks_to_run[sample_offset:]
        print(f"Sample offset: skipped first {sample_offset} deterministic samples")

    results = []

    if args.debug is not None:
        tasks_to_run = tasks_to_run[:args.debug]
        print("Num of sample: " + str(len(tasks_to_run)) + f" (limited to {args.debug} for quick evaluation)")
    else:
        print("Num of sample: " + str(len(tasks_to_run)))

    # Chunk data for multi-GPU parallelism
    total_tasks = len(tasks_to_run)
    if args.num_chunks > 1:
        chunk_size = (total_tasks + args.num_chunks - 1) // args.num_chunks
        start_idx = args.chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, total_tasks)
        tasks_to_run = tasks_to_run[start_idx:end_idx]
        print(f"Chunk {args.chunk_id}/{args.num_chunks}: [{start_idx}:{end_idx}] ({len(tasks_to_run)} tasks)")
    else:
        start_idx = 0
        end_idx = total_tasks

    # Start timing here
    eval_start_time = time.time()
    if getattr(args, "runtime_metrics", False):
        reset_runtime_metrics()
        RUNTIME_METRICS.enable(True)
        RUNTIME_METRICS.mark_run_start()

    if args.max_samples is not None and args.max_samples > 0:
        tasks_to_run = tasks_to_run[:args.max_samples]
        print(f"[SMOKE TEST] Limited to first {len(tasks_to_run)} samples")

    for sample in tqdm(tasks_to_run, desc="Processing samples", total=len(tasks_to_run)):
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)

       
        
        image = Image.open(img_path)
        # img_width, img_height = image.size  # PIL Image.size returns (width, height)
        instruction = sample["prompt_to_evaluate"]
        img_width, img_height = image.size

        # resized_height, resized_width = smart_resize(img_height, img_width)
        if "UI-TARS" in args.model_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path, 
                        },
                        {"type": "text", "text": GROUNDING_DOUBAO.format(instruction=instruction)},
                    ],
                }
            ]
            
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            

            image_inputs, video_inputs = process_vision_info(messages)
            
            ### HF
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
        elif "opencua" in args.model_path.lower():
            # Match screenspotv2_eval OpenCUA path: system text folded into user turn.
            # Separate system-role + bare instruction yields non-pyautogui JSON and 0% parse.
            OPENCUA_SYSTEM_PROMPT = (
                "You are a GUI agent. You are given a task and a screenshot of the screen. "
                "You need to perform a series of pyautogui actions to complete the task."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": OPENCUA_SYSTEM_PROMPT + "\n" + instruction},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
        

        # Analyze vision tokens for KV cache methods that require it
        if needs_vision_spans(args.kv_cache):
            if "UI-TARS" in args.model_path:
                vision_analysis = analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=1)
            elif "opencua" in args.model_path.lower():
                vision_analysis = analyze_vision_tokens_opencua_multi_images(tokenizer, inputs.input_ids, image_grid_thw=inputs.image_grid_thw, merge_size=2, image_count=1)
                vsi = vision_analysis.get("vision_start_idx")
                vei = vision_analysis.get("vision_end_idx")
                print(f'[VISION] vision_start_idx={vsi}, vision_end_idx={vei}')
            else:
                raise NotImplementedError("Model not supported")

        set_window_size(model, args)
        if needs_vision_spans(args.kv_cache):
            apply_vision_analysis_to_model(model, vision_analysis, args)
            set_alpha(model, args)
            set_temperature(model, args)

        # Reset online MI and temporal state to avoid cross-sample leakage.
        reset_starkv_per_sample_state(model, args)

        output_text = ""
        try:
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
                output_attentions=False,
                use_cache=True,
                do_sample=False,
                return_dict_in_generate=True,
            )
            if getattr(args, "runtime_metrics", False):
                outputs, _lat = timed_generate(
                    model,
                    synchronize_cuda=bool(getattr(args, "runtime_metrics_sync_cuda", True)),
                    record=True,
                    **inputs,
                    **gen_kwargs,
                )
            elif "UI-TARS" in args.model_path:
                outputs = model.generate(**inputs, **gen_kwargs)
            elif "opencua" in args.model_path.lower():
                outputs = model.generate(**inputs, **gen_kwargs)
            else:
                outputs = None

            if outputs is not None and ("UI-TARS" in args.model_path or "opencua" in args.model_path.lower()):
                generated_ids = outputs if not hasattr(outputs, 'sequences') else outputs.sequences
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                print("output_text: ", output_text)
        except Exception as e:
            import traceback
            print(f"Error in generation: {e}")
            traceback.print_exc()
            print("Using outputs from the previous iteration")
            print(f"Image dimensions: {image.size}")

        sample_result = {
            "id": sample["id"],
            "img_path": img_path,
            "group": sample["group"] if "group" in sample else None,
            "platform": sample["platform"],
            "application": sample["application"],
            "lang": sample["language"],
            "instruction_style": sample["instruction_style"],
            "prompt_to_evaluate": sample["prompt_to_evaluate"],
            "gt_type": sample["gt_type"],
            "ui_type": sample["ui_type"],
            "task_filename": sample["task_filename"],

            "raw_response": output_text
        }

        try:
            if "UI-TARS" in args.model_path:
                parsed_actions = parse_action_to_structure_output(output_text,
                    origin_resized_height=img_height,
                    origin_resized_width=img_width,
                    max_pixels=MAX_PIXELS,
                    min_pixels=MIN_PIXELS,
                    factor=IMAGE_FACTOR,
                    model_type="qwen25vl")[0]

                click_point = list(parsed_actions["action_inputs"].values())[0]

                click_point = ast.literal_eval(click_point)
            elif "opencua" in args.model_path.lower():
                parsed_actions = opencua_parse_action(output_text,
                            origin_resized_height=img_height,
                            origin_resized_width=img_width,
                            max_pixels=MAX_PIXELS,
                            min_pixels=MIN_PIXELS,
                            factor=IMAGE_FACTOR,
                            model_type="qwen25vl")
                click_point =  parsed_actions[0]["coordinate"]

            else:
                raise NotImplementedError("Model not supported")

            response = {
                "point": click_point,
            }
            if sample["gt_type"] == "positive":
                correctness = eval_sample_positive_gt(sample, response)
                sample_result.update({
                    "bbox": sample["bbox"],
                })
            else:
                raise NotImplementedError("Negative samples are not supported")

        except Exception as e:
            print(output_text)
            print(e)
            click_point = None
            correctness = "wrong"

        sample_result.update({
            "pred": click_point,
            "correctness": correctness,
        })

        # Optional MI diagnostics (does not affect scoring).
        if bool(getattr(args, "kv_group_mi_debug", False) or getattr(args, "runtime_metrics", False)):
            try:
                mi_diag = collect_mi_diagnostics_from_model(model, args.model_path)
                sample_result["mi_stats"] = mi_diag.get("mi_stats")  # layer-0 compat
                sample_result["mi_stats_by_layer"] = mi_diag.get("mi_stats_by_layer")
                sample_result["mi_stats_aggregate"] = mi_diag.get("mi_stats_aggregate")
            except Exception:
                sample_result["mi_stats"] = None
                sample_result["mi_stats_by_layer"] = None
                sample_result["mi_stats_aggregate"] = None

        if bool(getattr(args, "kv_group_cvi_debug", False) or getattr(args, "runtime_metrics", False)
                or getattr(args, "kv_group_cvi_recon_diag_enable", False)):
            try:
                cvi_diag = collect_cvi_diagnostics_from_model(model, args.model_path)
                sample_result["cvi_stats"] = cvi_diag.get("cvi_stats")
                sample_result["cvi_stats_by_layer"] = cvi_diag.get("cvi_stats_by_layer")
                sample_result["cvi_recon"] = cvi_diag.get("cvi_recon")
                sample_result["cvi_recon_by_layer"] = cvi_diag.get("cvi_recon_by_layer")
            except Exception:
                sample_result["cvi_stats"] = None
                sample_result["cvi_stats_by_layer"] = None
                sample_result["cvi_recon"] = None
                sample_result["cvi_recon_by_layer"] = None

        if bool(
            getattr(args, "kv_group_specialization_diag_enable", False)
            or getattr(args, "kv_group_per_group_base_only", False)
        ):
            try:
                gs = collect_group_specialization_from_model(model, args.model_path)
                sample_result["group_specialization"] = gs.get("group_specialization")
                sample_result["group_specialization_by_layer"] = gs.get(
                    "group_specialization_by_layer"
                )
                sample_result["per_group"] = gs.get("per_group")
                sample_result["per_group_by_layer"] = gs.get("per_group_by_layer")
            except Exception:
                sample_result["group_specialization"] = None
                sample_result["group_specialization_by_layer"] = None
                sample_result["per_group"] = None
                sample_result["per_group_by_layer"] = None

        results.append(sample_result)


    result_report = screenspotpro_evaluate(results)

    print("\n" + "=" * 80)
    print("SCREENSPOT PRO EVALUATION RESULTS")
    print("=" * 80)
    print(json.dumps(result_report["metrics"], indent=2, ensure_ascii=False))

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

        detailed_results_path = os.path.join(args.results_dir, 'screenspotpro_detailed_results.json')
        with open(detailed_results_path, 'w') as f:
            json.dump(result_report, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved ScreenSpot Pro detailed results to {detailed_results_path}")

        overall_metrics = result_report["metrics"]["overall"]
        if getattr(args, "runtime_metrics", False):
            RUNTIME_METRICS.mark_run_end()
            runtime_metrics_summary = collect_runtime_metrics()
        else:
            runtime_metrics_summary = None
        summary_results = {
            "model_family": "opencua" if "opencua" in args.model_path.lower() else "uitars",
            "model_path": args.model_path,
            "task": args.task,
            "total_samples": overall_metrics["num_total"],
            "overall_accuracy": overall_metrics["action_acc"],
            "text_accuracy": overall_metrics["text_acc"],
            "icon_accuracy": overall_metrics["icon_acc"],
            "wrong_format_samples": overall_metrics["wrong_format_num"],
            "kv_cache": args.kv_cache,
            "kv_cache_budget": args.kv_cache_budget,
            "kv_group_soft_prior_lambda": getattr(args, "kv_group_soft_prior_lambda", None),
            "kv_group_online_profile_decay": getattr(args, "kv_group_online_profile_decay", None),
            "kv_group_online_profile_tau": getattr(args, "kv_group_online_profile_tau", None),
            "kv_group_mi_mode": getattr(args, "kv_group_mi_mode", None),
            "kv_group_mi_reliability_gamma": getattr(args, "kv_group_mi_reliability_gamma", None),
            "kv_group_mi_reliability_center": getattr(args, "kv_group_mi_reliability_center", None),
            "kv_group_mi_reliability_temperature": getattr(
                args, "kv_group_mi_reliability_temperature", None
            ),
            "kv_group_mi_routing_tau": getattr(args, "kv_group_mi_routing_tau", None),
            "kv_group_mi_routing_normalization": getattr(
                args, "kv_group_mi_routing_normalization", None
            ),
            "kv_group_mi_candidate_multiplier": getattr(args, "kv_group_mi_candidate_multiplier", None),
            "kv_group_mi_positive_only": bool(getattr(args, "kv_group_mi_positive_only", True)),
            "kv_group_mi_max_multiplier": getattr(args, "kv_group_mi_max_multiplier", None),
            "kv_group_mi_cmi_boundary_w": getattr(args, "kv_group_mi_cmi_boundary_w", None),
            "kv_group_mi_cmi_alpha": getattr(args, "kv_group_mi_cmi_alpha", None),
            "kv_group_mi_cmi_beta": getattr(args, "kv_group_mi_cmi_beta", None),
            "kv_group_mi_cmi_clip": getattr(args, "kv_group_mi_cmi_clip", None),
            "kv_group_mi_cmi_margin": getattr(args, "kv_group_mi_cmi_margin", None),
            "kv_group_mi_cmi_max_swaps_frac": getattr(args, "kv_group_mi_cmi_max_swaps_frac", None),
            "kv_group_mi_cmi_cross_fit": getattr(args, "kv_group_mi_cmi_cross_fit", None),
            "kv_group_cvi_enable": bool(getattr(args, "kv_group_cvi_enable", False)),
            "kv_group_cvi_only": bool(getattr(args, "kv_group_cvi_only", False)),
            "kv_group_cvi_mi_gate": bool(getattr(args, "kv_group_cvi_mi_gate", False)),
            "kv_group_cvi_alpha": getattr(args, "kv_group_cvi_alpha", None),
            "kv_group_cvi_clip": getattr(args, "kv_group_cvi_clip", None),
            "kv_group_cvi_candidate_multiplier": getattr(args, "kv_group_cvi_candidate_multiplier", None),
            "kv_group_cvi_query_scope": getattr(args, "kv_group_cvi_query_scope", "all"),
            "kv_group_cvi_recon_diag_enable": bool(
                getattr(args, "kv_group_cvi_recon_diag_enable", False)
            ),
            "kv_group_per_group_base_only": bool(
                getattr(args, "kv_group_per_group_base_only", False)
            ),
            "kv_group_specialization_diag_enable": bool(
                getattr(args, "kv_group_specialization_diag_enable", False)
            ),
            "seed": 1234,
            "kv_group_online_profile_last_lam_eff": (
                getattr(model.model.language_model.layers[0].self_attn, "kv_group_online_profile_last_lam_eff", None)
                if "UI-TARS" in args.model_path
                else None
            ),
            "original_mi_debug": collect_original_mi_diagnostics_from_model(
                model, args.model_path
            ) if args.kv_cache == "starkv" else None,
            "kv_group_temporal_enable": bool(getattr(args, "kv_group_temporal_enable", False)),
            "kv_group_temporal_rho": getattr(args, "kv_group_temporal_rho", None),
            "kv_group_temporal_delta": getattr(args, "kv_group_temporal_delta", None),
            "kv_group_temporal_eps": getattr(args, "kv_group_temporal_eps", None),
            "kv_group_temporal_discount_min": getattr(args, "kv_group_temporal_discount_min", None),
            "kv_group_temporal_warmup_steps": getattr(args, "kv_group_temporal_warmup_steps", None),
            "kv_group_temporal_debug": bool(getattr(args, "kv_group_temporal_debug", False)),
            "enable_grd": bool(getattr(args, "enable_grd", False)),
            "grd_alpha": getattr(args, "grd_alpha", None),
            "grd_feature_dim": getattr(args, "grd_feature_dim", None),
            "alpha": getattr(args, "alpha", None),
            "temperature": getattr(args, "temperature", None),
            "attention_implementation": args.attention_implementation,
            "model_dtype": args.model_dtype,
            "num_chunks": args.num_chunks,
            "chunk_id": args.chunk_id,
            "chunk_start_idx": start_idx,
            "chunk_end_idx": end_idx,
            "num_tasks_this_chunk": len(tasks_to_run),
            "total_tasks_before_chunk": total_tasks,
            "max_new_tokens": args.max_new_tokens,
            "gpu_memory_stats": collect_gpu_memory_stats(),
            "kv_cache_stats": compute_kv_cache_memory_summary(collect_kv_cache_stats()),
            "runtime_metrics": runtime_metrics_summary,
            "temporal_discount_debug": (
                collect_temporal_diagnostics_from_model(model, args.model_path)
                if bool(getattr(args, "kv_group_temporal_enable", False)) and is_starkv_kv(args.kv_cache)
                else None
            ),
            "eval_duration_seconds": round(time.time() - eval_start_time, 2),
        }

        summary_results_path = os.path.join(args.results_dir, 'screenspotpro_summary_results.json')
        with open(summary_results_path, 'w') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved ScreenSpot Pro summary results to {summary_results_path}")
        # Read-only CVI reconstruction diagnostics (does not affect scoring).
        if bool(getattr(args, "kv_group_cvi_recon_diag_enable", False) or getattr(args, "kv_group_cvi_debug", False)):
            try:
                diag = _aggregate_cvi_recon_from_results(results)
                diag_path = os.path.join(args.results_dir, "cvi_diagnostics.json")
                with open(diag_path, "w") as f:
                    json.dump(diag, f, indent=2, ensure_ascii=False)
                md_path = os.path.join(args.results_dir, "cvi_diagnostics.md")
                with open(md_path, "w") as f:
                    f.write("# CVI diagnostics\n\n")
                    for k, v in diag.items():
                        f.write(f"- **{k}**: {v}\n")
                    f.write(
                        "\nExpected if CVI target is correct: "
                        "`frac_layers_recon_improved` ≫ 0.5 and "
                        "`recon_error_delta_mean` < 0.\n"
                    )
                logging.info(f"Saved CVI diagnostics to {diag_path}")
            except Exception as e:
                logging.warning(f"CVI diagnostics write failed: {e}")

        if bool(
            getattr(args, "kv_group_specialization_diag_enable", False)
            or getattr(args, "kv_group_per_group_base_only", False)
        ):
            try:
                gs_path_json, gs_path_md, gs_path_raw = _write_group_specialization_reports(
                    args.results_dir, results, summary_results
                )
                logging.info(f"Saved group specialization diagnostics to {gs_path_json}")
            except Exception as e:
                logging.warning(f"Group specialization diagnostics write failed: {e}")

        print(f"\nScreenSpot Pro Summary:")
        print(f"{'Metric':<25} {'Value':<15}")
        print("-" * 40)
        print(f"{'Overall Accuracy':<25} {overall_metrics['action_acc']*100:<15.2f}%")
        print(f"{'Text Accuracy':<25} {overall_metrics['text_acc']*100:<15.2f}%")
        print(f"{'Icon Accuracy':<25} {overall_metrics['icon_acc']*100:<15.2f}%")
        print(f"{'Total Samples':<25} {overall_metrics['num_total']:<15}")
        print(f"{'Wrong Format':<25} {overall_metrics['wrong_format_num']:<15}")
        print("-" * 40)

    print("=" * 80)
