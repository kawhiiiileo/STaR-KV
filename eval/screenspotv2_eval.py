import torch
import ast
import json
import time
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from multiprocessing import freeze_support

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info
from opencua_utils import opencua_parse_action, analyze_vision_tokens_opencua_multi_images
from ui_tars_utils import (
    parse_action_to_structure_output, MIN_PIXELS, MAX_PIXELS, IMAGE_FACTOR,
    analyze_vision_tokens_multi_images
)
from attention_helpers import (
    replace_qwen2_5_vl,
    replace_opencua,
    set_attention_implementation,
    configure_accelerate_skip_attention,
    set_kv_cache_budget,
    set_move_attention_to_cpu,
    set_vision_start_idx,
    set_vision_end_idx,
    set_temperature,
    set_alpha,
    set_window_size,
    apply_entropy_budget_runtime,
    collect_entropy_budget_stats,
    reset_gpu_memory_stats,
    collect_gpu_memory_stats,
    reset_starkv_per_sample_state,
    reset_kv_cache_stats,
    collect_kv_cache_stats,
    compute_kv_cache_memory_summary,
    set_starkv_group_config,
    is_starkv_kv,
)
from attention_helpers import add_starkv_kv_arguments, finalize_starkv_args
from eval_paths import (
    add_model_path_argument,
    add_results_dir_argument,
    add_screenspot_v2_dataset_args,
    validate_required_paths,
)

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""

# Single-image benchmark: temporal discount is a no-op when num_visual_spans <= 1.

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)


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


if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser()
    add_model_path_argument(parser)
    add_screenspot_v2_dataset_args(parser)
    parser.add_argument('--task', type=str, required=True, choices=["all"])
    parser.add_argument('--debug', default=None, type=int)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/mps/cpu). If not specified, will auto-detect best available device.')
    parser.add_argument('--model_dtype', type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help='Data type to use (auto/bfloat16/float16/float32).')
    parser.add_argument('--attention_implementation', type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help='Attention implementation to use (eager/flash_attention_2).')
    add_starkv_kv_arguments(
        parser,
        kv_cache_budget_default=100,
        kv_cache_budget_type=int,
        temperature_default=1.0,
        alpha_default=None,
        include_mi_granularity=True,
    )
    add_results_dir_argument(parser)
    args = parser.parse_args()
    validate_required_paths(args, ("model_path", "screenspot_imgs", "screenspot_test"))
    finalize_starkv_args(args, disable_starkv_extras_for_original=True)

    print(
        "[ScreenSpot-v2 config]",
        f"kv_cache={args.kv_cache}",
        f"kv_group_selection_mode={args.kv_group_selection_mode}",
        f"kv_cache_budget={args.kv_cache_budget}",
        f"kv_group_soft_prior_lambda={args.kv_group_soft_prior_lambda}",
        f"kv_group_soft_prior_source={args.kv_group_soft_prior_source}",
        f"kv_group_online_profile_steps={args.kv_group_online_profile_steps}",
        f"temporal_enable={bool(getattr(args, 'kv_group_temporal_enable', False))}",
        f"alpha={args.alpha}",
        f"temperature={args.temperature}",
        f"window_size={args.window_size}",
        f"attention_implementation={args.attention_implementation}",
        f"model_dtype={args.model_dtype}",
        f"max_new_tokens={args.max_new_tokens}",
        flush=True,
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

    model_path = args.model_path
    print("model_path: ", model_path)
    if "OpenCUA" in args.model_path:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
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

    if "OpenCUA" in args.model_path:
        replace_opencua(kv_cache_mode=args.kv_cache)
        print("Replaced OpenCUA attention with custom implementation for CPU memory management")
    else:
        replace_qwen2_5_vl(kv_cache_mode=args.kv_cache)
        print("Replaced Qwen2.5-VL attention with custom implementation for CPU memory management")

    if device == "cpu":
        if "OpenCUA" in args.model_path:
            model = AutoModel.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu",
                attn_implementation=args.attention_implementation,
                trust_remote_code=True
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu",
                attn_implementation=args.attention_implementation,
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if is_starkv_kv(args.kv_cache):
            set_starkv_group_config(model, args)
        apply_entropy_budget_runtime(model, args)
        reset_gpu_memory_stats()
        reset_kv_cache_stats()
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            configure_accelerate_skip_attention(model)
    else:
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            device_map = {"": "cuda:0"}

        if "OpenCUA" in args.model_path:
            model = AutoModel.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map,
                attn_implementation=args.attention_implementation,
                trust_remote_code=True
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map,
                attn_implementation=args.attention_implementation,
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if is_starkv_kv(args.kv_cache):
            set_starkv_group_config(model, args)
        apply_entropy_budget_runtime(model, args)
        reset_gpu_memory_stats()
        reset_kv_cache_stats()
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            configure_accelerate_skip_attention(model)

    print("Load Success")

    tasks = ["mobile", "desktop", "web"]
    tasks_result = []
    result = []
    eval_start_time = time.time()
    for task in tasks:
        dataset = "screenspot_" + task + "_v2.json"
        screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
        if args.debug is not None:
            screenspot_data = screenspot_data[:args.debug]
            print(f"Num of sample: {len(screenspot_data)} (limited to {args.debug} for quick evaluation)")
        else:
            print("Num of sample: " + str(len(screenspot_data)))

        num_action = 0
        corr_action = 0
        text_correct = []
        icon_correct = []
        num_wrong_format = 0
        for j, item in tqdm(enumerate(screenspot_data), desc=f"Processing {task} data", total=len(screenspot_data)):
            num_action += 1
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)
            if not os.path.exists(img_path):
                logging.warning("Image not found, skipping: %s", img_path)
                num_action -= 1
                continue
            image = Image.open(img_path)
            instruction = item["instruction"]

            bbox = item["bbox"]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            img_size = image.size
            img_width, img_height = img_size
            bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]

            if "OpenCUA" in args.model_path:
                SYSTEM_PROMPT = (
                    "You are a GUI agent. You are given a task and a screenshot of the screen. "
                    "You need to perform a series of pyautogui actions to complete the task."
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": SYSTEM_PROMPT + "\n" + instruction},
                        ],
                    },
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(device)
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": GROUNDING_DOUBAO.format(instruction=instruction)},
                        ],
                    }
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

            if is_starkv_kv(args.kv_cache):
                if "UI-TARS" in args.model_path:
                    vision_analysis = analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=1)
                elif "OpenCUA" in args.model_path:
                    vision_analysis = analyze_vision_tokens_opencua_multi_images(tokenizer, inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"], merge_size=2, image_count=1)

            set_window_size(model, args)
            if is_starkv_kv(args.kv_cache):
                set_vision_start_idx(model, vision_analysis['vision_start_idx'], args)
                set_vision_end_idx(model, vision_analysis['vision_end_idx'], args)
                if args.alpha is not None:
                    set_alpha(model, args)
                    set_temperature(model, args)

            reset_starkv_per_sample_state(model, args)

            try:
                if "OpenCUA" in args.model_path:
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        do_sample=False,
                        return_dict_in_generate=False
                    )
                    prompt_len = inputs["input_ids"].shape[1]
                    generated_ids = generated_ids[:, prompt_len:]
                    output_text = tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                else:
                    outputs = model.generate(**inputs,
                                            max_new_tokens=args.max_new_tokens,
                                            pad_token_id=processor.tokenizer.eos_token_id,
                                            output_attentions=False,
                                            use_cache=True,
                                            do_sample=False,
                                            return_dict_in_generate=True)
                    generated_ids = outputs if not hasattr(outputs, 'sequences') else outputs.sequences
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
            except Exception as e:
                print(f"Error in generation: {e}")
                print("Using outputs from the previous iteration")
                if "OpenCUA" not in args.model_path:
                    generated_ids = outputs if not hasattr(outputs, 'sequences') else outputs.sequences
                continue

            print("output_text: ", output_text)

            try:
                if "OpenCUA" in args.model_path:
                    parsed_actions = opencua_parse_action(output_text,
                                origin_resized_height=img_height,
                                origin_resized_width=img_width,
                                max_pixels=MAX_PIXELS,
                                min_pixels=MIN_PIXELS,
                                factor=IMAGE_FACTOR,
                                model_type="qwen25vl")
                    click_point = parsed_actions[0]["coordinate"]
                else:
                    parsed_actions = parse_action_to_structure_output(output_text,
                        origin_resized_height=img_height,
                        origin_resized_width=img_width,
                        max_pixels=MAX_PIXELS,
                        min_pixels=MIN_PIXELS,
                        factor=IMAGE_FACTOR,
                        model_type="qwen25vl")[0]
                    click_point = list(parsed_actions["action_inputs"].values())[0]
                    click_point = ast.literal_eval(click_point)

                if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                    corr_action += 1
                    if item["data_type"] == 'text':
                        text_correct.append(1)
                    else:
                        icon_correct.append(1)
                else:
                    if item["data_type"] == 'text':
                        text_correct.append(0)
                    else:
                        icon_correct.append(0)

                result.append({"img_path": img_path, "text": instruction, "bbox": bbox, "pred": click_point,
                               "type": item["data_type"], "source": item["data_source"]})
            except Exception as e:
                print(output_text)
                print(e)
                num_wrong_format += 1
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("Step: " + str(j) + " wrong format")

        action_acc = corr_action / num_action
        text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
        icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0

        logging.info("=" * 60)
        logging.info(f"Task Results:")
        logging.info(f"  Action Accuracy: {action_acc:.4f} ({corr_action}/{num_action})")
        logging.info(f"  Text Accuracy:   {text_acc:.4f} ({sum(text_correct)}/{len(text_correct)})")
        logging.info(f"  Icon Accuracy:   {icon_acc:.4f} ({sum(icon_correct)}/{len(icon_correct)})")
        logging.info(f"  Wrong Format:    {num_wrong_format}")
        logging.info("=" * 60)

        tasks_result.append([text_acc, icon_acc])

    logging.info("\n" + "=" * 80)
    logging.info("FINAL EVALUATION RESULTS")
    logging.info("=" * 80)

    if len(tasks_result) > 0:
        logging.info(f"\nResults by Task:")
        logging.info(f"{'Task':<15} {'Text Acc':<12} {'Icon Acc':<12} {'Combined':<12}")
        logging.info("-" * 51)
        for i, (text_acc, icon_acc) in enumerate(tasks_result):
            combined_acc = (text_acc + icon_acc) / 2
            logging.info(f"{'Task ' + str(i+1):<15} {text_acc*100:<12.2f}% {icon_acc*100:<12.2f}% {combined_acc*100:<12.2f}%")
        logging.info("-" * 51)

    all_text_accs = [result[0] for result in tasks_result]
    all_icon_accs = [result[1] for result in tasks_result]

    avg_text_acc = sum(all_text_accs) / len(all_text_accs) if all_text_accs else 0
    avg_icon_acc = sum(all_icon_accs) / len(all_icon_accs) if all_icon_accs else 0
    overall_acc = (avg_text_acc + avg_icon_acc) / 2

    os.makedirs(args.results_dir, exist_ok=True)

    detailed_results_path = os.path.join(args.results_dir, 'detailed_results.json')
    with open(detailed_results_path, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved detailed results to {detailed_results_path}")

    temporal_discount_debug = None
    if "UI-TARS" in args.model_path:
        try:
            sa = model.model.language_model.layers[0].self_attn
            temporal_discount_debug = getattr(sa, "kv_group_temporal_last_stats", None)
        except Exception:
            temporal_discount_debug = None

    summary_results = {
        "model_path": args.model_path,
        "task": args.task,
        "debug": args.debug,
        "total_samples": len(result),
        "overall_accuracy": overall_acc,
        "text_accuracy": avg_text_acc,
        "icon_accuracy": avg_icon_acc,
        "kv_cache": args.kv_cache,
        "kv_cache_budget": args.kv_cache_budget,
        "kv_group_selection_mode": getattr(args, "kv_group_selection_mode", "soft_global"),
        "kv_group_soft_prior_lambda": getattr(args, "kv_group_soft_prior_lambda", None),
        "kv_group_soft_prior_source": getattr(args, "kv_group_soft_prior_source", None),
        "kv_group_mi_saliency_weight": getattr(args, "kv_group_mi_saliency_weight", None),
        "kv_group_token_spatial_beta": getattr(args, "kv_group_token_spatial_beta", None),
        "kv_group_token_spatial_scale": getattr(args, "kv_group_token_spatial_scale", None),
        "kv_group_online_profile_steps": getattr(args, "kv_group_online_profile_steps", None),
        "kv_group_online_profile_decay": getattr(args, "kv_group_online_profile_decay", None),
        "kv_group_online_profile_tau": getattr(args, "kv_group_online_profile_tau", None),
        "kv_group_online_profile_lambda_ramp_steps": getattr(args, "kv_group_online_profile_lambda_ramp_steps", None),
        "kv_group_temporal_enable": bool(getattr(args, "kv_group_temporal_enable", False)),
        "kv_group_temporal_rho": getattr(args, "kv_group_temporal_rho", None),
        "kv_group_temporal_delta": getattr(args, "kv_group_temporal_delta", None),
        "kv_group_temporal_eps": getattr(args, "kv_group_temporal_eps", None),
        "kv_group_temporal_discount_min": getattr(args, "kv_group_temporal_discount_min", None),
        "kv_group_temporal_warmup_steps": getattr(args, "kv_group_temporal_warmup_steps", None),
        "kv_group_temporal_debug": bool(getattr(args, "kv_group_temporal_debug", False)),
        "temporal_discount_debug": temporal_discount_debug,
        "entropy_budget_enable": bool(getattr(args, "kv_entropy_budget_enable", False)),
        "entropy_budget_active": bool(getattr(args, "kv_entropy_budget_enable", False) and is_starkv_kv(getattr(args, "kv_cache", None))),
        "entropy_budget_skipped_reason": (
            "AEB applies only when --kv_cache starkv."
            if (bool(getattr(args, "kv_entropy_budget_enable", False)) and not is_starkv_kv(getattr(args, "kv_cache", None)))
            else None
        ),
        "gpu_memory_stats": collect_gpu_memory_stats(),
        "kv_cache_stats": compute_kv_cache_memory_summary(collect_kv_cache_stats()),
        "eval_duration_seconds": round(time.time() - eval_start_time, 2),
        "entropy_budget_debug": collect_entropy_budget_stats(model, args),
        "screenspot_temporal_note": "Single-image benchmark; temporal discount is a no-op (num_visual_spans <= 1).",
        "alpha": getattr(args, "alpha", None),
        "temperature": getattr(args, "temperature", None),
        "window_size": getattr(args, "window_size", None),
        "max_new_tokens": args.max_new_tokens,
        "attention_implementation": args.attention_implementation,
        "model_dtype": args.model_dtype,
        "task_breakdown": []
    }

    for i, (text_acc, icon_acc) in enumerate(tasks_result):
        task_name = tasks[i] if i < len(tasks) else f"task_{i+1}"
        combined_acc = (text_acc + icon_acc) / 2
        summary_results["task_breakdown"].append({
            "task": task_name,
            "text_accuracy": text_acc,
            "icon_accuracy": icon_acc,
            "combined_accuracy": combined_acc
        })

    summary_results_path = os.path.join(args.results_dir, 'summary_results.json')
    with open(summary_results_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    logging.info(f"Saved summary results to {summary_results_path}")

    logging.info(f"\nOverall Summary:")
    logging.info(f"{'Metric':<20} {'Accuracy':<10}")
    logging.info("-" * 30)
    logging.info(f"{'Text Average':<20} {avg_text_acc*100:<10.2f}%")
    logging.info(f"{'Icon Average':<20} {avg_icon_acc*100:<10.2f}%")
    logging.info(f"{'Overall Average':<20} {overall_acc*100:<10.2f}%")
    logging.info("-" * 30)

    logging.info("=" * 80)
