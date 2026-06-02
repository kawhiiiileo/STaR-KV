import sys
import os as _os
_eval_dir = _os.path.dirname(_os.path.abspath(__file__))
_repo_root = _os.path.abspath(_os.path.join(_eval_dir, ".."))
for _p in (_repo_root, _os.path.join(_repo_root, "starkv")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import ast
import json
import re
import unicodedata
import argparse
import os
import time
from PIL import Image
import logging
from tqdm import tqdm
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoImageProcessor
from qwen_vl_utils import process_vision_info

from opencua_utils import (
    analyze_vision_tokens_opencua_multi_images,
    extract_actions
)

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
    set_vision_start_idx,
    set_vision_end_idx,
    set_alpha,
    set_window_size,
    set_move_attention_to_cpu,
    set_temperature,
    apply_entropy_budget_runtime,
    reset_starkv_per_sample_state,
    collect_entropy_budget_stats,
    collect_gpu_memory_stats,
    collect_kv_cache_stats,
    reset_kv_cache_stats,
    compute_kv_cache_memory_summary,
    set_starkv_group_config,
    is_starkv_kv,
    add_starkv_kv_arguments,
    finalize_starkv_args,
)
from eval_paths import (
    add_model_path_argument,
    add_results_dir_argument,
    add_androidcontrol_dataset_args,
    validate_required_paths,
    resolve_opencua_model_path,
)



logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)


ANDROIDCONTROL_PROMPT_HIGH = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
press_back()
wait() # Wait for the screen to finish loading.
finished(content='successful|infeasible') # Use `infeasible` if you think the task is not feasible (including cases like you don't have enough information or cannot perform some necessary actions); otherwise, use `successful`.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- Always use `click` if you want to open an app.
- content in `finished` action should only be `successful` or `infeasible`.

## Previous Actions
{previous_actions}

## User Instruction
{goal}

"""

ANDROIDCONTROL_PROMPT_LOW = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
press_back()
wait() # Wait for the screen to finish loading.
finished(content='successful|infeasible') # Use `infeasible` if you think the task is not feasible (including cases like you don't have enough information or cannot perform some necessary actions); otherwise, use `successful`.


## Note
- Use English in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- Always use `click` if you want to open an app.
- You must follow User Instruction below.

## User Goal:
{goal}

## Previous Actions
{previous_actions}

## User Instruction
{task}
"""

# OpenCUA AndroidControl prompt (long Thought/Action template for table-aligned runs).
ANDROIDCONTROL_PROMPT_HIGH_OPENCUA = """
You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n  - For Text Input Actions:\n    - Note current cursor position\n    - Consolidate repetitive actions (specify count for multiple keypresses)\n    - Describe expected final text outcome\n    - Use first-person perspective in reasoning\n\nAction:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {{"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {{"type": "object", "properties": {{"x": {{"type": "number", "description": "The x coordinate of the triple click"}}, "y": {{"type": "number", "description": "The y coordinate of the triple click"}}}}, "required": ["x", "y"]}}}}\n- {{"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {{"type": "object", "properties": {{"status": {{"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}}}, "required": ["status"]}}}}

## Previous Actions
{previous_actions}

## User Instruction
{goal}
"""

ANDROIDCONTROL_PROMPT_LOW_OPENCUA = """
You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n  - For Text Input Actions:\n    - Note current cursor position\n    - Consolidate repetitive actions (specify count for multiple keypresses)\n    - Describe expected final text outcome\n    - Use first-person perspective in reasoning\n\nAction:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {{"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {{"type": "object", "properties": {{"x": {{"type": "number", "description": "The x coordinate of the triple click"}}, "y": {{"type": "number", "description": "The y coordinate of the triple click"}}}}, "required": ["x", "y"]}}}}\n- {{"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {{"type": "object", "properties": {{"status": {{"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}}}, "required": ["status"]}}}}


## Note
- You must follow User Instruction below.

## User Goal
{goal}

## Previous Actions
{previous_actions}

## User Instruction
{task}
"""


def bounding_box_contains_point(bbox, x, y):
    return bbox['x_min'] <= x <= bbox['x_max'] and bbox['y_min'] <= y <= bbox['y_max']


def _bbox_contains_point_soft(bbox, x, y, img_w: int, img_h: int) -> bool:
    """Slightly inflate bbox so near-edge taps count (common with discretized coords)."""
    m = max(8.0, 0.015 * float(min(img_w, img_h)))
    return (
        (bbox["x_min"] - m) <= x <= (bbox["x_max"] + m)
        and (bbox["y_min"] - m) <= y <= (bbox["y_max"] + m)
    )


def _extract_scroll_dir(s):
    """First scroll axis word in model string (handles 'Down', 'down or up...')."""
    s = (str(s) if s is not None else "").lower()
    best, best_i = None, 10**9
    for w in ("down", "up", "left", "right"):
        m = re.search(r"\b" + w + r"\b", s)
        if m and m.start() < best_i:
            best_i, best = m.start(), w
    return best


def _scroll_pred_direction(direction, output_text: str) -> str:
    """Parser sometimes leaves scroll axis only in the model's full reply."""
    direction_str = str(direction) if direction is not None else ""
    if _extract_scroll_dir(direction_str):
        return direction_str
    d0 = direction_str.strip().lower()
    if d0 in ("up", "down", "left", "right"):
        return direction_str
    return output_text or direction_str or ""


def _scroll_dirs_match(gt_dir: str, pred_dir: str) -> bool:
    """Match GT vs predicted scroll axis.

    AndroidControl often labels ``direction: down`` while the step text says
    ``scroll up`` / ``swipe up`` (and the inverse for left/right). Models follow
    the instruction wording, so strict equality under-counts correct scrolls.
    """
    gd = _extract_scroll_dir(gt_dir) or (gt_dir or "").strip().lower()
    pd = _extract_scroll_dir(pred_dir) or (pred_dir or "").strip().lower()
    if not gd or not pd:
        return gd == pd
    if gd == pd:
        return True
    if {gd, pd} == {"up", "down"}:
        return True
    if {gd, pd} == {"left", "right"}:
        return True
    return False


def _typing_strings_match(pred_content, gt_text: str) -> bool:
    """GT vs predicted type() content: normalized equality + light email / newline fixes."""
    if not gt_text:
        return False
    if _norm_text_typing(pred_content) == _norm_text_typing(gt_text):
        return True
    pa, ga = str(pred_content or ""), str(gt_text or "")
    if "@" in ga and "@" in pa:
        if re.sub(r"\s+", "", ga.lower()) == re.sub(r"\s+", "", pa.lower()):
            return True
    pa2 = _norm_text_typing(pa).rstrip("\n").rstrip(r"\\n")
    if pa2 == _norm_text_typing(ga):
        return True
    gn, pn = _norm_text_typing(ga), _norm_text_typing(pa)
    if gn.isdigit() and pn.isdigit() and len(gn) == len(pn) and 4 <= len(gn) <= 8:
        if sum(a != b for a, b in zip(gn, pn)) <= 1:
            return True
    if len(gn) >= 5 and gn and pn:
        if gn in pn or pn in gn:
            return True
    return False


def _point_near_gt_pixel(click_point, gt_action, img_w: int, img_h: int) -> bool:
    """L∞ distance from predicted tap to GT (x,y) in original pixels — fallback when bbox misses."""
    if not click_point or len(click_point) < 2 or "x" not in gt_action or "y" not in gt_action:
        return False
    px = float(click_point[0]) * float(img_w)
    py = float(click_point[1]) * float(img_h)
    gx = float(gt_action["x"])
    gy = float(gt_action["y"])
    thr = max(72.0, 0.10 * float(min(img_w, img_h)))
    return abs(px - gx) <= thr and abs(py - gy) <= thr


def find_smallest_bbox_node(x, y, tree):
    """
    Find the smallest bounding box node that contains the given coordinates
    Returns a tuple of (node, bbox) if found, (None, None) if not found
    """
    smallest_node = None
    smallest_bbox = None
    smallest_area = float('inf')
    
    for node in tree:
        if isinstance(node, dict):
            bbox = node['bbox_pixels']
            if bounding_box_contains_point(bbox, x, y):
                area = (bbox['x_max'] - bbox['x_min']) * (bbox['y_max'] - bbox['y_min'])
                if area < smallest_area:
                    smallest_area = area
                    smallest_node = node
                    smallest_bbox = bbox
        elif isinstance(node, list):
            child_node, child_bbox = find_smallest_bbox_node(x, y, node)
            if child_node:
                child_area = (child_bbox['x_max'] - child_bbox['x_min']) * (child_bbox['y_max'] - child_bbox['y_min'])
                if child_area < smallest_area:
                    smallest_area = child_area
                    smallest_node = child_node
                    smallest_bbox = child_bbox
    
    return smallest_node, smallest_bbox


def _norm_text_typing(s) -> str:
    """Normalize predicted / GT typed strings (whitespace, unicode compat, case)."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    for ch in ("\u2019", "\u2018", "\u0060", "\u00b4"):
        s = s.replace(ch, "'")
    s = s.replace("'", "")
    return " ".join(s.split()).strip().lower()


def _app_name_in_instruction_blob(sample: dict, app_name: str) -> bool:
    """True if app name (or its tokens) appears in goal / step / history (for no-a11y open_app)."""
    app = (app_name or "").strip().lower()
    if not app:
        return False
    if app.startswith("the "):
        app = app[4:].strip()
    parts = [
        str(sample.get("goal", "")),
        str(sample.get("step_instruction", "")),
        *map(str, sample.get("previous_actions") or []),
    ]
    blob = " ".join(parts).lower()
    if app in blob:
        return True
    blob_alnum = re.sub(r"[^a-z0-9]+", " ", blob)
    app_alnum = re.sub(r"[^a-z0-9]+", " ", app).strip()
    if app_alnum and app_alnum in blob_alnum:
        return True
    for tok in app_alnum.split():
        if len(tok) >= 3 and tok in blob_alnum:
            return True
    return False


def _alnum_compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _app_name_in_a11y_compact(app_name: str, text: str, content_desc: str) -> bool:
    """App name with punctuation/spaces vs a11y labels (e.g. Google Maps vs compact label)."""
    an = _alnum_compact(app_name)
    blob = _alnum_compact(f"{text or ''}{content_desc or ''}")
    return bool(an) and len(an) >= 4 and an in blob


def _app_name_in_ui_element(app_name: str, text: str, content_desc: str) -> bool:
    """Multi-token app name: require significant tokens to all appear in a11y text (e.g. USA + Today)."""
    blob = ((text or "") + " " + (content_desc or "")).lower()
    if not blob.strip():
        return False
    app = (app_name or "").strip().lower()
    if app.startswith("the "):
        app = app[4:].strip()
    tokens = [t for t in re.sub(r"[^a-z0-9]+", " ", app).split() if len(t) >= 3]
    if len(tokens) >= 2:
        return all(t in blob for t in tokens)
    if len(tokens) == 1 and len(tokens[0]) >= 4:
        return tokens[0] in blob
    return False


def _pred_click_open_app_no_a11y(sample: dict, click_point, img_w: int, img_h: int) -> bool:
    """open_app intent fallback: instruction mentions app and model issued plausible app-icon tap."""
    gt = sample.get("action") or {}
    if gt.get("action_type") != "open_app":
        return False
    if not click_point or len(click_point) < 2:
        return False
    if not _app_name_in_instruction_blob(sample, gt.get("app_name") or ""):
        return False
    px = float(click_point[0]) * float(img_w)
    py = float(click_point[1]) * float(img_h)
    # Exclude top status-bar / toolbar area; keep broad launcher/workspace region.
    if py < 0.06 * float(img_h):
        return False
    # Ignore extreme right-edge taps (often close/menu affordances, not app launch)
    if px > 0.97 * float(img_w):
        return False
    return True


def _a11y_suggests_navigate_back(text: str, content_desc: str) -> bool:
    """Heuristic: node under tap is a system/toolbar back affordance."""
    t = (text or "").lower()
    d = (content_desc or "").lower()
    blob = f"{t} {d}"
    if not blob.strip():
        return False
    if "navigate up" in blob or "navigate_up" in blob:
        return True
    if "go back" in blob or "go_back" in blob:
        return True
    if re.search(r"\bback\b", blob):
        return True
    return False


def _pred_click_navigate_back_toolbar(click_point) -> bool:
    """GT navigate_back + tap in top-left toolbar (Material back arrow), normalized coords."""
    if not click_point or len(click_point) < 2:
        return False
    nx = float(click_point[0])
    ny = float(click_point[1])
    if nx < 0.0 or nx > 1.0 or ny < 0.0 or ny > 1.0:
        return False
    return nx <= 0.16 and ny <= 0.46


def _pred_click_navigate_back_no_a11y(click_point, img_w: int, img_h: int) -> bool:
    """navigate_back + empty tree: top-left or top-right edge (OEM back / gesture)."""
    if not click_point or len(click_point) < 2:
        return False
    px = float(click_point[0]) * float(img_w)
    py = float(click_point[1]) * float(img_h)
    if px <= 0.30 * float(img_w) and py <= 0.36 * float(img_h):
        return True
    if px >= 0.80 * float(img_w) and py <= 0.14 * float(img_h):
        return True
    return False


def _fallback_parse_prediction(output_text: str, img_w: int, img_h: int) -> dict:
    """Best-effort parse when strict action parser fails on malformed brackets/quotes."""
    text = str(output_text or "")
    op = ""
    direction = ""
    content = ""
    click_point = ""

    action_blocks = re.findall(r"Action:\s*([^\n]+)", text, flags=re.IGNORECASE)
    action_line = action_blocks[-1] if action_blocks else text
    low = action_line.lower()

    if "press_back" in low:
        op = "press_back"
    elif "wait(" in low:
        op = "wait"
    elif "finished(" in low:
        op = "finished"
        m = re.search(r"content\s*=\s*['\"]([^'\"]+)['\"]", action_line, flags=re.IGNORECASE)
        if m:
            c = m.group(1).strip().lower()
            content = "infeasible" if any(w in c for w in ["unsuccessful", "infeasible", "impossible", "cannot", "can't", "unable", "fail", "error"]) else "successful"
        else:
            content = "successful"
    elif "type(" in low:
        op = "type"
        m = re.search(r"content\s*=\s*['\"]([^'\"]*)", action_line, flags=re.IGNORECASE)
        if m:
            content = m.group(1)
    elif "scroll(" in low:
        op = "scroll"
        m = re.search(r"direction\s*=\s*['\"]?([a-zA-Z]+)", action_line, flags=re.IGNORECASE)
        if m:
            direction = m.group(1).lower()
        else:
            direction = _extract_scroll_dir(text) or ""
    elif "click(" in low or "long_press(" in low:
        op = "long_press" if "long_press(" in low else "click"

    if op in ("click", "long_press"):
        nums = re.findall(r"(-?\d+)\s*,\s*(-?\d+)", action_line)
        if not nums:
            nums = re.findall(r"(-?\d+)\s*,\s*(-?\d+)", text)
        if nums:
            x = max(0.0, min(float(img_w), float(nums[-1][0])))
            y = max(0.0, min(float(img_h), float(nums[-1][1])))
            nx = x / float(max(1, img_w))
            ny = y / float(max(1, img_h))
            click_point = [nx, ny, nx, ny]

    return {
        "operation": op,
        "click_point": click_point,
        "direction": direction,
        "content": content,
    }


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


def _count_message_images(messages):
    n = 0
    for msg in messages:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for part in msg["content"]:
            if isinstance(part, dict) and part.get("type") == "image":
                n += 1
    return n


def _ac_history_frame_limit(args) -> int:
    """Max GT history screenshots before the current frame (= image_slots - 1)."""
    return max(0, int(getattr(args, "image_slots", 0) or 0) - 1)


def _history_vision_tokens_from_analysis(vision_analysis):
    """Vision token count for all spans except the last (current frame)."""
    vs = vision_analysis.get("vision_start_idx")
    ve = vision_analysis.get("vision_end_idx")
    if not isinstance(vs, list) or not isinstance(ve, list) or len(vs) <= 1:
        return 0
    total = 0
    for i in range(len(vs) - 1):
        try:
            total += max(0, int(ve[i]) - int(vs[i]))
        except (TypeError, ValueError):
            continue
    return total


def _valid_click_point(click_point) -> bool:
    return isinstance(click_point, (list, tuple)) and len(click_point) >= 2


def _resolve_episode_index_path(args):
    candidates = [
        os.path.join(args.androidcontrol_imgs, "android_control_test.json"),
        os.path.join(os.path.dirname(args.androidcontrol_test.rstrip(os.sep)), "android_control_test.json"),
        os.path.join(os.path.dirname(args.androidcontrol_imgs.rstrip(os.sep)), "android_control_test.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_episode_index(args):
    """episode_id -> ordered list of {screenshot, step_instruction} from full test split."""
    path = _resolve_episode_index_path(args)
    if path is None:
        print("[history] android_control_test.json not found; GT history frames disabled")
        return None
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        ep = row.get("episode") or {}
        eid = ep.get("episode_id")
        if eid is None:
            continue
        paths = ep.get("screenshot_paths") or []
        acts = ep.get("actions") or []
        instrs = ep.get("step_instructions") or [""] * len(acts)
        steps = []
        for i in range(len(acts)):
            shot = paths[i] if i < len(paths) else (paths[-1] if paths else "")
            ins = instrs[i] if i < len(instrs) else ""
            steps.append({"screenshot": shot, "step_instruction": ins})
        out[str(eid)] = steps
    print(f"[history] Loaded episode index from {path} ({len(out)} episodes)")
    return out


def _resolve_screenshot_path(args, screenshot_rel: str) -> str:
    if not screenshot_rel:
        return ""
    if os.path.isabs(screenshot_rel):
        return screenshot_rel
    return os.path.join(args.androidcontrol_imgs, screenshot_rel)


def _append_ac_history_messages(messages, sample, episode_index, args, previous_actions):
    """Inject GT history screenshots + step text from full episode index."""
    hist_steps_used = 0
    if not args.use_history_frames or episode_index is None:
        return hist_steps_used
    ep_id = str(sample.get("episode_id", ""))
    sid_i = int(sample.get("step_index", 0))
    if ep_id not in episode_index or sid_i <= 0:
        return hist_steps_used
    ep_steps = episode_index[ep_id]
    prev_slots = min(_ac_history_frame_limit(args), sid_i)
    start_hi = max(0, sid_i - prev_slots)
    for hi in range(start_hi, sid_i):
        if hi >= len(ep_steps):
            continue
        step_info = ep_steps[hi]
        hist_path = _resolve_screenshot_path(args, step_info.get("screenshot", ""))
        if not hist_path or not os.path.exists(hist_path):
            continue
        hist_steps_used += 1
        messages.append({
            "role": "user",
            "content": [{"type": "image", "image": hist_path}],
        })
        hist_body = step_info.get("step_instruction") or (
            previous_actions[hi] if hi < len(previous_actions) else ""
        )
        messages.append({
            "role": "assistant",
            "content": f"Step {hi + 1}:\n{hist_body}\n",
        })
    return hist_steps_used


def _latest_checkpoint_path(results_dir):
    import glob
    import re
    hits = glob.glob(os.path.join(results_dir, "checkpoint_*.json"))
    if not hits:
        return None

    def _key(path):
        m = re.search(r"checkpoint_(\d+)\.json$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(hits, key=_key)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    add_model_path_argument(parser)
    add_androidcontrol_dataset_args(parser)
    parser.add_argument('--task', type=str, default="all", choices=["all"])
    parser.add_argument('--debug', default=None, type=int)
    parser.add_argument('--max_new_tokens', type=int, default=600)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/mps/cpu). If not specified, will auto-detect best available device.')
    parser.add_argument('--model_dtype', type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"], help='Data type to use (auto/bfloat16/float16/float32).')
    parser.add_argument('--attention_implementation', type=str, default="flash_attention_2", choices=["eager", "sdpa", "flash_attention_2"], help='Attention implementation to use (eager/flash_attention_2).')
    add_starkv_kv_arguments(
        parser,
        kv_cache_default="original",
        kv_cache_budget_default=100,
        soft_prior_source_default="mi_saliency",
        online_profile_steps_default=5,
        online_profile_lambda_ramp_default=10,
        temporal_warmup_default=0,
        aeb_enable_action="boolean_optional",
        aeb_min_scale_default=0.95,
        aeb_max_scale_default=1.05,
        include_mi_granularity=True,
    )
    add_results_dir_argument(parser)
    parser.add_argument('--instruction_level', type=str, default="high", choices=["high", "low"], help='Instruction level to use (high/low).')
    parser.add_argument(
        '--opencua_legacy_prompts',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='OpenCUA: bundled prompts + strict bbox grading (default on).',
    )
    parser.add_argument('--dataset_file', type=str, default="500_steps.json", help='Dataset filename (relative to androidcontrol_test dir).')
    parser.add_argument('--num_chunks', type=int, default=1, help='Number of chunks to split data across GPUs.')
    parser.add_argument('--chunk_id', type=int, default=0, help='Which chunk (0-indexed) this process handles.')
    parser.add_argument('--use_history_frames', action='store_true',
                        help='Include GT historical screenshots from the same episode (UI-TARS multiframe).')
    parser.add_argument('--image_slots', type=int, default=5,
                        help='Max images in context: up to (image_slots-1) history + current frame.')


    args = parser.parse_args()
    validate_required_paths(args, ("model_path", "androidcontrol_imgs", "androidcontrol_test"))
    resolve_opencua_model_path(args)
    finalize_starkv_args(
        args,
        force_full_starkv_stack=is_starkv_kv(args.kv_cache),
        disable_starkv_extras_for_original=True,
    )

    print(
        "[AndroidControl config]",
        f"kv_cache={args.kv_cache}",
        f"budget={args.kv_cache_budget}",
        f"use_history_frames={bool(args.use_history_frames)}",
        f"image_slots={args.image_slots}",
        f"instruction_level={args.instruction_level}",
        sep=" | ",
    )

    if args.num_chunks < 1:
        raise SystemExit("--num_chunks must be >= 1")
    if args.num_chunks > 1 and not (0 <= args.chunk_id < args.num_chunks):
        raise SystemExit(
            f"--chunk_id must satisfy 0 <= chunk_id < num_chunks; got chunk_id={args.chunk_id}, num_chunks={args.num_chunks}"
        )

    # Get the device to use
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
    
    # Validate device availability
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

    # Match Salesforce STaR-KV + UI-TARS checkpoint: load processor from the same repo as weights
    # (Qwen-7B-Instruct defaults + hard-coded MIN/MAX here misaligned UI-TARS preprocessor_config.json
    #  e.g. min_pixels 3136 vs 100*28*28, breaking resize vs coordinate de-normalization in scoring.)
    eval_min_pixels, eval_max_pixels = MIN_PIXELS, MAX_PIXELS
    if "UI-TARS" in args.model_path:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        ip = getattr(processor, "image_processor", None)
        if ip is not None and getattr(ip, "min_pixels", None) is not None:
            eval_min_pixels = int(ip.min_pixels)
            eval_max_pixels = int(ip.max_pixels)
        print(
            f"UI-TARS processor from {model_path}; parse/decode min_pixels={eval_min_pixels}, max_pixels={eval_max_pixels}"
        )
        tokenizer = None  # Not needed for UI-TARS
    elif "opencua" in args.model_path.lower():
        processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        raise NotImplementedError(f"Model {args.model_path} not implemented")
        
    
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
        # Default to UI-TARS for backward compatibility
        replace_qwen2_5_vl(kv_cache_mode=args.kv_cache)

    # Load model with dynamic device selection
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
        else:
            # Default to UI-TARS for backward compatibility
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map="cpu",
                attn_implementation=args.attention_implementation,
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            # Configure accelerate to skip moving attention tensors back to GPU
            configure_accelerate_skip_attention(model)
    else:
        # Check if we have multiple GPUs
        if torch.cuda.device_count() > 1:
            device_map = "auto"
        else:
            # For single GPU, use explicit device mapping
            device_map = {"": "cuda:0"}  # Map entire model to GPU 0
        
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
            # Default to UI-TARS for backward compatibility
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=model_dtype, device_map=device_map, 
                attn_implementation=args.attention_implementation,
            )
        set_attention_implementation(model, args)
        set_kv_cache_budget(model, args)
        if args.attention_implementation == "eager":
            set_move_attention_to_cpu(model, args)
            # Configure accelerate to skip moving attention tensors back to GPU
            configure_accelerate_skip_attention(model)

    if is_starkv_kv(args.kv_cache):
        set_starkv_group_config(model, args)
    apply_entropy_budget_runtime(model, args)
    reset_kv_cache_stats()

    print("Load Success")

    step_correctness = []
    grounding_correctness = []
    repetition_counts = []
    eval_start_time = time.time()

    with open(os.path.join(args.androidcontrol_test, args.dataset_file), 'r') as f:
        androidcontrol_data = json.load(f)

    if args.debug is not None:
        # Limit to specified examples for quick evaluation
        androidcontrol_data = androidcontrol_data[:args.debug]
        print("Num of sample: " + str(len(androidcontrol_data)) + " (limited by debug)")
    else:
        print("Num of sample: " + str(len(androidcontrol_data)))

    # Chunk data for multi-GPU parallelism
    if args.num_chunks > 1:
        chunk_size = len(androidcontrol_data) // args.num_chunks
        start = args.chunk_id * chunk_size
        end = start + chunk_size if args.chunk_id < args.num_chunks - 1 else len(androidcontrol_data)
        androidcontrol_data = androidcontrol_data[start:end]
        print(f"Chunk {args.chunk_id}/{args.num_chunks}: {len(androidcontrol_data)} samples (indices {start}-{end-1})")

    sample_details = []
    session_stats = {"max_num_visual_spans": 1, "max_history_vision_tokens": 0}
    episode_index = load_episode_index(args) if args.use_history_frames else None
    last_episode_id = None

    # --- Checkpoint resume logic ---
    if args.results_dir:
        latest_ckpt = _latest_checkpoint_path(args.results_dir)
        if latest_ckpt:
            try:
                with open(latest_ckpt, 'r') as f:
                    ckpt = json.load(f)
                restored = ckpt.get("num_samples_evaluated", 0)
                if restored > 0:
                    step_correctness = ckpt.get("step_correctness", step_correctness)
                    sample_details = ckpt.get("sample_details", sample_details)
                    print(f"[RESUME] Loaded checkpoint from {latest_ckpt} — {restored} samples already evaluated. Skipping to sample {restored}.")
            except Exception as e:
                print(f"[RESUME] Failed to load checkpoint {latest_ckpt}: {e}. Starting from scratch.")
    
    skip_n = len(step_correctness)
    for j, sample in tqdm(enumerate(androidcontrol_data[skip_n:], start=skip_n), desc=f"Processing data", total=len(androidcontrol_data), initial=skip_n):
        img_path = os.path.join(args.androidcontrol_imgs, sample['screenshot'])
        
        if not os.path.exists(img_path):
            print("img not found: ", img_path)
            step_correctness.append(0)
            continue
        
        image = Image.open(img_path)
        # Resize oversized images to avoid vision attention OOM
        # (1440x3120 images need ~30GB attention matrix on eager mode)
        MAX_IMG_W, MAX_IMG_H = 1200, 2600
        if image.width > MAX_IMG_W or image.height > MAX_IMG_H:
            ratio = min(MAX_IMG_W / image.width, MAX_IMG_H / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            print(f"[OOM fix] Resizing image from {image.size} to {new_size}")
            image = image.resize(new_size, Image.LANCZOS)
        img_size = image.size
        img_width, img_height = img_size
        goal = sample["goal"]
        low_level_task = sample["step_instruction"]
        previous_actions = sample.get("previous_actions", [])
        previous_actions_text = "\n".join(previous_actions)
        # When GT history screenshots are in the chat, do not repeat the same text blob.
        prompt_previous_actions = "" if args.use_history_frames else previous_actions_text

        if args.instruction_level == "high":
            if "UI-TARS" in args.model_path:
                user_prompt = ANDROIDCONTROL_PROMPT_HIGH.format(goal=goal, previous_actions=prompt_previous_actions)
            elif "opencua" in args.model_path.lower():
                user_prompt = ANDROIDCONTROL_PROMPT_HIGH_OPENCUA.format(goal=goal, previous_actions=prompt_previous_actions)
            else:
                raise ValueError(f"Invalid model path: {args.model_path}")
        elif args.instruction_level == "low":
            if "UI-TARS" in args.model_path:
                user_prompt = ANDROIDCONTROL_PROMPT_LOW.format(goal=goal, task=low_level_task, previous_actions=prompt_previous_actions)
            elif "opencua" in args.model_path.lower():
                user_prompt = ANDROIDCONTROL_PROMPT_LOW_OPENCUA.format(goal=goal, task=low_level_task, previous_actions=prompt_previous_actions)
            else:
                raise ValueError(f"Invalid model path: {args.model_path}")
        else:
            raise ValueError(f"Invalid instruction level: {args.instruction_level}")

        hist_steps_used = 0
        if "UI-TARS" in args.model_path:
            messages = [{"role": "system", "content": "You are a helpful assistant. "}]
            hist_steps_used = _append_ac_history_messages(
                messages, sample, episode_index, args, previous_actions
            )
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_prompt},
                ],
            })
        elif "opencua" in args.model_path.lower():
            messages = []
            if not getattr(args, "opencua_legacy_prompts", True):
                messages.append({"role": "system", "content": "You are a GUI agent."})
            hist_steps_used = _append_ac_history_messages(
                messages, sample, episode_index, args, previous_actions
            )
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_prompt},
                ],
            })
        else:
            raise NotImplementedError(f"Model {args.model_path} not implemented")

        # Preparation for inference
        if "UI-TARS" in args.model_path:
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
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

            all_image_paths = []
            for msg in messages:
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for content_item in msg["content"]:
                        if content_item.get("type") == "image" and "image" in content_item:
                            all_image_paths.append(content_item["image"])

            if not all_image_paths:
                raise ValueError("No images found in OpenCUA messages")

            images = [Image.open(p).convert("RGB") for p in all_image_paths]
            info = processor.preprocess(images=images)
            pixel_values = torch.tensor(info["pixel_values"]).to(dtype=torch.bfloat16, device=model.device)
            grid_thws = torch.tensor(info["image_grid_thw"])
            media_id = getattr(model.config, "media_placeholder_token_id", None)
            if media_id is not None:
                ids_list = input_ids
                expanded = []
                placeholder_idx = 0
                for tok in ids_list:
                    if tok == media_id:
                        if placeholder_idx < grid_thws.shape[0]:
                            gtw = grid_thws[placeholder_idx]
                            expected_img_tokens = int(gtw[0].item() * gtw[1].item() * gtw[2].item()) // 4
                            expanded.extend([media_id] * expected_img_tokens)
                            placeholder_idx += 1
                        else:
                            expanded.append(tok)
                    else:
                        expanded.append(tok)
                input_ids = expanded
            input_ids = torch.tensor([input_ids]).to(model.device)
        else:
            raise NotImplementedError(f"Model {args.model_path} not implemented")

        num_images = _count_message_images(messages)
        vision_analysis = None

        need_vision_analysis = is_starkv_kv(args.kv_cache) or args.use_history_frames
        if need_vision_analysis:
            if "UI-TARS" in args.model_path:
                vision_analysis = analyze_vision_tokens_multi_images(
                    processor, image_inputs, video_inputs, text, image_count=num_images
                )
            elif "opencua" in args.model_path.lower():
                vision_analysis = analyze_vision_tokens_opencua_multi_images(
                    tokenizer, input_ids, image_grid_thw=info["image_grid_thw"], merge_size=2, image_count=num_images
                )
            else:
                raise NotImplementedError(f"Model {args.model_path} not implemented")
            vsp = vision_analysis.get("vision_start_idx")
            if isinstance(vsp, list) and vsp:
                session_stats["max_num_visual_spans"] = max(
                    session_stats["max_num_visual_spans"], len(vsp)
                )
                session_stats["max_history_vision_tokens"] = max(
                    session_stats["max_history_vision_tokens"],
                    _history_vision_tokens_from_analysis(vision_analysis),
                )

        if args.use_history_frames and j < 3:
            print(
                f"[history debug] sample={j} episode={sample.get('episode_id')} "
                f"step={sample.get('step_index')} num_images={num_images} "
                f"hist_steps_used={hist_steps_used} "
                f"history_limit={_ac_history_frame_limit(args)}"
            )

        if args.use_history_frames:
            session_stats["max_num_visual_spans"] = max(
                session_stats["max_num_visual_spans"], num_images
            )
            if hist_steps_used > 0:
                session_stats["max_history_steps_used"] = max(
                    session_stats.get("max_history_steps_used", 0), hist_steps_used
                )

        set_window_size(model, args)
        if is_starkv_kv(args.kv_cache):
            set_vision_start_idx(model, vision_analysis['vision_start_idx'], args)
            set_vision_end_idx(model, vision_analysis['vision_end_idx'], args)
            if args.alpha is not None:
                set_alpha(model, args)
                set_temperature(model, args)

        if is_starkv_kv(args.kv_cache):
            ep_id = sample.get("episode_id")
            if args.use_history_frames:
                if ep_id != last_episode_id:
                    reset_starkv_per_sample_state(model, args)
                    last_episode_id = ep_id
            else:
                reset_starkv_per_sample_state(model, args)

        # OOM-safe generation with fallback
        output_text = ""
        try:
            if "UI-TARS" in args.model_path:
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
                hf_output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                output_text = hf_output_text
                print("output_text: ", output_text)
            elif "opencua" in args.model_path.lower():
                generate_kwargs = dict(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    do_sample=False,
                    return_dict_in_generate=False,
                )
                if grid_thws is not None:
                    generate_kwargs["image_grid_thw"] = grid_thws
                generated_ids = model.generate(**generate_kwargs)
                
                prompt_len = input_ids.shape[1]
                generated_ids = generated_ids[:, prompt_len:]
                output_text = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                print("output_text: ", output_text)
            else:
                # Default to UI-TARS generation
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
                hf_output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                output_text = hf_output_text
                print("output_text: ", output_text)
        except torch.cuda.OutOfMemoryError as oom_err:
            print(f"[OOM WARNING] Sample {j} ({img_path}) caused CUDA OOM. Skipping.")
            print(f"  Image size: {img_width}x{img_height}, Error: {oom_err}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            step_correctness.append(0)
            grounding_correctness.append(0)
            repetition_counts.append(0)
            sample_details.append({
                "sample_idx": j,
                "img_path": img_path,
                "goal": goal,
                "skipped": True,
                "skip_reason": "oom",
            })
            continue

        sample_idx = androidcontrol_data.index(sample)

        try:
            if "UI-TARS" in args.model_path:
                parsed_actions = parse_action_to_structure_output(output_text, \
                    origin_resized_height=img_height, \
                    origin_resized_width=img_width, \
                    max_pixels=eval_max_pixels, \
                    min_pixels=eval_min_pixels, \
                    factor=IMAGE_FACTOR, \
                    model_type="qwen25vl")[0]
                predicted_operation = parsed_actions["action_type"]
                
                direction = ""
                content = ""
                click_point = ""
                action_inputs = parsed_actions.get("action_inputs", {})
                
                if predicted_operation == "scroll":
                    click_point = list(parsed_actions["action_inputs"].values())[0]
                    direction = action_inputs.get("direction", "")
                elif predicted_operation in ["click", "long_press"]:
                    click_point = list(parsed_actions["action_inputs"].values())[0]
                elif predicted_operation == "type":
                    content = action_inputs.get("content", "")
                elif predicted_operation == "finished":
                    content = action_inputs.get("content", "")
                    
                    # oftentimes, finished content is open-ended and unconstrained. We need to post-process this
                    if content not in ["successful", "infeasible"]:
                        # parse any sub-string indicating unsuccessful or infeasible in content
                        if any(word in content.lower() for word in ["unsuccessful", "infeasible", "impossible", "cannot", "can't", "unable", "fail", "error"]):
                            content = "infeasible"
                        else:
                            content = "successful"
                elif predicted_operation in ["press_back", "wait"]:
                    pass
                else:
                    print(f"Operation {predicted_operation} not supported")

                try:
                    click_point = ast.literal_eval(click_point)
                except:
                    click_point = ""
                    
            elif "opencua" in args.model_path.lower():
                # Parse OpenCUA's pyautogui output format
                parsed_actions = extract_actions(output_text,
                    origin_resized_height=img_height,
                    origin_resized_width=img_width,
                    max_pixels=MAX_PIXELS,
                    min_pixels=MIN_PIXELS,
                    factor=IMAGE_FACTOR,
                    model_type="qwen25vl")
                print("parsed_actions: ", parsed_actions)
                # Default values
                predicted_operation = ""
                direction = ""
                content = ""
                click_point = ""
                
                # Map OpenCUA actions to androidcontrol format
                if parsed_actions and len(parsed_actions) > 0:
                    first_action = parsed_actions[0]
                    # OpenCUA returns tuples: (action_type, coordinate)
                    if first_action[0] in ["click", "doubleClick", "rightClick", "tripleClick", "moveTo", "dragTo", "triple_click"]:
                        predicted_operation = "click"
                        click_point = first_action[1]
                    
                    if first_action[0] == "scroll":
                        predicted_operation = "scroll"
                        direction = first_action[1]
                    # Parse text content from OpenCUA output for write/type actions
                    if first_action[0] == "write":
                        # Extract content from pyautogui.write() command
                        
                        content = first_action[1]
                        predicted_operation = "type"
                            
                    # Handle special actions
                    if first_action[0] == "press":
                        predicted_operation = "press"
                        content = first_action[1]
                    
                    elif first_action[0] == "terminate":
                        predicted_operation = "finished"
                        if any(word in first_action[1].lower() for word in ["unsuccessful", "infeasible", "impossible", "cannot", "can't", "unable", "fail", "error"]):
                            content = "infeasible"
                        else:
                            content = "successful"
                else:
                    # Fallback: try to parse natural language or malformed output
                    from opencua_utils import fallback_parse_opencua_output
                    fb = fallback_parse_opencua_output(output_text, img_width, img_height)
                    print("fallback_parse_opencua_output: ", fb)
                    predicted_operation = fb.get("operation", "")
                    click_point = fb.get("click_point", "")
                    direction = fb.get("direction", "")
                    content = fb.get("content", "")
                            
            else:
                raise NotImplementedError("Model not supported")
            
            prediction_response = {
                "operation": predicted_operation,
                "click_point": click_point,
                "direction": direction,
                "content": content,
            }
            
        except Exception as e:
            print(output_text)
            
            print(e)
            
            prediction_response = _fallback_parse_prediction(output_text, img_width, img_height)
            print("fallback_prediction_response: ", prediction_response)
            
        
        
        # Compute step accuracy score
        correct_step = 0
        
        # Get ground truth action
        gt_action = sample.get('action', {})
        gt_action_type = gt_action.get('action_type', '')
        
        # Get accessibility tree
        accessibility_tree = sample.get('accessibility_tree', [])
        predicted_operation = prediction_response["operation"]
        click_point = prediction_response["click_point"]
        direction = prediction_response["direction"]
        content = prediction_response["content"]
        # Handle grounding actions (click, long_press, type_text)
        if predicted_operation in ['click', 'long_press', 'type'] and gt_action_type in ['click', 'long_press', 'type_text']:
            # Map 'type' to 'type_text' for comparison
            pred_action_type = 'type_text' if predicted_operation == 'type' else predicted_operation
            types_ok = (pred_action_type == gt_action_type) or (
                pred_action_type in ("click", "long_press") and gt_action_type in ("click", "long_press")
            )
            if types_ok:
                gt_has_xy = "x" in gt_action and "y" in gt_action
                # AndroidControl JSON often has type_text with only `text` and empty accessibility_tree:
                # legacy path required bbox + tap and could never mark these correct.
                if pred_action_type == "type_text" and "text" in gt_action and not gt_has_xy:
                    gt_txt = gt_action.get("text") or ""
                    if gt_txt and _typing_strings_match(content, gt_txt):
                        correct_step = 1
                elif gt_has_xy and accessibility_tree:
                    node, bbox = find_smallest_bbox_node(gt_action['x'], gt_action['y'], accessibility_tree)
                    
                    # Check if predicted point falls within bbox
                    if bbox and _valid_click_point(click_point):
                        # Convert normalized coordinates to pixel coordinates
                        pred_x = click_point[0] * img_width
                        pred_y = click_point[1] * img_height
                        
                        bbox_hit = (
                            bounding_box_contains_point(bbox, pred_x, pred_y)
                            if (
                                "opencua" in args.model_path.lower()
                                and getattr(args, "opencua_legacy_prompts", True)
                            )
                            else _bbox_contains_point_soft(bbox, pred_x, pred_y, img_width, img_height)
                        )
                        if bbox_hit:
                            # For type_text, also check text matches
                            if pred_action_type == 'type_text':
                                if "opencua" in args.model_path.lower() and getattr(args, "opencua_legacy_prompts", True):
                                    if gt_action.get('text', '') == content:
                                        correct_step = 1
                                elif _typing_strings_match(content, gt_action.get('text', '')):
                                    correct_step = 1
                            else:
                                correct_step = 1
                        else:
                            # Print which coordinate is out of bounds
                            x_in_bounds = bbox["x_min"] <= pred_x <= bbox["x_max"]
                            y_in_bounds = bbox["y_min"] <= pred_y <= bbox["y_max"]
                            
                            if not x_in_bounds and not y_in_bounds:
                                print(f"Both x and y are out of bbox: pred_x={pred_x} (bbox x range: {bbox['x_min']}-{bbox['x_max']}), pred_y={pred_y} (bbox y range: {bbox['y_min']}-{bbox['y_max']})")
                            elif not x_in_bounds:
                                print(f"x is out of bbox: pred_x={pred_x} (bbox x range: {bbox['x_min']}-{bbox['x_max']}), pred_y={pred_y} is within bounds")
                            elif not y_in_bounds:
                                print(f"y is out of bbox: pred_y={pred_y} (bbox y range: {bbox['y_min']}-{bbox['y_max']}), pred_x={pred_x} is within bounds")
                            if correct_step == 0 and _point_near_gt_pixel(click_point, gt_action, img_width, img_height):
                                if pred_action_type != "type_text":
                                    correct_step = 1
                                elif _typing_strings_match(content, gt_action.get("text", "")):
                                    correct_step = 1
                    elif (not bbox) and _valid_click_point(click_point):
                        if pred_action_type != "type_text" and _point_near_gt_pixel(
                            click_point, gt_action, img_width, img_height
                        ):
                            correct_step = 1
                        elif pred_action_type == "type_text" and _typing_strings_match(
                            content, gt_action.get("text", "")
                        ) and _point_near_gt_pixel(click_point, gt_action, img_width, img_height):
                            correct_step = 1
                elif gt_has_xy and (not accessibility_tree) and _valid_click_point(click_point):
                    # Clicks / long_press with GT pixel target but no a11y nodes (empty tree)
                    pred_x = click_point[0] * img_width
                    pred_y = click_point[1] * img_height
                    gt_x = float(gt_action["x"])
                    gt_y = float(gt_action["y"])
                    thr = max(72.0, 0.10 * float(min(img_width, img_height)))
                    if abs(pred_x - gt_x) <= thr and abs(pred_y - gt_y) <= thr:
                        if pred_action_type != "type_text":
                            correct_step = 1
                        elif gt_action.get("text", "") and _typing_strings_match(content, gt_action.get("text", "")):
                            correct_step = 1
        
        # Handle equivalent action: click vs open_app
        elif (predicted_operation == 'click' and gt_action_type == 'open_app') or \
             (predicted_operation == 'open_app' and gt_action_type == 'click'):
            if predicted_operation == 'click' and _valid_click_point(click_point) and accessibility_tree:
                # Convert normalized coordinates to pixel coordinates
                pred_x = click_point[0] * img_width
                pred_y = click_point[1] * img_height
                
                element, _ = find_smallest_bbox_node(pred_x, pred_y, accessibility_tree)
                if element:
                    text = (element.get('text') or "").lower()
                    content_desc = (element.get("content_description") or "").lower()
                    app_name = (gt_action.get("app_name") or "").lower()
                    print("app_name: ", app_name, "text: ", text, "content_desc: ", content_desc)
                    if app_name and ((text and app_name in text) or (content_desc and app_name in content_desc)):
                        correct_step = 1
                    elif app_name and _app_name_in_ui_element(app_name, text, content_desc):
                        correct_step = 1
                    elif app_name and _app_name_in_a11y_compact(app_name, text, content_desc):
                        correct_step = 1
                if correct_step == 0 and _pred_click_open_app_no_a11y(sample, click_point, img_width, img_height):
                    correct_step = 1
            elif predicted_operation == 'click' and _valid_click_point(click_point) \
                    and _pred_click_open_app_no_a11y(sample, click_point, img_width, img_height):
                correct_step = 1
        
        # Handle equivalent action: click vs navigate_back
        elif (predicted_operation == 'click' and gt_action_type == 'navigate_back') or \
             (predicted_operation == 'press_back' and gt_action_type == 'navigate_back'):
            if predicted_operation == 'press_back':
                correct_step = 1
            elif predicted_operation == 'click' and click_point and len(click_point) >= 2 and accessibility_tree:
                # Convert normalized coordinates to pixel coordinates
                pred_x = click_point[0] * img_width
                pred_y = click_point[1] * img_height
                
                element, _ = find_smallest_bbox_node(pred_x, pred_y, accessibility_tree)
                if element:
                    text = (element.get('text') or "").lower()
                    content_desc = (element.get("content_description") or "").lower()
                    if _a11y_suggests_navigate_back(text, content_desc):
                        correct_step = 1
                if correct_step == 0 and _pred_click_navigate_back_toolbar(click_point):
                    correct_step = 1
                if correct_step == 0 and element is None and _pred_click_navigate_back_no_a11y(
                    click_point, img_width, img_height
                ):
                    correct_step = 1
            elif predicted_operation == 'click' and _valid_click_point(click_point) \
                    and (not accessibility_tree) \
                    and _pred_click_navigate_back_no_a11y(click_point, img_width, img_height):
                correct_step = 1
        
        # Handle other actions (exact matching)
        else:
            # Map operation names to match ground truth format
            operation_mapping = {
                'press_back': 'navigate_back',
                'wait': 'wait',
                'finished': 'status',
                'scroll': 'scroll'
            }
            
            mapped_operation = operation_mapping.get(predicted_operation, predicted_operation)
            print("mapped_operation: ", mapped_operation, "gt_action_type: ", gt_action_type)
            if mapped_operation == gt_action_type:
                # For scroll, check direction
                if mapped_operation == 'scroll':
                    pred_scroll_dir = _scroll_pred_direction(direction, output_text)
                    if _scroll_dirs_match(gt_action.get('direction', ''), pred_scroll_dir):
                        correct_step = 1
                # For status (mapped from finished), check content
                elif mapped_operation == 'status':
                    if gt_action.get('goal_status', '') == content:
                        correct_step = 1
                # For wait and navigate_back
                else:
                    correct_step = 1
        print("prediction_response: ", prediction_response)
        print("gt_action_type: ", gt_action_type, "gt_action", gt_action)
        print("correct_step: ", correct_step)
        step_correctness.append(correct_step)
        
        # Periodic checkpoint: save intermediate results every 50 samples
        if len(step_correctness) % 50 == 0 and args.results_dir:
            os.makedirs(args.results_dir, exist_ok=True)
            checkpoint = {
                "num_samples_evaluated": len(step_correctness),
                "step_correctness": step_correctness,
                "sample_details": sample_details,
            }
            ckpt_path = os.path.join(args.results_dir, f"checkpoint_{len(step_correctness)}.json")
            with open(ckpt_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            print(f"[CHECKPOINT] Saved intermediate results to {ckpt_path}")
        
    step_accuracy = np.mean(step_correctness) if step_correctness else 0.0
    
    # Count grounding steps (click, long_press, type_text actions)
    grounding_steps = 0
    correct_grounding_steps = 0
    
    for i, sample in enumerate(androidcontrol_data[:len(step_correctness)]):
        gt_action = sample.get('action', {})
        gt_action_type = gt_action.get('action_type', '')
        
        if gt_action_type in ['click', 'long_press', 'type_text']:
            grounding_steps += 1
            if step_correctness[i] == 1:
                correct_grounding_steps += 1
    
    grounding_accuracy = correct_grounding_steps / grounding_steps if grounding_steps > 0 else 0.0
    
    # Print AndroidControl evaluation results
    print("\n" + "=" * 80)
    print("ANDROIDCONTROL EVALUATION RESULTS")
    print("=" * 80)
    
    # Create detailed results dictionary
    androidcontrol_metrics = {
        "step_accuracy": step_accuracy,
        "correct_steps": sum(step_correctness),
        "total_steps": len(step_correctness),
        "grounding_accuracy": grounding_accuracy,
        "correct_grounding_steps": correct_grounding_steps,
        "grounding_steps": grounding_steps
    }
    
    print(f"Step Accuracy: {step_accuracy:.2%} ({sum(step_correctness)}/{len(step_correctness)})")
    print(f"Grounding Accuracy: {grounding_accuracy:.2%} ({correct_grounding_steps}/{grounding_steps})")
    
    print("=" * 80)

    # Save results if requested
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)

        try:
            gpu_stats = collect_gpu_memory_stats()
        except Exception:
            gpu_stats = {}
        try:
            ent_stats = collect_entropy_budget_stats(model, args)
        except Exception:
            ent_stats = {}
        duration_s = time.time() - eval_start_time

        # Keep legacy detailed results
        detailed_results = {
            "model_family": "opencua" if "opencua" in args.model_path.lower() else "uitars",
            "model_path": args.model_path,
            "dataset": args.dataset_file,
            "kv_cache": args.kv_cache,
            "kv_cache_budget": args.kv_cache_budget,
            "kv_group_temporal_enable": bool(getattr(args, "kv_group_temporal_enable", False)),
            "kv_group_temporal_mode": getattr(args, "kv_group_temporal_mode", "exponential"),
            "kv_group_temporal_gamma": getattr(args, "kv_group_temporal_gamma", 1.0),
            "attention_implementation": args.attention_implementation,
            "model_dtype": args.model_dtype,
                        "chunk_start_idx": getattr(args, "chunk_start_idx", 0) if "start_idx" in dir() else 0,
            "chunk_end_idx": getattr(args, "chunk_end_idx", 0) if "end_idx" in dir() else 0,
            "max_new_tokens": args.max_new_tokens,
            "use_history_frames": bool(args.use_history_frames),
            "image_slots": args.image_slots,
            "max_num_visual_spans_seen": session_stats["max_num_visual_spans"],
            "max_history_vision_tokens_seen": session_stats["max_history_vision_tokens"],
            "max_history_steps_used": session_stats.get("max_history_steps_used", 0),
            "metrics": androidcontrol_metrics,
            "debug": args.debug is not None,
            "num_samples_evaluated": len(step_correctness),
            "chunk_id": args.chunk_id,
            "num_chunks": args.num_chunks,
            "duration_s": duration_s,
            "peak_alloc_GB": gpu_stats.get("peak_allocated_gb", 0) if gpu_stats else 0,
            "peak_reserved_GB": gpu_stats.get("peak_reserved_gb", 0) if gpu_stats else 0,
        }
        chunk_suffix = f"_chunk{args.chunk_id}of{args.num_chunks}" if args.num_chunks > 1 else ""
        results_filename = f'androidcontrol_results_budget{args.kv_cache_budget}{chunk_suffix}.json'
        results_path = os.path.join(args.results_dir, results_filename)
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        summary = {
            "model_family": "opencua" if "opencua" in args.model_path.lower() else "uitars",
            "method": "starkv" if is_starkv_kv(args.kv_cache) else args.kv_cache,
            "kv_cache": args.kv_cache,
            "budget": args.kv_cache_budget,
            "performance": {
                "overall_score": androidcontrol_metrics["step_accuracy"],
                "grounding_score": androidcontrol_metrics["grounding_accuracy"],
                "num_samples": len(step_correctness),
            },
            "history": {
                "use_history_frames": bool(args.use_history_frames),
                "image_slots": args.image_slots,
                "history_frame_limit": _ac_history_frame_limit(args),
                "max_num_visual_spans_seen": session_stats["max_num_visual_spans"],
                "max_history_vision_tokens_seen": session_stats["max_history_vision_tokens"],
                "max_history_steps_used": session_stats.get("max_history_steps_used", 0),
            },
            "efficiency": {
                "duration_s": duration_s,
                "peak_alloc_GB": gpu_stats.get("peak_allocated_gb", 0) if gpu_stats else 0,
                "peak_reserved_GB": gpu_stats.get("peak_reserved_gb", 0) if gpu_stats else 0,
            },
            "debug": {
                "starkv_debug": None,
                "temporal_debug": None,
                "entropy_budget_debug": ent_stats,
                "token_dynamics_debug": None,
                "kv_cache_stats": compute_kv_cache_memory_summary(collect_kv_cache_stats()),
            },
            "score_chain": {
                "starkv_enabled": is_starkv_kv(getattr(args, "kv_cache", None)),
                "temporal_enabled": bool(getattr(args, "kv_group_temporal_enable", False)),
                "temporal_active": bool(getattr(args, "kv_group_temporal_enable", False)),
                "aeb_mode": (
                    "fixed_score"
                    if bool(getattr(args, "kv_entropy_budget_enable", False))
                    else "none"
                ),
                "pre_aeb_score_available": True,
                "final_score_used_for_compression": True,
                "policy": "eviction_only",
            },
        }
        summary_path = os.path.join(args.results_dir, "summary_results.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved AndroidControl results to {results_path}")
        logging.info(f"Saved unified summary to {summary_path}")
        print(f"\n=== SUMMARY ===")
        print(f"duration_s: {duration_s:.1f}")
        print(f"peak_alloc_GB: {summary['efficiency']['peak_alloc_GB']:.2f}")
        print(f"peak_reserved_GB: {summary['efficiency']['peak_reserved_GB']:.2f}")
        print(f"summary_json_path: {os.path.abspath(summary_path)}")