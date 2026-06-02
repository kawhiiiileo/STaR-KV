import re
import math
from typing import Any, List, Optional, Tuple


def analyze_vision_tokens_opencua_multi_images(tokenizer, input_ids, image_grid_thw, merge_size=2, image_count=1):
    """
    Analyze vision token lengths and positions for multiple images in the input sequence.

    Args:
        tokenizer: The tokenizer
        input_ids: Token IDs
        image_grid_thw: Tensor of shape [num_images, 3] where 3 = (T, H, W)
        merge_size: Merge size for visual tokens (default: 2)
        image_count: Expected number of images

    Returns:
        dict: Contains vision token analysis information with lists of start/end indices
    """
    if isinstance(input_ids, list):
        ids_for_tokens = input_ids
    elif hasattr(input_ids, "shape") and len(input_ids.shape) > 1:
        ids_for_tokens = input_ids[0]
    else:
        ids_for_tokens = input_ids
    full_tokens = tokenizer.convert_ids_to_tokens(ids_for_tokens)

    if len(image_grid_thw.shape) == 3:
        image_grid_thw = image_grid_thw[0]
    elif len(image_grid_thw.shape) == 1:
        image_grid_thw = image_grid_thw.unsqueeze(0)

    vision_start_indices = []
    vision_end_indices = []

    media_begin_positions = []
    media_end_positions = []
    
    for i, token in enumerate(full_tokens):
        if '<|media_begin|>' in token:
            media_begin_positions.append(i)
        elif '<|media_end|>' in token:
            media_end_positions.append(i)

    for img_idx in range(min(len(media_begin_positions), len(image_grid_thw))):
        _, patch_h, patch_w = image_grid_thw[img_idx]

        num_visual_tokens = int(patch_h * patch_w / (merge_size ** 2))

        if img_idx == 0:
            vision_start_idx = media_begin_positions[0] + 1
        else:
            prev_vision_end = vision_end_indices[-1]
            prev_media_end = media_end_positions[img_idx - 1]
            curr_media_begin = media_begin_positions[img_idx]

            text_tokens_between = curr_media_begin - prev_media_end - 1

            vision_start_idx = prev_vision_end + text_tokens_between + 1
        
        vision_end_idx = vision_start_idx + num_visual_tokens

        vision_start_indices.append(vision_start_idx)
        vision_end_indices.append(vision_end_idx)

    assert len(vision_start_indices) == image_count, f"Expected {image_count} images but found {len(vision_start_indices)} <|media_begin|> markers"

    analysis = {
        'vision_start_idx': vision_start_indices,
        'vision_end_idx': vision_end_indices,
    }

    return analysis

def opencua_parse_action(code, origin_resized_height, origin_resized_width,
                    max_pixels, min_pixels, factor, model_type):
    """
    Convert pyautogui code to normalized coordinates.
    Example: pyautogui.click(x=1424, y=264)
    """
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
                origin_resized_height,
                origin_resized_width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels)

    coordinates = parse_coordinates_from_code(code)
    actions = []
    for coordinate in coordinates:
        x, y = coordinate
        if  model_type == "qwen25vl":
            x = float(x / smart_resize_width)
            y = float(y / smart_resize_height)
        else:
            x = float(x / factor)
            y = float(y / factor)
        actions.append({
            "action_type": "click",
            "coordinate": [x, y],
            "text": code
        })
    return actions
    

def parse_coordinates_from_line(line, max_num = 2):
    if not line:
        return None

    if line.startswith((
        "pyautogui.click", 
        "pyautogui.moveTo", 
        "pyautogui.dragTo",
        "pyautogui.doubleClick", 
        "pyautogui.rightClick", 
        "pyautogui.middleClick", 
        "pyautogui.tripleClick",
        "computer.tripleClick",
    )):
        numbers = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        floats = [float(n) for n in numbers][:max_num]
        return tuple(floats)

    return None

def parse_coordinates_from_code(code, max_num = 2):
    if not code:
        return None

    all_coords = []
    code_lines = code.split("\n")
    for line in code_lines:
        line = line.strip()
        if not line:
            continue
        
        coords = parse_coordinates_from_line(line)
        if coords:
            x, y = coords
            all_coords.append((x, y))
            
    return all_coords
    

def smart_resize(
    height: int,
    width: int,
    factor=28, 
    min_pixels=3136, 
    max_pixels=12845056,
    max_aspect_ratio_allowed: float | None = None,
    size_can_be_smaller_than_factor: bool = False,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if not size_can_be_smaller_than_factor and (height < factor or width < factor):
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor} "
            f"(when size_can_be_smaller_than_factor is False)"
        )
    elif max_aspect_ratio_allowed is not None and max(height, width) / min(height, width) > max_aspect_ratio_allowed:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_aspect_ratio_allowed}, "
            f"got {max(height, width) / min(height, width)}"
            f"(when max_aspect_ratio_allowed is not None)"
        )
    h_bar = max(1, round(height / factor)) * factor
    w_bar = max(1, round(width / factor)) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(1, math.floor(height / beta / factor)) * factor
        w_bar = max(1, math.floor(width / beta / factor)) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def parse_response_actions_opencua(
    response: str, trajectory=None, step_idx=None
) -> Optional[str]:
    """Parse model output, extracting pyautogui/computer lines.

    Returns None if no pyautogui/computer lines are found.
    """
    if response is None:
        return None

    lines = response.split("\n")
    action_lines: List[str] = []

    # First pass: lines that start with commands
    for raw in lines:
        line = raw.strip()
        if line.startswith("pyautogui.") or line.startswith("computer."):
            action_lines.append(line)

    # If we already have extracted lines, optionally normalize coordinates
    if action_lines:
        return "\n".join(action_lines)

    return None

def extract_actions(action: str, origin_resized_height, origin_resized_width, 
                    max_pixels, min_pixels, factor, model_type) -> List[Tuple[str, Any]]:
    """Extract (type, value) tuples from parsed action string.

    Follows the logic of extract_actions() in opencua_eval_all_in_one.py.
    """
    
    if model_type == "qwen25vl":

        smart_resize_height, smart_resize_width = smart_resize(
                origin_resized_height,
                origin_resized_width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels)
        
    if not action:
        return []

    actions: List[Tuple[str, Any]] = []

    action_lines = action.strip().split("\n")
    for raw in action_lines:
        line = raw.strip()

        # computer.terminate
        if line.startswith("computer.terminate"):
            status_match = re.search(r"status=['\"](\w+)['\"]", line)
            if status_match:
                actions.append(("terminate", status_match.group(1)))
                continue

        # computer.triple_click
        if line.startswith("computer.triple_click"):
            coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
            if coord_match:
                x, y = map(float, coord_match.groups())
                if model_type == "qwen25vl":
                    x = float(x / smart_resize_width)
                    y = float(y / smart_resize_height)
                else:
                    x = float(x / factor)
                    y = float(y / factor)
                actions.append(("triple_click", (x, y)))
                continue

        # pyautogui.*
        if line.startswith("pyautogui."):
            coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
            if coord_match:
                x, y = map(float, coord_match.groups())
                if model_type == "qwen25vl":
                    x = float(x / smart_resize_width)
                    y = float(y / smart_resize_height)
                else:
                    x = float(x / factor)
                    y = float(y / factor)
                if "click" in line and "doubleClick" not in line and "rightClick" not in line:
                    actions.append(("click", (x, y)))
                elif "moveTo" in line:
                    actions.append(("moveTo", (x, y)))
                elif "doubleClick" in line:
                    actions.append(("doubleClick", (x, y)))
                elif "rightClick" in line:
                    actions.append(("rightClick", (x, y)))
                elif "dragTo" in line:
                    actions.append(("dragTo", (x, y)))

            # write(message=...)
            write_match = re.search(r"message=['\"](.+?)['\"]", line)
            if write_match:
                text = write_match.group(1)
                actions.append(("write", text))

            # write('...') positional
            if not write_match:
                write_positional = re.search(r"pyautogui\.write\((['\"])(.*?)\1\)", line)
                if write_positional:
                    actions.append(("write", write_positional.group(2)))

            # press/hotkey keys=[...]
            keys_match = re.findall(r"keys=\[(.*?)\]", line)
            if keys_match:
                key_string = keys_match[0]
                key_list = re.findall(r"['\"]([^'\"]*)['\"]|(\w+)", key_string)
                keys = [m[0] or m[1] for m in key_list if m[0] or m[1]]
                normalized_keys: List[str] = []
                for k in keys:
                    k = k.strip()
                    normalized_keys.append("ctrl" if k.lower() in ("cmd", "command") else k)
                if "hotkey" in line:
                    actions.append(("hotkey", normalized_keys))
                else:
                    actions.append(("press", normalized_keys))

            # hotkey positional: pyautogui.hotkey('ctrl', 'v')
            if "hotkey(" in line and "keys=" not in line:
                inside = re.search(r"pyautogui\.hotkey\((.*)\)", line)
                if inside:
                    arg_str = inside.group(1)
                    parts = re.findall(r"['\"]([^'\"]+)['\"]", arg_str)
                    if parts:
                        normalized_keys = [
                            ("ctrl" if p.strip().lower() in ("cmd", "command") else p.strip()) for p in parts
                        ]
                        actions.append(("hotkey", normalized_keys))

            # press positional: pyautogui.press('enter') or press(['ctrl','v'])
            if "press(" in line and "keys=" not in line:
                inside = re.search(r"pyautogui\.press\((.*)\)", line)
                if inside:
                    arg_str = inside.group(1).strip()
                    keys: List[str] = []
                    if arg_str.startswith("["):
                        parts = re.findall(r"['\"]([^'\"]+)['\"]", arg_str)
                        keys = [p.strip() for p in parts]
                    else:
                        one = re.search(r"['\"]([^'\"]+)['\"]", arg_str)
                        if one:
                            keys = [one.group(1).strip()]
                    if keys:
                        normalized_keys = [
                            ("ctrl" if k.lower() in ("cmd", "command") else k) for k in keys
                        ]
                        if len(normalized_keys) > 1:
                            actions.append(("hotkey", normalized_keys))
                        else:
                            actions.append(("press", normalized_keys))

            # scroll
            scroll_match = re.search(r"pyautogui\.scroll\(([-\d]+)\)", line)
            if scroll_match:
                actions.append(("scroll", int(scroll_match.group(1))))

    return actions


def fallback_parse_opencua_output(output_text: str, img_w: int, img_h: int) -> dict:
    """Best-effort parse when extract_actions returns empty (natural language / malformed)."""
    text = str(output_text or "").strip().lower()
    op = ""
    direction = ""
    content = ""
    click_point = ""

    if any(k in text for k in ["press back", "go back", "navigate back", "back button", "return to"]):
        op = "press_back"
    elif any(k in text for k in ["wait", "pause", "sleep", "loading", "finish loading"]):
        op = "wait"
    elif any(k in text for k in ["scroll", "swipe", "slide"]):
        op = "scroll"
        if "down" in text or "bottom" in text:
            direction = "down"
        elif "up" in text or "top" in text:
            direction = "up"
        elif "right" in text:
            direction = "right"
        elif "left" in text:
            direction = "left"
        else:
            direction = "down"
    elif any(k in text for k in ["type", "write", "enter", "input", "fill"]):
        op = "type"
        quotes = re.findall(r"['\"]([^'\"]+)['\"]", output_text)
        if quotes:
            content = quotes[-1]
        else:
            m = re.search(r"(?:type|write|enter)\s+(?:'|\"|`)?(.+?)(?:\.|$)", text, re.IGNORECASE)
            if m:
                content = m.group(1).strip()
    elif "long press" in text or "hold" in text or "long-press" in text:
        op = "long_press"
    elif "click" in text or "tap" in text or "press" in text or "select" in text:
        op = "click"

    if op in ("click", "long_press"):
        nx, ny = 0.5, 0.5
        if "top-left" in text or "upper left" in text:
            nx, ny = 0.1, 0.1
        elif "top-right" in text or "upper right" in text:
            nx, ny = 0.9, 0.1
        elif "bottom-left" in text or "lower left" in text:
            nx, ny = 0.1, 0.9
        elif "bottom-right" in text or "lower right" in text:
            nx, ny = 0.9, 0.9
        elif "top" in text or "upper" in text or "header" in text:
            ny = 0.15
        elif "bottom" in text or "lower" in text or "footer" in text:
            ny = 0.85
        elif "left" in text:
            nx = 0.2
        elif "right" in text:
            nx = 0.8
        elif "center" in text or "middle" in text:
            nx, ny = 0.5, 0.5

        nums = re.findall(r"(-?\d+)\s*,\s*(-?\d+)", output_text)
        if not nums:
            nums = re.findall(r"x[=:]\s*(-?\d+)[,\s]+y[=:]\s*(-?\d+)", output_text, re.IGNORECASE)
        if nums:
            x = max(0, min(img_w, int(nums[-1][0])))
            y = max(0, min(img_h, int(nums[-1][1])))
            nx = x / max(1, img_w)
            ny = y / max(1, img_h)
        click_point = [nx, ny]

    return {
        "operation": op,
        "click_point": click_point,
        "direction": direction,
        "content": content,
    }