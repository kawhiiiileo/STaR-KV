import math
import re
import ast

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def convert_point_to_coordinates(text, is_answer=False):
    """Replace <point>x y</point> with (x,y) for AST parsing (supports ints/floats)."""
    pattern = r"<point>\s*([\d.]+)\s+([\d.]+)\s*</point>"

    def replace_match(match):
        x1, y1 = match.groups()
        x = int(float(x1))
        y = int(float(y1))
        return f"({x},{y})"

    text = re.sub(r"\[EOS\]", "", text)
    return re.sub(pattern, replace_match, text).strip()


def _ast_value_to_python(node):
    """Extract Python values from AST nodes (constants, tuples, lists)."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.Tuple):
        return tuple(_ast_value_to_python(e) for e in node.elts)
    if isinstance(node, ast.List):
        return [_ast_value_to_python(e) for e in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _ast_value_to_python(node.operand)
        if isinstance(v, (int, float)):
            return -v
        return None
    try:
        if hasattr(ast, "unparse"):
            return ast.literal_eval(ast.unparse(node))
    except Exception:
        pass
    return None


# Parse each action string from model output.
def parse_action(action_str):
    try:
        node = ast.parse(action_str.strip(), mode='eval')
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")
        call = node.body
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            val = _ast_value_to_python(kw.value)
            kwargs[key] = val

        # Positional args: click(512, 300), wait(), etc.
        if func_name == "click" and len(call.args) >= 2:
            x = _ast_value_to_python(call.args[0])
            y = _ast_value_to_python(call.args[1])
            if x is not None and y is not None:
                kwargs.setdefault("point", f"({int(float(x))},{int(float(y))})")
        elif func_name == "wait" and not kwargs:
            pass

        return {'function': func_name, 'args': kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None


def _strip_redacted_thinking(text):
    out = re.sub(
        r"<think>[\s\S]*?</think>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    return out.strip()


def _extract_first_balanced_call(text, start_pos):
    """From start_pos at opening '(', find matching closing ')'."""
    if start_pos >= len(text) or text[start_pos] != "(":
        return None
    depth = 0
    j = start_pos
    while j < len(text):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[start_pos : j + 1]
        j += 1
    return None


def _extract_first_action_call_substring(text):
    """Find first UI-TARS–style action call (click/wait/...) for fallback parsing."""
    pattern = re.compile(
        r"(?is)\b(click|left_double|right_single|drag|scroll|type|hotkey|wait|finished)\s*\(",
    )
    m = pattern.search(text)
    if not m:
        return None
    func_start = m.start(1)
    paren = text.find("(", m.start())
    if paren < 0:
        return None
    tail = _extract_first_balanced_call(text, paren)
    if tail is None:
        return None
    return text[func_start : paren + len(tail)]


def escape_single_quotes(text):
    # Match unescaped single quotes (not \\').
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def parse_action_to_structure_output(text,
                                     factor=IMAGE_FACTOR,
                                     origin_resized_height=None,
                                     origin_resized_width=None,
                                     model_type="qwen25vl",
                                     max_pixels=16384 * 28 * 28,
                                     min_pixels=100 * 28 * 28,
                                     raise_on_parse_error=True):
    """Parse UI-TARS output into structured actions. Supports Action: prefix, fallback first-call extraction, <think>."""
    text = text.strip()
    text = _strip_redacted_thinking(text)

    if "<point>" in text:
        text = convert_point_to_coordinates(text)
    if "start_point=" in text:
        text = text.replace("start_point=", "start_box=")
    if "end_point=" in text:
        text = text.replace("end_point=", "end_box=")
    if "point=" in text:
        text = text.replace("point=", "start_box=")

    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(
            origin_resized_height,
            origin_resized_width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels)

    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action: |$)"
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action: |$)"
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action: |$)"
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()

    has_action_keyword = re.search(r"(?i)\bAction:\s*", text) is not None
    all_action = []

    if has_action_keyword:
        action_body = re.split(r"(?i)\bAction:\s*", text, maxsplit=1)[-1]
        tmp_all_action = action_body.split(")\n\n")
        for action_str in tmp_all_action:
            action_str = action_str.strip()
            if not action_str:
                continue
            if "type(content" in action_str:
                if not action_str.strip().endswith(")"):
                    action_str = action_str.strip() + ")"
                pattern = r"type\(content='(.*?)'\)"
                if re.search(pattern, action_str):
                    content = re.sub(
                        pattern,
                        lambda m: m.group(1),
                        action_str,
                    )
                    action_str = escape_single_quotes(content)
                    action_str = "type(content='" + action_str + "')"
                else:
                    if raise_on_parse_error:
                        raise ValueError("Pattern not found in the input string.")
                    continue
            if not action_str.endswith(")"):
                action_str = action_str + ")"
            all_action.append(action_str)
    else:
        fb = _extract_first_action_call_substring(text)
        if fb:
            all_action = [fb.strip()]

    if not all_action:
        return []

    parsed_actions = [
        parse_action(action.replace("\n", "\\n").lstrip())
        for action in all_action
    ]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance is None:
            print(f"Action can't parse: {raw_str}")
            if raise_on_parse_error:
                raise ValueError(f"Action can't parse: {raw_str}")
            continue
        action_type = action_instance["function"]
        params = action_instance["args"]

        action_inputs = {}
        for param_name, param in params.items():
            if param == "":
                continue
            param = param.lstrip()
            action_inputs[param_name.strip()] = param

            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(
                                float(num / smart_resize_height))
                        else:
                            float_numbers.append(
                                float(num / smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [
                        float_numbers[0], float_numbers[1], float_numbers[0],
                        float_numbers[1]
                    ]
                action_inputs[param_name.strip()] = str(float_numbers)

        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
        break

    return actions


def analyze_vision_tokens_multi_images(processor, image_inputs, video_inputs, text, image_count=1):
    """Vision token span indices for UI-TARS / Qwen2.5-VL (used by STaR-KV eval)."""
    text_only_inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full_length = full_inputs.input_ids.shape[1]
    tokenizer = processor.tokenizer
    full_tokens = tokenizer.convert_ids_to_tokens(full_inputs.input_ids[0])
    text_only_tokens = tokenizer.convert_ids_to_tokens(text_only_inputs.input_ids[0])

    vision_start_indices = []
    vision_end_indices = []
    vision_token_count = full_tokens.count("<|image_pad|>")
    current_vision_start = None
    for i, token in enumerate(full_tokens):
        if "<|vision_start|>" in token:
            current_vision_start = i + 1
        elif "<|vision_end|>" in token and current_vision_start is not None:
            vision_start_indices.append(current_vision_start)
            vision_end_indices.append(i)
            current_vision_start = None

    if len(vision_start_indices) == 0 and vision_token_count > 0:
        estimated_system_length = len(
            [t for t in text_only_tokens if t in ["<|im_start|>", "", "system", "user"]]
        )
        vision_start_idx = min(estimated_system_length, full_length - vision_token_count)
        vision_end_idx = vision_start_idx + vision_token_count
        vision_start_indices = [vision_start_idx]
        vision_end_indices = [vision_end_idx]

    assert image_count == len(vision_start_indices) == len(vision_end_indices), (
        f"Expected {image_count} images but found {len(vision_start_indices)} vision token pairs"
    )
    return {
        "vision_start_idx": vision_start_indices,
        "vision_end_idx": vision_end_indices,
    }


