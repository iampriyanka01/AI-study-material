"""
Parses the model's text reply to detect a tool call.

The model is instructed (see prompt.txt) to reply in this exact format
when it wants to use a tool:

    CALL_FUNCTION:function_name:param1=value,param2=value,...

Array-type values look like:  genres=["fiction","sci_fi"]

If the model's reply doesn't match that format at all, we treat it as a
normal conversational answer (no tool call).
"""

import re

# Matches:  CALL_FUNCTION:<name>:<the rest of the params string>
FUNCTION_CALL_PATTERN = r"CALL_FUNCTION:(\w+):(.*)"


def parse_function_call(text):
    """
    Look for a CALL_FUNCTION line in `text`.

    Returns:
        (function_name, parameters_dict) if a tool call was found,
        (None, None) otherwise.
    """
    match = re.search(FUNCTION_CALL_PATTERN, text)
    if not match:
        return None, None

    function_name = match.group(1)
    params_text = match.group(2)

    parameters = _parse_params(params_text)
    return function_name, parameters


def _parse_params(params_text):
    """
    Turn "action=add,book_title=The Great Gatsby" into
    {"action": "add", "book_title": "The Great Gatsby"}.

    Handles three kinds of values:
      - arrays, like genres=["fiction","sci_fi"]  -> Python list
      - integers, like max_results=5              -> Python int
      - everything else                            -> plain string
    """
    parameters = {}

    # Split on commas, but NOT commas that are inside [ ... ] (arrays).
    param_pairs = re.split(r",(?![^[]*\])", params_text)

    for pair in param_pairs:
        if "=" not in pair:
            continue  # skip anything malformed

        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            parameters[key] = _parse_array(value)
        elif value.isdigit():
            parameters[key] = int(value)
        else:
            parameters[key] = value

    return parameters


def _parse_array(array_text):
    """
    Turn '["Tokyo","Seoul"]' into ['Tokyo', 'Seoul'].
    Returns [] for an empty array like '[]'.
    """
    inner = array_text[1:-1].strip()  # drop the [ and ]
    if not inner:
        return []

    items = [item.strip().strip('"').strip("'") for item in inner.split(",")]
    return items


# Kept for backward compatibility with any code importing this name directly.
parse_array_parameter = _parse_array