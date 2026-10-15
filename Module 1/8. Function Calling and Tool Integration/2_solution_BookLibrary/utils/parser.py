import re

def parse_function_call(text):
    """
    Parse function call with intermediate complexity (arrays and optional parameters)
    
    Args:
        text (str): Text that may contain a function call
        
    Returns:
        tuple: (function_name, parameters) or (None, None) if no function call found
    """
    pattern = r"CALL_FUNCTION:(\w+):(.+)"
    match = re.search(pattern, text)
    
    if not match:
        return None, None
    
    function_name = match.group(1)
    params_str = match.group(2)
    
    # Parse parameters string into a dictionary
    parameters = {}
    
    if params_str:
        # Split by commas, but be careful of commas inside arrays
        param_parts = smart_split_parameters(params_str)
        
        for param_part in param_parts:
            if '=' in param_part:
                key, value = param_part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Parse the value (could be array, number, or string)
                if value.startswith('[') and value.endswith(']'):
                    # It's an array
                    parameters[key] = parse_array_parameter(value)
                elif value.isdigit():
                    # It's an integer
                    parameters[key] = int(value)
                elif is_float(value):
                    # It's a float
                    parameters[key] = float(value)
                else:
                    # It's a string (remove quotes if present)
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        parameters[key] = value[1:-1]
                    else:
                        parameters[key] = value
    
    return function_name, parameters

def smart_split_parameters(params_str):
    """
    Split parameter string by commas, but ignore commas inside arrays
    
    Args:
        params_str (str): Parameter string to split
        
    Returns:
        list: Split parameter parts
    """
    parts = []
    current_part = ""
    bracket_depth = 0
    in_quotes = False
    quote_char = None
    
    for char in params_str:
        if char in ['"', "'"] and not in_quotes:
            in_quotes = True
            quote_char = char
            current_part += char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
            current_part += char
        elif not in_quotes:
            if char == '[':
                bracket_depth += 1
                current_part += char
            elif char == ']':
                bracket_depth -= 1
                current_part += char
            elif char == ',' and bracket_depth == 0:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        else:
            current_part += char
    
    if current_part.strip():
        parts.append(current_part.strip())
    
    return parts

def parse_array_parameter(param_str):
    """
    Parse an array parameter like ["fiction", "sci_fi"] or ["Tokyo", "Seoul"]
    
    Args:
        param_str (str): Array parameter string
        
    Returns:
        list: Parsed array
    """
    # Remove outer brackets
    if param_str.startswith('[') and param_str.endswith(']'):
        inner_str = param_str[1:-1].strip()
    else:
        inner_str = param_str
    
    if not inner_str:
        return []
    
    # Split by comma and clean up
    items = []
    current_item = ""
    in_quotes = False
    quote_char = None
    
    for char in inner_str:
        if char in ['"', "'"] and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif char == ',' and not in_quotes:
            item = current_item.strip()
            if item:
                # Remove quotes if present
                if (item.startswith('"') and item.endswith('"')) or \
                   (item.startswith("'") and item.endswith("'")):
                    item = item[1:-1]
                items.append(item)
            current_item = ""
        else:
            current_item += char
    
    # Don't forget the last item
    if current_item.strip():
        item = current_item.strip()
        if (item.startswith('"') and item.endswith('"')) or \
           (item.startswith("'") and item.endswith("'")):
            item = item[1:-1]
        items.append(item)
    
    return items

def is_float(value):
    """
    Check if a string represents a float
    
    Args:
        value (str): String to check
        
    Returns:
        bool: True if it's a valid float
    """
    try:
        float(value)
        return '.' in value
    except ValueError:
        return False