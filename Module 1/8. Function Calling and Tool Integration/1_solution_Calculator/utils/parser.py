import re

def parse_function_call(text):
    """
    Parse function call from text in format: CALL_FUNCTION:function_name:param1=value1,param2=value2
    
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
    
    # Parse parameters
    parameters = {}
    if params_str:
        for param_pair in params_str.split(","):
            if "=" in param_pair:
                key, value = param_pair.split("=", 1)
                parameters[key.strip()] = value.strip()
    
    return function_name, parameters
