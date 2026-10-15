# parser.py
import re   # Import Python's regular expression library to help match patterns in text

def parse_function_call(text):
    """
    Parse function call from text in format: CALL_FUNCTION:function_name:param1=value1,param2=value2   
    Args:
        text (str): Text that may contain a function call        
    Returns:
        tuple: (function_name, parameters) or (None, None) if no function call found
    """
    # Define regex pattern to match the function call format
    # Example string: "CALL_FUNCTION:add_numbers:num1=5,num2=3"
    # Group 1 → function name (e.g., add_numbers)
    # Group 2 → parameter string (e.g., num1=5,num2=3)
    pattern = r"CALL_FUNCTION:(\w+):(.+)"

    # Try to find a match in the given text
    match = re.search(pattern, text.strip())

    # If no match is found, return (None, None) to indicate no function call
    if not match:
        return None, None

    # Extract the function name from the first regex group
    function_name = match.group(1)

    # Extract the parameter string from the second regex group
    params_str = match.group(2)

    # Initialize an empty dictionary to store parsed parameters
    parameters = {}

    # Split the parameter string by commas to get each "key=value" pair
    for param in params_str.split(","):
        if "=" in param:  # Ensure the pair contains an equals sign
            # Split into key and value
            key, value = param.split("=", 1)
            key = key.strip()     # Remove any extra spaces around the key
            value = value.strip() # Remove any extra spaces around the value

            try:
                # Try converting the value to a number
                if "." in value:        # If it contains a decimal point, convert to float
                    parameters[key] = float(value)
                else:                   # Otherwise, convert to integer
                    parameters[key] = int(value)
            except ValueError:
                # If conversion fails (e.g., non-numeric string), keep the raw string
                parameters[key] = value

    # Return the function name and the dictionary of parameters
    return function_name, parameters
