import re


def clean_response(response):
    """
    Fast response cleaning - minimal regex for speed
    """
    try:
        # If response is a dict/list, convert to string
        if isinstance(response, (dict, list)):
            response_str = str(response)
            if isinstance(response, dict) and 'text' in response:
                response_str = response['text']
            elif isinstance(response, list) and len(response) > 0:
                if isinstance(response[0], dict) and 'text' in response[0]:
                    response_str = response[0]['text']
        else:
            response_str = str(response)
        
        # Minimal cleaning - only remove signature metadata
        response_str = re.sub(r"'signature':[^}]*}?", "", response_str, flags=re.DOTALL)
        response_str = re.sub(r"'extras':\s*\{[^}]*\}", "", response_str)
        
        # Remove excessive newlines only
        response_str = re.sub(r"\n\s*\n\s*\n+", "\n\n", response_str)
        
        return response_str.strip().strip("'\"{}[]")
    
    except Exception:
        return str(response)
