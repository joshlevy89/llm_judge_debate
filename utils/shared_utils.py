import random
import string
import yaml
import unicodeit

def generate_run_id():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

def extract_config(config_module):
    """
    Extract all UPPERCASE config variables from a module and return as a dict.
    
    Args:
        config_module: The config module to extract from
    
    Returns:
        dict: Config variables with lowercase keys
    """
    config = {}
    for key in dir(config_module):
        if key.isupper() and not key.startswith('_'):
            value = getattr(config_module, key)
            config[key.lower()] = value
    return config


def load_prompts(prompt_type, variant=None):
    if prompt_type == 'debate':
        with open('prompts/debater_prompts.yaml', 'r') as f:
            prompts = yaml.safe_load(f)
        return prompts
    elif prompt_type == 'interactive':
        with open('prompts/interactive_prompts.yaml', 'r') as f:
            prompts = yaml.safe_load(f)
        return prompts['action_prompt_template']
    elif prompt_type == 'judge':
        with open('prompts/judge_prompts.yaml', 'r') as f:
            prompts = yaml.safe_load(f)
        return prompts['judge_prompt_template']
    elif prompt_type == 'shared':
        with open('prompts/shared_prompts.yaml', 'r') as f:
            prompts = yaml.safe_load(f)
        if variant == 'simple':
            return prompts['response_format_prompt_simple']
        elif variant == 'simple_no_thinking':
            return prompts['response_format_prompt_simple_no_thinking']
        return prompts['response_format_prompt']
    elif prompt_type == 'qa':
        with open('prompts/qa_prompts.yaml', 'r') as f:
            prompts = yaml.safe_load(f)
        return prompts['qa_prompt_template']
    elif prompt_type == 'leak':
        with open('prompts/leak_prompts.yaml', 'r') as f:
            prompts = yaml.safe_load(f)
        return prompts['check_prompt_template']
    

def format_latex(text):
    import re
    text = str(text)
    text = text.replace('\\\\', '\\')
    text = text.replace('$', '')
    
    text = text.replace('\\cdot', '·')
    text = text.replace('\\times', '×')
    text = text.replace('\\approx', '≈')
    
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    
    def replace_sqrt(match):
        content = match.group(1)
        return f'√({content})'
    text = re.sub(r'\\sqrt\{([^}]*)\}', replace_sqrt, text)
    
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', text)
    
    text = re.sub(r'\\[()\[\]]', '', text)
    text = re.sub(r'\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}', r'[\1]', text, flags=re.DOTALL)
    
    return unicodeit.replace(text)