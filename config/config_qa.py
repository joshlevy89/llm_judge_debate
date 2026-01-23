DATASET_NAME = "Idavidrein/gpqa"
DATASET_SUBSET = "gpqa_diamond"
DATASET_SPLIT = "train"
DATASET_FILTERS = None
DATASET_PATH = None  # Only used when DATASET_NAME = "local"

# Local dataset example (llama_binary):
# DATASET_NAME = "local"
# DATASET_SUBSET = None
# DATASET_SPLIT = None
# DATASET_FILTERS = None
# DATASET_PATH = "datasets/data/llama_binary_20260123_094608.json"

# DATASET_NAME = "TIGER-Lab/MMLU-Pro"
# DATASET_SUBSET = None
# DATASET_SPLIT = "test"
# DATASET_FILTERS = {
#     "category": "physics"
# }
# DATASET_FILTERS = None

# DATASET_NAME = "m-a-p/SuperGPQA"
# DATASET_SUBSET = None
# DATASET_SPLIT = "train"
# # additional filters that can be applied
# DATASET_FILTERS = {  
#     'field': 'Physics',
#     'difficulty': 'hard'
# }

PROMPT_FORMAT = None  # None for default, 'simple' for simplified, 'simple_no_thinking' for no thinking

# MODEL_NAME = "google/gemini-3-flash-preview"
# MODEL_NAME = "moonshotai/kimi-k2-thinking"
# MODEL_NAME = "moonshotai/kimi-k2-0905"
# MODEL_NAME = "meta-llama/llama-4-maverick"
# MODEL_NAME = "openai/gpt-4o-mini" 
MODEL_NAME = "x-ai/grok-4-fast" 
# MODEL_NAME = "x-ai/grok-4.1-fast:free"
# MODEL_NAME = "openai/gpt-3.5-turbo"
# MODEL_NAME = "qwen/qwen-2.5-7b-instruct" 
# MODEL_NAME = "meta-llama/llama-3-8b-instruct"
# MODEL_NAME = "qwen/qwen3-8b"
# MODEL_NAME = "qwen/qwen3-235b-a22b"

# MODEL_NAME = "deepseek/deepseek-v3.1-terminus"
# MODEL_NAME = "deepseek/deepseek-r1-distill-llama-70b"

# MODEL_NAME = "openai/o3"
# MODEL_NAME = "openai/gpt-oss-120b"

# MODEL_NAME = "google/gemini-3-pro-preview"

# MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

TEMPERATURE = 0.5 # 1.0 # 0.0 # 0.5

MAX_TOKENS = 10000
REASONING_EFFORT = None # 'low'
REASONING_MAX_TOKENS = None
REASONING_ENABLED = False

NUM_QUESTIONS = 198
RANDOM_SEED = 42 
NUM_CHOICES = 2

RERUN = True  # will rerun a question/choice set even if it already exists
LENIENT_PARSING = False

# SPECIFIC_QUESTION_IDXS = "results/question_idxs_mmlu_stratified_100_per_category.txt" # [183] # [167] # List of specific question indices to run QA for, e.g., [0, 5, 10]
SPECIFIC_QUESTION_IDXS = None

MAX_THREADS = 200
