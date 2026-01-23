# DEBATER_MODEL = "x-ai/grok-4-fast"
# DEBATER_MODEL = "x-ai/grok-4.1-fast"
# DEBATER_MODEL = "google/gemini-3-flash-preview"
# DEBATER_MODEL = "openai/gpt-4o-mini"
# DEBATER_MODEL = "meta-llama/llama-3-8b-instruct"
# DEBATER_MODEL = "meta-llama/llama-3.1-8b-instruct"
# DEBATER_MODEL = "meta-llama/llama-3.1-70b-instruct"
# DEBATER_MODEL = "meta-llama/llama-3.1-405b-instruct"
# DEBATER_MODEL = "meta-llama/llama-3.3-70b-instruct"
# DEBATER_MODEL = "meta-llama/llama-4-scout"
# DEBATER_MODEL = "meta-llama/llama-4-maverick"
# DEBATER_MODEL = "openai/gpt-3.5-turbo"
# DEBATER_MODEL = "google/gemma-3-12b-it"
# DEBATER_MODEL = "google/gemma-3-27b-it"
# DEBATER_MODEL = "qwen/qwen-2.5-72b-instruct"
DEBATER_MODEL = "moonshotai/kimi-k2-thinking"
# DEBATER_MODEL = "moonshotai/kimi-k2-0905"

# DEBATER_MODEL = "deepseek/deepseek-v3.2-exp"
# DEBATER_MODEL = "google/gemini-3-pro-preview"
# DEBATER_MODEL = "openai/o3"
# DEBATER_MODEL = "openai/gpt-oss-120b"

DATASET_NAME = "Idavidrein/gpqa"
DATASET_SUBSET = "gpqa_diamond"
# DATASET_SUBSET = "gpqa_main"
DATASET_SPLIT = "train"
DATASET_FILTERS = None
DATASET_PATH = None  # Only used when DATASET_NAME = "local"

# Local dataset example (llama_binary):
# DATASET_NAME = "local"
# DATASET_SUBSET = None
# DATASET_SPLIT = None
# DATASET_FILTERS = None
# DATASET_PATH = "datasets/data/llama_binary_20260123_094608.json"

# DATASET_NAME = "m-a-p/SuperGPQA"
# DATASET_SUBSET = None
# DATASET_SPLIT = "train"
# # additional filters that can be applied
# DATASET_FILTERS = {  
#     'field': 'Physics',
#     'difficulty': 'hard'
# }
                
# DATASET_NAME = "TIGER-Lab/MMLU-Pro"
# DATASET_SUBSET = None
# DATASET_SPLIT = "test"
# DATASET_FILTERS = None
# DATASET_FILTERS = {
#     "category": ["physics", "chemistry"]
# }

DEBATER_TEMPERATURE = 0.5

MAX_OUTPUT_TOKENS = 20000
# MAX_OUTPUT_TOKENS = 10000
# MAX_OUTPUT_TOKENS = 5000

DEBATER_REASONING_EFFORT = None # 'medium' # 'low' # 'high' # 'low' # None # 'high' # None # 'low'
DEBATER_REASONING_MAX_TOKENS = None
REASONING_ENABLED=None
PROVIDE_INTERNAL_REASONING_TO_SELF = True # When True, the debater will have access to its own past thought processes

PRIVATE_SCRATCHPAD = False

PUBLIC_ARGUMENT_WORD_LIMIT = None
PRIVATE_REASONING_WORD_LIMIT = None

LENIENT_PARSING_ARGUMENT = False

NUM_QUESTIONS = 198 # TODO 
RANDOM_SEED = 42
NUM_CHOICES = 2  
NUM_TURNS = 6 # TODO

MAX_THREADS = 200

DEBATE_MODE = "sequential"
# DEBATE_MODE = "simultaneous"

INCLUDE_CLOSING_ARGUMENTS=True # TODO 

# INTERACTIVE_JUDGE = "x-ai/grok-4-fast"
# INTERACTIVE_JUDGE_TEMPERATURE = 0.3
# INTERACTIVE_JUDGE_MAX_OUTPUT_TOKENS = 500
# INTERACTIVE_JUDGE_REASONING_EFFORT = None
# INTERACTIVE_JUDGE_REASONING_MAX_TOKENS = None

INTERACTIVE_JUDGE = None
INTERACTIVE_JUDGE_TEMPERATURE = None
INTERACTIVE_JUDGE_MAX_OUTPUT_TOKENS = None
INTERACTIVE_JUDGE_REASONING_EFFORT = None
INTERACTIVE_JUDGE_REASONING_MAX_TOKENS = None
INTERACTIVE_JUDGE_REASONING_ENABLED = None

MOCK_DEBATE_RESPONSE = False
MOCK_INTERACTIVE_JUDGE_RESPONSE = False

# SPECIFIC_IDXS = [32, 92, 125, 63, 136, 179, 123, 97, 9, 129]
# SPECIFIC_IDXS = [176, 76, 118, 149, 56, 139, 3, 142, 21, 182, 22, 33, 192, 105, 110, 189, 68, 45, 75, 173]
# SPECIFIC_IDXS = [125]
# SPECIFIC_IDXS = None
# SPECIFIC_IDXS = [7894, 25241, 18343, 22468, 7759] # [4880] # [25803, 5627, 4880, 10347, 10561]
# SPECIFIC_IDXS = [10561]
# SPECIFIC_IDXS = [25803, 5627, 4880, 10347, 10561]
# SPECIFIC_IDXS = [19485, 18347, 12849, 14191, 25441, 861, 12266, 12563, 24926, 6143, 5868, 21169, 6120, 10361, 5136, 4049, 23588, 5021, 19402, 7567]
# SPECIFIC_IDXS = [12563]
# SPECIFIC_IDXS = [18526, 8530, 1265, 16177, 16245, 1262, 7451, 20988, 21328, 14734, 3924, 10491, 20764, 14412]
# SPECIFIC_IDXS = [10361, 5136, 4049, 23588, 5021, 19402, 7567, 1904, 11812, 14101, 5984, 11661, 14195]
# SPECIFIC_IDXS = "results/question_idxs_mmlu_stratified_100_per_category.txt"
SPECIFIC_IDXS = None



########################################################################################################################################################################
# # HUMAN CONFIG 
# # DEBATER_MODEL = "x-ai/grok-4-fast"
# # DEBATER_MODEL = "x-ai/grok-4"
# DEBATER_MODEL = "moonshotai/kimi-k2-thinking"

# # DATASET_NAME = "Idavidrein/gpqa"
# # DATASET_SUBSET = "gpqa_diamond"
# # DATASET_SPLIT = "train"

# DATASET_NAME = "TIGER-Lab/MMLU-Pro"
# DATASET_SUBSET = None
# DATASET_SPLIT = "test"

# DEBATER_TEMPERATURE = 0.3

# MAX_OUTPUT_TOKENS = 5000
# DEBATER_REASONING_EFFORT = None #'low'
# DEBATER_REASONING_MAX_TOKENS = 1000

# PRIVATE_SCRATCHPAD = False

# PUBLIC_ARGUMENT_WORD_LIMIT = 200
# PRIVATE_REASONING_WORD_LIMIT = 1000

# LENIENT_PARSING_ARGUMENT = False

# NUM_QUESTIONS = 1
# RANDOM_SEED = 42
# NUM_CHOICES = 2
# NUM_TURNS = 100

# MAX_THREADS = 5000

# DEBATE_MODE = "sequential"

# MOCK_DEBATE_RESPONSE = False
# MOCK_INTERACTIVE_JUDGE_RESPONSE = False