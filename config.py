"""
Configuration file for LLM debate system.

This file contains model configurations and other settings
used across the debate system.
"""

# Model configuration

# DEBATE_MODEL = 'gpt-4o-mini'
# JUDGE_MODEL = 'gpt-4o-mini'

# DEBATE_MODEL = 'gemini-2.5-flash'
# JUDGE_MODEL = 'gpt-4o-mini'

# DEBATE_MODEL = 'gemini-2.5-flash'
# JUDGE_MODEL = 'gemini-2.5-flash'

DEBATE_MODEL = 'gemini-2.5-flash'
JUDGE_MODEL = 'gpt-3.5-turbo'

# DEBATE_MODEL = 'gemini-2.5-flash'
# JUDGE_MODEL = 'claude-3-haiku-20240307'

# DEBATE_MODEL = 'gemini-2.5-flash'
# JUDGE_MODEL = 'claude-3-5-haiku-20241022'

# DEBATE_MODEL = 'gpt-5-mini'
# DEBATE_MODEL_EFFORT = 'medium'  # Options: 'low', 'medium', 'high'
# JUDGE_MODEL = 'gpt-4o-mini'

# DEBATE_MODEL = 'gpt-4o-mini'
# JUDGE_MODEL = 'gemini-2.5-flash'

# Temperature settings
DIRECT_QA_TEMPERATURE = 0.0  # Temperature for initial direct QA tests
# JUDGE_DECISION_TEMPERATURE = 0.7  # Temperature for judge's debate decisions
# FINAL_VERDICT_TEMPERATURE = 0.3  # Temperature for judge's final verdict
JUDGE_DECISION_TEMPERATURE = 0.0  # Temperature for judge's debate decisions
FINAL_VERDICT_TEMPERATURE = 0.0  # Temperature for judge's final verdict

# Retry configuration
MAX_RETRIES = 2  # Maximum retry attempts for API calls (1 retry after initial attempt)
RETRY_BASE_WAIT = 2  # Base wait time in seconds for exponential backoff
API_TIMEOUT = 120  # Timeout in seconds for API calls (2 minutes) - NOTE: Converted to ms for Gemini

# Debate limits
MAX_TURNS_DEFAULT = 1  # Default maximum number of debate turns
DEBATER_WORD_LIMIT = 200  # Maximum words per debater response
NUM_CHOICES = 4  # Number of answer choices to use in debate (2-4 for GPQA)

# Debate mode configuration
DEBATE_MODE = 'non_interactive'  # Options: 'interactive', 'non_interactive', or 'both'
# - 'interactive': Judge can ask clarifying questions to debaters
# - 'non_interactive': Judge can only say 'next' or 'end'
# - 'both': Run both modes for comparison (doubles execution time)

# Debate style configuration
DEBATE_STYLE = 'simultaneous'  # Options: 'sequential' or 'simultaneous'
# - 'sequential': A speaks, then B speaks with A's response in history (1 turn = 1 response)
# - 'simultaneous': A and B both respond to same history (1 turn = both responses)
# NOTE: 'simultaneous' mode does NOT support 'interactive' DEBATE_MODE

# Baseline caching configuration
USE_BASELINE_CACHE = True  # Check cache for existing direct QA results before running
SAVE_TO_BASELINE_CACHE = True  # Save new direct QA results to cache
BASELINE_CACHE_DIR = './baseline_cache'  # Directory to store baseline cache files

# Dataset configuration (for cache validation)
DATASET_NAME = "Idavidrein/gpqa"
DATASET_SUBSET = "gpqa_diamond"
# DATASET_SUBSET = "gpqa_main"

DATASET_SPLIT = "train"

# Random seed configuration
# Run debates using run_parallel_debates.py (even for single debates with --num-debates 1)
# Master seed controls which questions are sampled (without replacement) for all debates
MASTER_SEED = 42  # Master seed for reproducibility (None = random question sampling)

# Cache key includes: model name, question_idx, temperature, dataset info
# This ensures cache is invalidated when models, datasets, or questions change
