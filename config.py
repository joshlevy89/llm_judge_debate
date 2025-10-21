"""
Configuration file for LLM debate system.

This file contains model configurations and other settings
used across the debate system.
"""

# Model configuration

# DEBATE_MODEL = 'gpt-4o-mini'
# JUDGE_MODEL = 'gpt-4o-mini'

DEBATE_MODEL = 'gemini-2.5-flash'
JUDGE_MODEL = 'gpt-4o-mini'

# DEBATE_MODEL = 'gemini-2.5-flash'
# JUDGE_MODEL = 'gemini-2.5-flash'

# DEBATE_MODEL = 'gemini-2.5-flash'
# JUDGE_MODEL = 'gpt-3.5-turbo'

# Temperature settings
DIRECT_QA_TEMPERATURE = 0.0  # Temperature for initial direct QA tests
JUDGE_DECISION_TEMPERATURE = 0.7  # Temperature for judge's debate decisions
FINAL_VERDICT_TEMPERATURE = 0.3  # Temperature for judge's final verdict

# Retry configuration
MAX_RETRIES = 2  # Maximum retry attempts for API calls (1 retry after initial attempt)
RETRY_BASE_WAIT = 2  # Base wait time in seconds for exponential backoff
API_TIMEOUT = 120  # Timeout in seconds for API calls (2 minutes) - NOTE: Converted to ms for Gemini

# Debate limits
MAX_TURNS_DEFAULT = 2  # Default maximum number of debate turns
DEBATER_WORD_LIMIT = 200  # Maximum words per debater response

# Debate mode configuration
DEBATE_MODE = 'non_interactive'  # Options: 'interactive', 'non_interactive', or 'both'
# - 'interactive': Judge can ask clarifying questions to debaters
# - 'non_interactive': Judge can only say 'next' or 'end'
# - 'both': Run both modes for comparison (doubles execution time)

# Baseline caching configuration
USE_BASELINE_CACHE = True  # Check cache for existing direct QA results before running
SAVE_TO_BASELINE_CACHE = True  # Save new direct QA results to cache
BASELINE_CACHE_DIR = './baseline_cache'  # Directory to store baseline cache files

# Dataset configuration (for cache validation)
DATASET_NAME = "Idavidrein/gpqa"
DATASET_SUBSET = "gpqa_diamond"
DATASET_SPLIT = "train"

# Random seed configuration
# Run debates using run_parallel_debates.py (even for single debates with --num-debates 1)
# Master seed controls which questions are sampled (without replacement) for all debates
MASTER_SEED = 42  # Master seed for reproducibility (None = random question sampling)

# Cache key includes: model name, question_idx, temperature, dataset info
# This ensures cache is invalidated when models, datasets, or questions change
