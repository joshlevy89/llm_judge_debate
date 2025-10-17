"""
Configuration file for LLM debate system.

This file contains model configurations and other settings
used across the debate system.
"""

# Model configuration
DEBATE_MODEL = 'gpt-4o-mini'
JUDGE_MODEL = 'gpt-4o-mini'

# Temperature settings
DIRECT_QA_TEMPERATURE = 0.0  # Temperature for initial direct QA tests
JUDGE_DECISION_TEMPERATURE = 0.7  # Temperature for judge's debate decisions
FINAL_VERDICT_TEMPERATURE = 0.3  # Temperature for judge's final verdict

# Retry configuration
MAX_RETRIES = 3  # Maximum retry attempts for API calls
RETRY_BASE_WAIT = 2  # Base wait time in seconds for exponential backoff

# Debate limits
MAX_TURNS_DEFAULT = 20  # Default maximum number of debate turns
DEBATER_WORD_LIMIT = 200  # Maximum words per debater response

