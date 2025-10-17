#!/usr/bin/env python3
"""
Baseline Cache Module

This module handles caching of direct QA results to avoid re-running stable baselines
when experimenting with different debate parameters.

Cache Structure:
- Separate JSON file per model (e.g., baseline_debater_gemini-2.5-flash.json)
- Keyed by question_idx
- Includes model name, temperature, and timestamp for validation
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from config import BASELINE_CACHE_DIR, DEBATE_MODEL, JUDGE_MODEL, DIRECT_QA_TEMPERATURE


def _get_cache_filename(model_name: str, model_type: str) -> str:
    """
    Get the cache filename for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'gpt-3.5-turbo')
        model_type: Type of model ('debater' or 'judge')
    
    Returns:
        Filename for the cache file
    """
    # Sanitize model name for filename
    safe_model_name = model_name.replace('/', '_').replace(':', '_')
    return f"baseline_{model_type}_{safe_model_name}_temp{DIRECT_QA_TEMPERATURE}.json"


def _load_cache(cache_file: Path) -> Dict:
    """
    Load cache from JSON file.
    
    Args:
        cache_file: Path to cache file
    
    Returns:
        Dictionary with cache data, or empty structure if file doesn't exist
    """
    if not cache_file.exists():
        return {
            "metadata": {
                "model_name": None,
                "model_type": None,
                "temperature": DIRECT_QA_TEMPERATURE,
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "results": {}
        }
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load cache file {cache_file}: {e}")
        return {
            "metadata": {
                "model_name": None,
                "model_type": None,
                "temperature": DIRECT_QA_TEMPERATURE,
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "results": {}
        }


def _save_cache(cache_file: Path, cache_data: Dict) -> None:
    """
    Save cache to JSON file.
    
    Args:
        cache_file: Path to cache file
        cache_data: Dictionary with cache data
    """
    # Ensure directory exists
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Update last_updated timestamp
    cache_data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")


def get_cached_qa(question_idx: int, model_type: str) -> Optional[Dict]:
    """
    Get cached direct QA result for a question and model.
    
    Args:
        question_idx: Index of the question
        model_type: Type of model ('debater' or 'judge')
    
    Returns:
        Cached QA result dictionary, or None if not found
    """
    # Determine which model to use
    model_name = DEBATE_MODEL if model_type == 'debater' else JUDGE_MODEL
    
    # Get cache file path
    cache_filename = _get_cache_filename(model_name, model_type)
    cache_file = Path(BASELINE_CACHE_DIR) / cache_filename
    
    # Load cache
    cache_data = _load_cache(cache_file)
    
    # Check if result exists for this question
    question_key = str(question_idx)
    if question_key in cache_data["results"]:
        result = cache_data["results"][question_key]
        # Validate that temperature matches
        if result.get("temperature") == DIRECT_QA_TEMPERATURE:
            return result
        else:
            print(f"Warning: Cached result for question {question_idx} has mismatched temperature. Ignoring cache.")
            return None
    
    return None


def save_qa_to_cache(question_idx: int, model_type: str, qa_result: Dict) -> None:
    """
    Save direct QA result to cache.
    
    Args:
        question_idx: Index of the question
        model_type: Type of model ('debater' or 'judge')
        qa_result: QA result dictionary from test_model_direct_qa
    """
    # Determine which model to use
    model_name = DEBATE_MODEL if model_type == 'debater' else JUDGE_MODEL
    
    # Get cache file path
    cache_filename = _get_cache_filename(model_name, model_type)
    cache_file = Path(BASELINE_CACHE_DIR) / cache_filename
    
    # Load existing cache
    cache_data = _load_cache(cache_file)
    
    # Update metadata
    cache_data["metadata"]["model_name"] = model_name
    cache_data["metadata"]["model_type"] = model_type
    cache_data["metadata"]["temperature"] = DIRECT_QA_TEMPERATURE
    
    # Prepare result to cache (exclude raw_response to save space)
    cached_result = {
        "selected_letter": qa_result.get("selected_letter"),
        "selected_answer": qa_result.get("selected_answer"),
        "is_correct": qa_result.get("is_correct"),
        "confidence": qa_result.get("confidence"),
        "reasoning": qa_result.get("reasoning"),
        "temperature": DIRECT_QA_TEMPERATURE,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to cache
    question_key = str(question_idx)
    cache_data["results"][question_key] = cached_result
    
    # Write to file
    _save_cache(cache_file, cache_data)


def clear_cache(model_type: Optional[str] = None) -> None:
    """
    Clear cache files.
    
    Args:
        model_type: If specified, only clear cache for this model type ('debater' or 'judge')
                   If None, clear all cache files
    """
    cache_dir = Path(BASELINE_CACHE_DIR)
    
    if not cache_dir.exists():
        return
    
    if model_type:
        model_name = DEBATE_MODEL if model_type == 'debater' else JUDGE_MODEL
        cache_filename = _get_cache_filename(model_name, model_type)
        cache_file = cache_dir / cache_filename
        if cache_file.exists():
            cache_file.unlink()
            print(f"Cleared cache file: {cache_file}")
    else:
        # Clear all cache files
        for cache_file in cache_dir.glob("baseline_*.json"):
            cache_file.unlink()
            print(f"Cleared cache file: {cache_file}")


def get_cache_stats(model_type: str) -> Dict:
    """
    Get statistics about the cache for a model type.
    
    Args:
        model_type: Type of model ('debater' or 'judge')
    
    Returns:
        Dictionary with cache statistics
    """
    model_name = DEBATE_MODEL if model_type == 'debater' else JUDGE_MODEL
    cache_filename = _get_cache_filename(model_name, model_type)
    cache_file = Path(BASELINE_CACHE_DIR) / cache_filename
    
    cache_data = _load_cache(cache_file)
    
    return {
        "model_name": cache_data["metadata"].get("model_name"),
        "model_type": cache_data["metadata"].get("model_type"),
        "temperature": cache_data["metadata"].get("temperature"),
        "created": cache_data["metadata"].get("created"),
        "last_updated": cache_data["metadata"].get("last_updated"),
        "num_cached_results": len(cache_data["results"]),
        "cache_file": str(cache_file),
        "cache_exists": cache_file.exists()
    }

