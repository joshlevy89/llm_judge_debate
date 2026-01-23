import json
import random
from pathlib import Path


def load_local_dataset(dataset_path, filter_valid=True):
    """Load a local JSON dataset file. Returns list of items."""
    path = Path(dataset_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    
    if filter_valid:
        results = [r for r in results if r.get('llama_answer') is not None]
    
    return results, data.get('metadata', {})


def parse_llama_binary_item(item):
    """Parse an item from the llama_binary dataset."""
    question = item.get('question')
    options = item.get('options', [])
    gt_idx = item.get('gt_idx')
    correct_answer = options[gt_idx] if gt_idx is not None and gt_idx < len(options) else None
    return question, correct_answer, options


def parse_gpqa_item(item):
    question = item.get('Question')
    correct_answer = item.get('Correct Answer')
    all_choices = [
        item.get('Correct Answer'),
        item.get('Incorrect Answer 1'),
        item.get('Incorrect Answer 2'),
        item.get('Incorrect Answer 3')
    ]
    all_choices = [c.strip() for c in all_choices if c is not None]
    correct_answer = correct_answer.strip() if correct_answer else None
    return question, correct_answer, all_choices

def parse_mmlu_pro_item(item):
    question = item.get('question')
    options = item.get('options', [])
    answer_index = item.get('answer_index')
    all_choices = [opt.strip() for opt in options if opt]
    correct_answer = all_choices[answer_index] if answer_index is not None and answer_index < len(all_choices) else None
    return question, correct_answer, all_choices


def parse_supergpqa_item(item):
    question = item.get('question')
    options = item.get('options', [])
    correct_answer = item.get('answer')
    all_choices = [opt.strip() for opt in options if opt]
    return question, correct_answer, all_choices


def select_questions_and_options(dataset_name, dataset, num_questions, num_choices, seed, specific_idxs=None):

    if specific_idxs is not None:
        question_indices = specific_idxs
    else:
        rng_questions = random.Random(seed)
        total_questions = len(dataset)
        question_indices = rng_questions.sample(range(total_questions), min(num_questions, total_questions))
    
    results = []
    for idx in question_indices:
        item = dataset[idx]
                    
        # Parse dataset item based on dataset name
        if dataset_name == "Idavidrein/gpqa":
            question, correct_answer, all_choices = parse_gpqa_item(item)
        elif dataset_name == "TIGER-Lab/MMLU-Pro":
            question, correct_answer, all_choices = parse_mmlu_pro_item(item)
        elif dataset_name == 'm-a-p/SuperGPQA':
            question, correct_answer, all_choices = parse_supergpqa_item(item)
        elif dataset_name == "local":
            question, correct_answer, all_choices = parse_llama_binary_item(item)
            # For local datasets, options are pre-selected, so skip reshuffling
            results.append({
                'question': question,
                'options': all_choices,
                'correct_idx': item.get('gt_idx'),
                'original_idx': item.get('idx', idx)
            })
            continue
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Use question index as seed for reproducible option selection
        rng_options = random.Random(idx)
        
        # Select the number of choices requested
        if len(all_choices) >= num_choices:
            incorrect_choices = [c for c in all_choices if c != correct_answer]
            num_incorrect = num_choices - 1
            selected_incorrect = rng_options.sample(incorrect_choices, num_incorrect)
            selected_options = [correct_answer] + selected_incorrect
        else:
            selected_options = all_choices
        
        # Shuffle the selected options
        rng_options.shuffle(selected_options)
        
        # Find correct answer position
        correct_idx = selected_options.index(correct_answer)
        
        results.append({
            'question': question,
            'options': selected_options,
            'correct_idx': correct_idx,
            'original_idx': idx
        })
    
    return results

def format_options(options):
    options_text = ""
    for i, option in enumerate(options):
        options_text += f"{i}. {option}\n"
    return options_text.strip()


class LocalDatasetWrapper:
    """Wraps a local dataset list to provide a similar interface to HuggingFace datasets."""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def add_column(self, name, values):
        for i, item in enumerate(self.data):
            item[name] = values[i]
        return self
    
    def filter(self, fn):
        filtered = [item for item in self.data if fn(item)]
        return LocalDatasetWrapper(filtered)


def load_dataset_unified(dataset_name, dataset_subset=None, dataset_split=None, dataset_path=None):
    """
    Unified dataset loading for both HuggingFace and local datasets.
    
    For local datasets: set dataset_name="local" and provide dataset_path.
    For HuggingFace: provide dataset_name, dataset_subset, and dataset_split.
    
    Returns (dataset, metadata) where metadata is only populated for local datasets.
    """
    if dataset_name == "local":
        if not dataset_path:
            raise ValueError("dataset_path required for local datasets")
        data, metadata = load_local_dataset(dataset_path, filter_valid=True)
        return LocalDatasetWrapper(data), metadata
    else:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, dataset_subset)[dataset_split]
        return dataset, {}


def stratified_split(indices, labels, train_size, val_size, test_size, seed=42):
    """
    Split indices into train/val/test sets, stratified by labels.
    
    Args:
        indices: List of indices to split
        labels: List of labels (same length as indices), used for stratification
        train_size: Number of samples for training
        val_size: Number of samples for validation
        test_size: Number of samples for testing
        seed: Random seed for reproducibility
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = random.Random(seed)
    
    label_to_indices = {}
    for idx, label in zip(indices, labels):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    for label in label_to_indices:
        rng.shuffle(label_to_indices[label])
    
    total_requested = train_size + val_size + test_size
    total_available = len(indices)
    
    if total_requested > total_available:
        raise ValueError(f"Requested {total_requested} samples but only {total_available} available")
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, label_indices in label_to_indices.items():
        n_label = len(label_indices)
        label_ratio = n_label / total_available
        
        n_train = round(train_size * label_ratio)
        n_val = round(val_size * label_ratio)
        n_test = round(test_size * label_ratio)
        
        n_total = n_train + n_val + n_test
        if n_total > n_label:
            excess = n_total - n_label
            if n_train > excess:
                n_train -= excess
            else:
                n_val -= min(n_val, excess - n_train)
                n_train = 0
        
        train_indices.extend(label_indices[:n_train])
        val_indices.extend(label_indices[n_train:n_train + n_val])
        test_indices.extend(label_indices[n_train + n_val:n_train + n_val + n_test])
    
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    
    return train_indices, val_indices, test_indices