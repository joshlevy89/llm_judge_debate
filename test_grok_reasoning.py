import os
from dotenv import load_dotenv
from utils.llm_utils import call_openrouter

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

model = "x-ai/grok-4.1-fast"
prompt = "What is 2 + 2?"
max_tokens = 5000

def get_reasoning_tokens_from_usage(usage):
    completion_details = usage.get('completion_tokens_details', {})
    return completion_details.get('reasoning_tokens', 0)

# Test with reasoning explicitly disabled
print("Testing with reasoning_enabled=False...")
response_disabled, usage_disabled = call_openrouter(
    prompt=prompt,
    model_name=model,
    api_key=api_key,
    max_tokens=max_tokens,
    reasoning_enabled=False
)

# Test with low reasoning
print("Testing with reasoning_effort='low'...")
response_low, usage_low = call_openrouter(
    prompt=prompt,
    model_name=model,
    api_key=api_key,
    max_tokens=max_tokens,
    reasoning_effort="low"
)

print("\n=== Results ===")
print(f"\nReasoning disabled (reasoning_enabled=False):")
print(f"  Reasoning tokens: {get_reasoning_tokens_from_usage(usage_disabled)}")
print(f"  Total completion tokens: {usage_disabled.get('completion_tokens', 'N/A')}")

print(f"\nLow reasoning effort:")
print(f"  Reasoning tokens: {get_reasoning_tokens_from_usage(usage_low)}")
print(f"  Total completion tokens: {usage_low.get('completion_tokens', 'N/A')}")

# Validate
disabled_reasoning = get_reasoning_tokens_from_usage(usage_disabled)
low_reasoning = get_reasoning_tokens_from_usage(usage_low)

print("\n=== Validation ===")
if disabled_reasoning == 0:
    print("PASS: No reasoning tokens when reasoning_enabled=False")
else:
    print(f"FAIL: Expected 0 reasoning tokens, got {disabled_reasoning}")

if low_reasoning > 0:
    print(f"PASS: Low effort produced {low_reasoning} reasoning tokens")
else:
    print("FAIL: Expected reasoning tokens with effort='low', got 0")
