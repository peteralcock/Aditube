import argparse
from typing import List

try:
    from transformers import pipeline
    import torch  # noqa: F401  # ensures backend libs are available
except Exception:
    pipeline = None

def generate_banner_text(prompt: str, max_tokens: int = 60) -> str:
    """Generate banner ad text using a text-generation model.

    If the transformers library is installed, a small GPT-Neo model is
    used. Otherwise, the function returns a placeholder string.
    """
    if pipeline is None:
        return "[Transformers unavailable] " + prompt

    try:
        generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
        result: List[dict] = generator(prompt, max_length=max_tokens, num_return_sequences=1)
        return result[0]["generated_text"].strip()
    except Exception as exc:
        return f"[Generation failed: {exc}] {prompt}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate banner ad copy using an LLM")
    parser.add_argument("prompt", help="Product information to build the ad")
    parser.add_argument("-m", "--max_tokens", type=int, default=60, help="Maximum tokens in the output")
    args = parser.parse_args()

    text = generate_banner_text(args.prompt, max_tokens=args.max_tokens)
    print(text)

if __name__ == "__main__":
    main()

