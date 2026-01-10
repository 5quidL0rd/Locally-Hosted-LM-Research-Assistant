#!/usr/bin/env python3
"""
run_hf_model.py - Minimal interactive runner for a HuggingFace model cached in the project

Usage:
  python run_hf_model.py                # runs hf-internal-testing/tiny-random-gpt2 from local cache
  python run_hf_model.py --model distilgpt2 --max_new_tokens 80 --temperature 0.8

Notes:
- Ensure your venv is activated and `transformers` and `torch` are installed.
- The script uses the cache dir `hf_cache` by default, so it will reuse the model you already downloaded.
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--cache_dir", default="hf_cache")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)
    model.to(device)

    print("Interactive prompt (type 'exit' or Ctrl-C to quit).")
    while True:
        try:
            prompt = input("PROMPT> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt):] if generated.startswith(prompt) else generated
        print("\n" + continuation.strip() + "\n")


if __name__ == "__main__":
    main()
