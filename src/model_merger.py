"""
Model Merger Script
Merges LoRA adapters with the base Llama-3.2-1B model for standalone deployment.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path


def merge_lora_model(
    base_model_name: str = "meta-llama/Llama-3.2-1B",
    adapter_path: str = None,
    output_path: str = None,
    use_auth_token: bool = True
) -> tuple:
    """
    Merge LoRA adapters with base model.

    Args:
        base_model_name: HuggingFace model identifier
        adapter_path: Path to LoRA adapter directory
        output_path: Path to save merged model
        use_auth_token: Whether to use HuggingFace auth token

    Returns:
        Tuple of (merged_model, tokenizer)
    """

    if adapter_path is None:
        # Use default adapter path from project
        adapter_path = "/home/shivam/pikky/models-20251126T202424Z-1-001/models/medical_chatbot_llama2_qlora"

    if output_path is None:
        output_path = "/home/shivam/pikky/models/merged_model"

    adapter_path = Path(adapter_path)
    output_path = Path(output_path)

    # Verify adapter exists
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    print(f"Loading base model: {base_model_name}")
    print(f"Adapter path: {adapter_path}")
    print(f"Output path: {output_path}")
    print("-" * 80)

    try:
        # Load base model
        print("Step 1: Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_auth_token=use_auth_token,
        )
        print("✓ Base model loaded")

        # Load tokenizer
        print("\nStep 2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_auth_token=use_auth_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✓ Tokenizer loaded")

        # Load LoRA model
        print("\nStep 3: Loading LoRA adapters...")
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            device_map="auto",
        )
        print("✓ LoRA adapters loaded")

        # Merge adapters into base model
        print("\nStep 4: Merging LoRA adapters with base model...")
        merged_model = model.merge_and_unload()
        print("✓ Adapters merged successfully")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Save merged model
        print(f"\nStep 5: Saving merged model to {output_path}...")
        merged_model.save_pretrained(
            str(output_path),
            safe_serialization=True,
        )
        print("✓ Merged model saved")

        # Save tokenizer
        print("\nStep 6: Saving tokenizer...")
        tokenizer.save_pretrained(str(output_path))
        print("✓ Tokenizer saved")

        print("\n" + "=" * 80)
        print("✓ Model merging completed successfully!")
        print(f"Merged model saved to: {output_path}")
        print("=" * 80)

        return merged_model, tokenizer

    except Exception as e:
        print(f"\n✗ Error during model merging: {str(e)}")
        raise


def verify_merged_model(model_path: str) -> bool:
    """
    Verify that merged model can be loaded and used.

    Args:
        model_path: Path to merged model

    Returns:
        True if model loads successfully
    """
    try:
        print(f"Verifying merged model at: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Test generation
        print("Testing model inference...")
        inputs = tokenizer("Question: What is diabetes?", return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Sample output: {decoded[:100]}...")
        print("✓ Model verification successful")
        return True

    except Exception as e:
        print(f"✗ Model verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    print("Medical Chatbot - Model Merger")
    print("=" * 80)

    # Check if you have the adapter path
    adapter_path = "/home/shivam/pikky/models-20251126T202424Z-1-001/models/medical_chatbot_llama2_qlora"
    if not Path(adapter_path).exists():
        print(f"✗ Adapter path not found: {adapter_path}")
        print("\nPlease ensure the LoRA adapter files exist at the specified path.")
        sys.exit(1)

    try:
        # Merge model
        merged_model, tokenizer = merge_lora_model()

        # Verify merged model
        output_path = "/home/shivam/pikky/models/merged_model"
        if verify_merged_model(output_path):
            print("\n✓ All checks passed! Your merged model is ready for deployment.")
        else:
            print("\n⚠ Model verification had issues, but merging completed.")

    except Exception as e:
        print(f"\n✗ Failed to merge model: {str(e)}")
        sys.exit(1)
