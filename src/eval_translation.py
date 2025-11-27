"""
Translation evaluation script for Spanish-English translation using OPUS Global Voices corpus.
Evaluates the trained model on translation tasks in both directions and computes BLEU scores.
"""

import argparse
import json
import os
from typing import List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sacrebleu
from sacrebleu.metrics import BLEU
import numpy as np

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype: Optional[torch.dtype] = None
):
    """Load the trained model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    print(f"Model loaded on {device}")
    
    return model, tokenizer


def create_translation_prompt(
    source_text: str,
    direction: str,
    prompt_template: Optional[str] = None
) -> str:
    """Create a prompt for translation task."""
    if prompt_template:
        # Use custom template with {source} placeholder
        prompt = prompt_template.replace("{source}", source_text)
    elif direction == "es-en":
        prompt = f"Translate the following Spanish text to English:\n\nSpanish: {source_text}\nEnglish:"
    elif direction == "en-es":
        prompt = f"Translate the following English text to Spanish:\n\nEnglish: {source_text}\nSpanish:"
    else:
        raise ValueError(f"Unknown direction: {direction}")
    return prompt


def generate_translation(
    model,
    tokenizer,
    source_text: str,
    direction: str = "es-en",
    prompt_template: Optional[str] = None,
    max_length: int = 512,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda"
) -> str:
    """Generate translation using the model with a prompt."""
    # Create prompt
    prompt = create_translation_prompt(source_text, direction, prompt_template)
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (remove input)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return translation.strip()


def generate_translations_batch(
    model,
    tokenizer,
    source_texts: List[str],
    direction: str = "es-en",
    prompt_template: Optional[str] = None,
    max_length: int = 512,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda"
) -> List[str]:
    """Generate translations for a batch of source texts."""
    # Create prompts for all texts
    prompts = [create_translation_prompt(text, direction, prompt_template) for text in source_texts]
    
    # Tokenize all inputs with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Get actual input lengths (excluding padding) for each sequence
    if inputs.get('attention_mask') is not None:
        input_lengths = inputs['attention_mask'].sum(dim=1).cpu().tolist()
    else:
        # If no attention mask, assume all sequences are the same length
        input_lengths = [inputs['input_ids'].shape[1]] * len(source_texts)
    
    # Generate translations for the batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated parts (remove input prompts)
    translations = []
    for i, output in enumerate(outputs):
        # Extract only the generated tokens (after the input prompt)
        input_len = input_lengths[i]
        generated_ids = output[input_len:]
        translation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        translations.append(translation.strip())
    
    return translations


def load_opus_globalvoices(
    split: str = "test",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Load OPUS Global Voices Spanish-English parallel corpus from HuggingFace.
    
    Args:
        split: Dataset split to use (train, validation, test)
        max_samples: Maximum number of samples to load
        cache_dir: Cache directory for HuggingFace datasets
    """
    print(f"Loading OPUS Global Voices from HuggingFace...")
    print(f"Note: Dataset only has 'train' split, will create {split} split manually")
    
    try:
        # Load dataset from HuggingFace using 'en-es' config for English-Spanish pairs
        # The dataset only has a 'train' split, so we'll load that and split it ourselves
        full_dataset = load_dataset(
            "sentence-transformers/parallel-sentences-global-voices",
            "en-es",  # Config name for English-Spanish pairs
            split="train",  # Only split available
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Create train/validation/test splits manually
        total_size = len(full_dataset)
        
        if split == "test":
            # Use last 10% for test
            start_idx = int(total_size * 0.9)
            end_idx = total_size
            dataset = full_dataset.select(range(start_idx, end_idx))
            print(f"Created test split: {len(dataset)} examples (from {start_idx} to {end_idx})")
        elif split == "validation":
            # Use 10-20% for validation
            start_idx = int(total_size * 0.1)
            end_idx = int(total_size * 0.2)
            dataset = full_dataset.select(range(start_idx, end_idx))
            print(f"Created validation split: {len(dataset)} examples (from {start_idx} to {end_idx})")
        elif split == "train":
            # Use first 90% for training
            start_idx = 0
            end_idx = int(total_size * 0.9)
            dataset = full_dataset.select(range(start_idx, end_idx))
            print(f"Created train split: {len(dataset)} examples (from {start_idx} to {end_idx})")
        else:
            # Use all data
            dataset = full_dataset
            print(f"Using full dataset: {len(dataset)} examples")
            
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        raise Exception(f"Could not load OPUS Global Voices from HuggingFace: {e}")
    
    # Extract Spanish and English sentences
    # The 'en-es' config has 'english' and 'non_english' fields
    # 'non_english' is Spanish for the 'en-es' config
    pairs = []
    for i in range(len(dataset)):
        example = dataset[i]
        # Field names are 'english' and 'non_english'
        english = example.get("english", "")
        spanish = example.get("non_english", "")
        
        # Clean and validate
        if spanish and english:
            spanish = str(spanish).strip()
            english = str(english).strip()
            if len(spanish) > 0 and len(english) > 0:
                # Store as (spanish, english) for consistency with our evaluation
                pairs.append((spanish, english))
        
        if max_samples and len(pairs) >= max_samples:
            break
    
    print(f"Loaded {len(pairs)} sentence pairs from HuggingFace")
    return pairs


def evaluate_translation(
    model,
    tokenizer,
    source_texts: List[str],
    target_texts: List[str],
    direction: str,
    prompt_template: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> dict:
    """Evaluate translation quality using BLEU scores."""
    print(f"\nEvaluating {direction} translation (batch_size={batch_size})...")
    
    if max_samples:
        source_texts = source_texts[:max_samples]
        target_texts = target_texts[:max_samples]
    
    predictions = []
    references = []
    
    # Generate translations in batches
    num_batches = (len(source_texts) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=f"Generating {direction} translations"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(source_texts))
        
        batch_sources = source_texts[start_idx:end_idx]
        batch_targets = target_texts[start_idx:end_idx]
        
        try:
            batch_predictions = generate_translations_batch(
                model, tokenizer,
                batch_sources,
                direction=direction,
                prompt_template=prompt_template,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device
            )
            predictions.extend(batch_predictions)
            references.extend(batch_targets)
        except Exception as e:
            print(f"Error translating batch {batch_idx} (examples {start_idx}-{end_idx-1}): {e}")
            # Fall back to individual generation for this batch
            for i, (source, reference) in enumerate(zip(batch_sources, batch_targets)):
                try:
                    prediction = generate_translation(
                        model, tokenizer, source, direction=direction,
                        prompt_template=prompt_template,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        device=device
                    )
                    predictions.append(prediction)
                    references.append(reference)
                except Exception as e2:
                    print(f"Error translating example {start_idx + i}: {e2}")
                    predictions.append("")
                    references.append(reference)
    
    # Compute BLEU scores
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])
    
    # Also compute sentence-level BLEU for statistics
    sentence_bleus = []
    for pred, ref in zip(predictions, references):
        try:
            sent_bleu = bleu.sentence_score(pred, [ref])
            sentence_bleus.append(sent_bleu.score)
        except:
            sentence_bleus.append(0.0)
    
    avg_sentence_bleu = sum(sentence_bleus) / len(sentence_bleus) if sentence_bleus else 0.0
    
    results = {
        "direction": direction,
        "num_examples": len(predictions),
        "corpus_bleu": bleu_score.score,
        "corpus_bleu_details": {
            "precisions": bleu_score.precisions,
            "bp": bleu_score.bp,
            "ratio": bleu_score.ratio,
            "hyp_len": bleu_score.hyp_len,
            "ref_len": bleu_score.ref_len,
        },
        "avg_sentence_bleu": avg_sentence_bleu,
    }
    
    return results, predictions, references


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    language: str,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> dict:
    """Compute perplexity on a list of texts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts to evaluate
        language: Language name (for logging)
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        device: Device to run on
        max_samples: Maximum number of samples to evaluate
    
    Returns:
        Dictionary with perplexity metrics
    """
    print(f"\nComputing perplexity on {language} texts...")
    
    if max_samples:
        texts = texts[:max_samples]
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_valid_samples = 0
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing {language} perplexity"):
        batch_texts = texts[i:i + batch_size]
        batch_losses = []
        batch_token_counts = []
        
        for text in batch_texts:
            try:
                # Tokenize text
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask", None)
                
                # Skip if empty
                if input_ids.shape[1] < 2:
                    continue
                
                # Get model outputs
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    # Compute loss (cross-entropy)
                    # Shift so that tokens < n predict n
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    
                    # Compute per-token loss
                    per_token_loss = loss_fct(flat_logits, flat_labels)
                    
                    # Mask out padding tokens if attention mask is provided
                    if attention_mask is not None:
                        shift_mask = attention_mask[..., 1:].contiguous().view(-1)
                        per_token_loss = per_token_loss * shift_mask
                        num_tokens = shift_mask.sum().item()
                    else:
                        num_tokens = flat_labels.numel()
                    
                    # Average loss (excluding padding)
                    if num_tokens > 0:
                        avg_loss = (per_token_loss.sum() / num_tokens).item()
                        batch_losses.append(avg_loss)
                        batch_token_counts.append(num_tokens)
                        num_valid_samples += 1
                        
            except Exception as e:
                print(f"Error computing perplexity for sample {i}: {e}")
                continue
        
        # Accumulate losses
        if batch_losses:
            for loss, num_tokens in zip(batch_losses, batch_token_counts):
                total_loss += loss * num_tokens
                total_tokens += num_tokens
    
    # Compute average loss and perplexity
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    
    results = {
        "language": language,
        "num_samples": num_valid_samples,
        "total_tokens": total_tokens,
        "average_loss": avg_loss,
        "perplexity": perplexity
    }
    
    print(f"{language} Perplexity: {perplexity:.4f} (avg loss: {avg_loss:.4f}, {num_valid_samples} samples, {total_tokens} tokens)")
    
    return results


def save_results(
    results: dict,
    predictions: List[str],
    references: List[str],
    output_dir: str,
    direction: str
):
    """Save evaluation results and translations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{direction}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save translations (predictions and references)
    translations_path = os.path.join(output_dir, f"{direction}_translations.txt")
    with open(translations_path, "w", encoding="utf-8") as f:
        for pred, ref in zip(predictions, references):
            f.write(f"PRED: {pred}\n")
            f.write(f"REF:  {ref}\n")
            f.write("\n")
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate translation quality on OPUS Global Voices corpus"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to use (train, validation, test)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./log/translation_eval",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--directions",
        type=str,
        default="both",
        choices=["both", "es-en", "en-es"],
        help="Translation directions to evaluate"
    )
    parser.add_argument(
        "--prompt_template_es_en",
        type=str,
        default=None,
        help="Custom prompt template for Spanish->English (use {source} placeholder)"
    )
    parser.add_argument(
        "--prompt_template_en_es",
        type=str,
        default=None,
        help="Custom prompt template for English->Spanish (use {source} placeholder)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--compute_perplexity",
        action="store_true",
        default=True,
        help="Compute perplexity on English and Spanish texts"
    )
    parser.add_argument(
        "--perplexity_batch_size",
        type=int,
        default=32,
        help="Batch size for perplexity computation (default: 32 for H100)"
    )
    parser.add_argument(
        "--translation_batch_size",
        type=int,
        default=16,
        help="Batch size for translation generation (default: 16 for H100)"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    pairs = load_opus_globalvoices(
        split=args.dataset_split,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir
    )
    
    if not pairs:
        raise ValueError("No sentence pairs loaded from dataset")
    
    spanish_texts = [pair[0] for pair in pairs]
    english_texts = [pair[1] for pair in pairs]
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
    
    all_results = {}
    
    # Compute perplexity on English and Spanish texts
    if args.compute_perplexity:
        print(f"\n{'='*60}")
        print(f"Computing Perplexity Metrics")
        print(f"{'='*60}")
        
        spanish_perplexity = compute_perplexity(
            model, tokenizer,
            spanish_texts,
            language="Spanish",
            batch_size=args.perplexity_batch_size,
            device=args.device,
            max_samples=args.max_samples
        )
        all_results["spanish_perplexity"] = spanish_perplexity
        
        english_perplexity = compute_perplexity(
            model, tokenizer,
            english_texts,
            language="English",
            batch_size=args.perplexity_batch_size,
            device=args.device,
            max_samples=args.max_samples
        )
        all_results["english_perplexity"] = english_perplexity
        
        # Save perplexity results
        perplexity_path = os.path.join(args.output_dir, "perplexity_metrics.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(perplexity_path, "w", encoding="utf-8") as f:
            json.dump({
                "spanish": spanish_perplexity,
                "english": english_perplexity
            }, f, indent=2, ensure_ascii=False)
        print(f"Perplexity results saved to {perplexity_path}")
        
        print(f"\n{'='*60}")
        print(f"Perplexity Summary:")
        print(f"{'='*60}")
        print(f"Spanish Perplexity: {spanish_perplexity['perplexity']:.4f}")
        print(f"English Perplexity: {english_perplexity['perplexity']:.4f}")
        print(f"{'='*60}\n")
    
    # Evaluate Spanish -> English
    if args.directions in ["both", "es-en"]:
        es_en_results, es_en_preds, es_en_refs = evaluate_translation(
            model, tokenizer,
            spanish_texts, english_texts,
            direction="es-en",
            prompt_template=args.prompt_template_es_en,
            max_samples=args.max_samples,
            batch_size=args.translation_batch_size,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        all_results["es-en"] = es_en_results
        save_results(
            es_en_results, es_en_preds, es_en_refs,
            args.output_dir, "es-en"
        )
        
        print(f"\n{'='*60}")
        print(f"Spanish -> English Results:")
        print(f"{'='*60}")
        print(f"Corpus BLEU: {es_en_results['corpus_bleu']:.4f}")
        print(f"BLEU-1: {es_en_results['corpus_bleu_details']['precisions'][0]:.4f}")
        print(f"BLEU-2: {es_en_results['corpus_bleu_details']['precisions'][1]:.4f}")
        print(f"BLEU-3: {es_en_results['corpus_bleu_details']['precisions'][2]:.4f}")
        print(f"BLEU-4: {es_en_results['corpus_bleu_details']['precisions'][3]:.4f}")
        print(f"Average Sentence BLEU: {es_en_results['avg_sentence_bleu']:.4f}")
        print(f"{'='*60}\n")
    
    # Evaluate English -> Spanish
    if args.directions in ["both", "en-es"]:
        en_es_results, en_es_preds, en_es_refs = evaluate_translation(
            model, tokenizer,
            english_texts, spanish_texts,
            direction="en-es",
            prompt_template=args.prompt_template_en_es,
            max_samples=args.max_samples,
            batch_size=args.translation_batch_size,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        all_results["en-es"] = en_es_results
        save_results(
            en_es_results, en_es_preds, en_es_refs,
            args.output_dir, "en-es"
        )
        
        print(f"\n{'='*60}")
        print(f"English -> Spanish Results:")
        print(f"{'='*60}")
        print(f"Corpus BLEU: {en_es_results['corpus_bleu']:.4f}")
        print(f"BLEU-1: {en_es_results['corpus_bleu_details']['precisions'][0]:.4f}")
        print(f"BLEU-2: {en_es_results['corpus_bleu_details']['precisions'][1]:.4f}")
        print(f"BLEU-3: {en_es_results['corpus_bleu_details']['precisions'][2]:.4f}")
        print(f"BLEU-4: {en_es_results['corpus_bleu_details']['precisions'][3]:.4f}")
        print(f"Average Sentence BLEU: {en_es_results['avg_sentence_bleu']:.4f}")
        print(f"{'='*60}\n")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

