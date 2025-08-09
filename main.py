#! /usr/bin/env python
import re
import torch
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "google/gemma-2-2b"
# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_SAMPLES = 500
# Lowered batch size for better compatibility with smaller GPUs
BATCH_SIZE = 16 # Lower this if you get OOM errors
MAX_NEW_TOKENS = 100


def main():
    """Main function to run the entire pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configure 4-bit quantization to save memory
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(f"Loading quantized model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto", # Automatically maps model layers to available devices
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Generate Samples
    generated_texts = generate_in_batches(model, tokenizer, device)

    # 2. Free up cached memory before the next big step
    torch.cuda.empty_cache()

    # 3. Compute Embeddings
    embeddings = compute_embeddings_batched(model, tokenizer, generated_texts, device)
    
    # 4. Create and show the map
    plot_map(embeddings, generated_texts)


def generate_in_batches(model, tokenizer, device):
    """Generates samples in manageable batches to prevent OOM errors."""
    print(f"Generating {NUM_SAMPLES} samples in batches of {BATCH_SIZE}...")
    all_texts = []
    for i in tqdm(range(0, NUM_SAMPLES, BATCH_SIZE), desc="Generating"):
        # Determine the size of the current batch
        current_batch_size = min(BATCH_SIZE, NUM_SAMPLES - i)
        
        input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
        batch_input_ids = input_ids.repeat(current_batch_size, 1)
        attention_mask = torch.ones_like(batch_input_ids)

        outputs = model.generate(
            batch_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_texts.extend(texts)

    print("Sample generation complete!")
    return all_texts


def compute_embeddings_batched(model, tokenizer, texts, device):
    """Computes embeddings in batches to save memory and improve speed."""
    print("Computing embeddings in batches...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_NEW_TOKENS
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        all_embeddings.append(mean_embeddings)
        
    print("Embeddings computed successfully!")
    return np.concatenate(all_embeddings, axis=0)


def categorize_text(text):
    """Applies a simple heuristic to categorize text content."""
    if re.search(r'[^\x00-\x7F]+', text): return "Multilingual/Special Chars"
    if re.search(r'\b(def|import|class|for|while|if|else)\b|\{|\}|[\[\]]', text): return "Code"
    if re.search(r'[\+\-\*\/=]{2,}|\\frac|\\sum|\\int|\$\$.+\$\$', text): return "Math/Equations"
    return "Prose"


def plot_map(embeddings, texts):
    """Reduces embedding dimensionality with UMAP and plots the map."""
    print("Running UMAP for dimensionality reduction...")
    output_filename = "embedding_map.png"

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    categories = [categorize_text(text) for text in texts]
    unique_categories = sorted(list(set(categories)))
    cmap = plt.get_cmap("viridis", len(unique_categories))
    category_colors = {cat: cmap(i) for i, cat in enumerate(unique_categories)}
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 12))
    
    for category in unique_categories:
        mask = np.array([c == category for c in categories])
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            color=category_colors[category],
            label=category,
            alpha=0.7,
            s=15
        )
    
    plt.title(f"Embedded Generation Map of {MODEL_ID} Outputs", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title="Categories", fontsize=10)

    # plt.show()
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to {output_filename}")


if __name__ == "__main__":
    main()
