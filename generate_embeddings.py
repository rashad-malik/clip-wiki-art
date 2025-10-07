import torch
import clip
from datasets import load_dataset
from tqdm import tqdm

print("Step 1/4: Loading the CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

print("\nStep 2/4: Loading the WikiArt dataset from Hugging Face...")
# This will download the dataset to your computer's cache
dataset = load_dataset("huggan/wikiart", split="train")

all_embeddings = []
batch_size = 256 # Adjust this based on your GPU/CPU memory

print("\nStep 3/4: Generating image embeddings in batches...")
with torch.no_grad():
    for i in tqdm(range(0, len(dataset), batch_size)):
        # Get a batch of images and preprocess them
        images = [dataset[j]['image'] for j in range(i, min(i + batch_size, len(dataset)))]
        # Filter out any potential None values if the dataset has corrupt images
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            continue
            
        preprocessed_images = torch.stack([preprocess(img) for img in valid_images]).to(device)

        # Compute embeddings
        batch_embeddings = model.encode_image(preprocessed_images)
        all_embeddings.append(batch_embeddings.cpu()) # Move to CPU to save GPU memory

# Concatenate all batch embeddings and save to a file
final_embeddings = torch.cat(all_embeddings)

print("\nStep 4/4: Saving embeddings to file...")
torch.save(final_embeddings, "wikiart_hf_embeddings.pt")

print("\n✅ Done! Your 'wikiart_hf_embeddings.pt' file is ready.")