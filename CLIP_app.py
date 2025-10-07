import streamlit as st
import torch
import clip
from datasets import load_dataset

# ------------------------------------------------------------------------
#                           Global Configuration
# ------------------------------------------------------------------------
# The new embeddings file you generated
EMBEDDINGS_FILE_PATH = "wikiart_hf_embeddings.pt"

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ------------------------------------------------------------------------
#                      Load Data (Done once and cached)
# ------------------------------------------------------------------------
# We use st.cache_data to load these large objects only once.
@st.cache_resource
def load_data():
    """Loads the dataset and embeddings from disk."""
    image_embeddings = torch.load(
        EMBEDDINGS_FILE_PATH, 
        map_location=torch.device(device)
    ).float()
    
    # Normalise embeddings for cosine similarity
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("huggan/wikiart", split="train")
    
    return image_embeddings, dataset

image_embeddings, dataset = load_data()

# ------------------------------------------------------------------------
#                   Initialise Streamlit State & Layout
# ------------------------------------------------------------------------
st.title("🎨 WikiArt Search Engine")
st.subheader("Search for art using natural language.")

if "top_indices" not in st.session_state:
    st.session_state.top_indices = None
if "similarity" not in st.session_state:
    st.session_state.similarity = None

# ------------------------------------------------------------------------
#                      Search Form
# ------------------------------------------------------------------------
with st.form(key="search_form"):
    query = st.text_input("Enter a description (e.g. 'a surreal dreamlike painting')")
    submitted = st.form_submit_button("Search")

# ------------------------------------------------------------------------
#                          Handle the Search
# ------------------------------------------------------------------------
if submitted and query:
    with torch.no_grad():
        tokens = clip.tokenize([query]).to(device)
        text_features = model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (text_features @ image_embeddings.T).squeeze(0)
        top_indices = similarity.argsort(descending=True)[:10]

    st.session_state.top_indices = top_indices
    st.session_state.similarity = similarity

# ------------------------------------------------------------------------
#                     Display the Search Results
# ------------------------------------------------------------------------
if st.session_state.top_indices is not None:
    st.subheader("Top 10 Results:")
    cols = st.columns(5)

    for i, idx_tensor in enumerate(st.session_state.top_indices):
        idx = idx_tensor.item() # Convert tensor to integer index
        
        # Retrieve the corresponding image and metadata
        img = dataset[idx]['image']
        artist = dataset[idx]['artist']
        style = dataset[idx]['style']
        
        similarity_score = st.session_state.similarity[idx].item()

        # Display in columns
        col = cols[i % 5]
        with col:
            st.image(img, use_container_width=True, caption=f"{artist} ({style})")
            st.write(f"Score: {similarity_score:.3f}")