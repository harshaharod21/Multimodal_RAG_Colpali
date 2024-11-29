import os
from typing import List, Tuple
from pdf2image import convert_from_path
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from colpali_engine.utils.colpali_processing_utils import process_images

def index(files: List[str], processor, retrieval_model, device) -> Tuple[List[torch.Tensor], List[Image.Image]]:
    images = []
    document_embeddings = []

    # Convert PDF pages to images
    for file in files:
        print(f"Indexing now: {file}")
        images.extend(convert_from_path(file))

    # Create DataLoader for image batches
    dataloader = DataLoader(
        images,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )

    # Process each batch and obtain embeddings
    for batch in dataloader:
        with torch.no_grad():
            batch = {key: value.to(device) for key, value in batch.items()}
            embeddings = retrieval_model(**batch)
        document_embeddings.extend(list(torch.unbind(embeddings.to("cpu"))))

    return document_embeddings, images

def load_pdfs(data_folder: str) -> List[str]:
    return [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.lower().endswith('.pdf')]
