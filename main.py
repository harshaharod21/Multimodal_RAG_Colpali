import torch
from data_ingestion_indexing import index, load_pdfs
from load_model import load_retrieval_model
from Retrieval import answer_query, configure_genai
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

# Retrieve the API key from environment variables
hf_token = os.getenv("HF_TOKEN")
api_key = os.getenv("GENAI_API_KEY")

print(hf_token)

DATA_FOLDER = "C:\All_projects\Multimodal_RAG\VF_FY2023_Environmental_Social_Responsibility_Report_FINAL_removed.pdf"
MODEL_NAME = "google/paligemma-3b-mix-448"
ADAPTER_NAME = "vidore/colpali"

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrieval_model, processor = load_retrieval_model(MODEL_NAME, ADAPTER_NAME,hf_token)

# Load and index PDFs
pdf_files = load_pdfs(DATA_FOLDER)
document_embeddings, images = index(pdf_files, processor, retrieval_model, device)

# Configure the GenAI model
if api_key:
    model = configure_genai(api_key)
else:
    raise ValueError("API key not found. Make sure it's set in the .env file.")

# Answer a query
search_query = "What is described in the purpose?"
prompt = "Explain this image in a 200-word summary."
answer, best_image, best_index = answer_query(search_query, prompt, retrieval_model, processor, device, document_embeddings, images, genai_model)
print(answer)
best_image.show()
