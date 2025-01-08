# Multimodal_RAG_Colpali

# Knowledge Extraction from Complex PDFs Using COLPALI

This project demonstrates knowledge extraction from complex PDFs by leveraging the COLPALI model. It involves converting PDFs into images, indexing the data, and querying specific information using a retrieval-augmented framework.

## Features
- **PDF to Image Conversion**: Converts PDFs into image datasets for processing.
- **Retrieval Model**: Utilizes the COLPALI model with adapters for advanced query-based retrieval.
- **Generative AI Integration**: Uses Google Gemini for answering queries with contextual understanding.
- **End-to-End Pipeline**: Handles data ingestion, model loading, document indexing, and query resolution.

## Key Components
1. **`main.py`**: Orchestrates the entire pipeline, including data loading, model configuration, and query answering.
2. **`load_model.py`**: Loads the COLPALI model and adapter with Hugging Face integration.
3. **`retrieval.py`**: Manages document embedding retrieval and query answering with image-based prompts.

## Requirements
- Python with `torch`, `transformers`, `colpali_engine`, and other dependencies.
- Environment variables for API keys:
  - `HF_TOKEN`
  - `GENAI_API_KEY`

## Usage
1. Set up a `.env` file with the required API keys.
2. Run `main.py` to:
   - Load the PDF.
   - Index the data.
   - Query the model.
3. View the query results and the best-matching document image.

Note: In the article, I have also mentioned the implementation with byaldi which is simpler to implement but has complexities when integrated with vector database. Although the byaldi Library is in constant development, so they can release an updated version with improved capabilities.

Article: https://www.superteams.ai/blog/extracting-knowledge-from-complex-pdf-documents-enterprise
