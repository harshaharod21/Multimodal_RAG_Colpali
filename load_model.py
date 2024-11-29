import torch
from transformers import AutoProcessor, BitsAndBytesConfig, ViTConfig
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from huggingface_hub import login

def load_retrieval_model(model_name: str, adapter_name: str, hf_token: str = None):
    

    if hf_token:
        login(hf_token)  # Log in to Hugging Face using token
    else:
        print("No Hugging Face token provided. Proceeding without authentication.")

    # Quantization configuration for BitsAndBytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load the retrieval model
    retrieval_model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        quantization_config=bnb_config,
    ).eval()

    # Load adapter and processor
    retrieval_model.load_adapter(adapter_name)
    processor = AutoProcessor.from_pretrained(adapter_name)

    return retrieval_model, processor
