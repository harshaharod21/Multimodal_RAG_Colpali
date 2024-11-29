import torch
from PIL import Image
from typing import List, Tuple
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
import google.generativeai as genai
 


def retrieve_top_document(query: str, processor, retrieval_model, device, document_embeddings: List[torch.Tensor], document_images: List[Image.Image]) -> Tuple[Image.Image, int]:
    placeholder_image = Image.new("RGB", (448, 448), (255, 255, 255))

    with torch.no_grad():
        query_batch = processor([query], images=[placeholder_image], return_tensors="pt").to(device)
        query_embeddings_tensor = retrieval_model(**query_batch)
        query_embeddings = list(torch.unbind(query_embeddings_tensor.to("cpu")))

    evaluator = CustomEvaluator(is_multi_vector=True)
    similarity_scores = evaluator.evaluate(query_embeddings, document_embeddings)
    best_index = int(similarity_scores.argmax(axis=1).item())

    return document_images[best_index], best_index

def configure_genai(api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name="gemini-1.5-flash")

def get_answer(model, prompt: str, image: Image):
    response = model.generate_content([prompt, image])
    return response.text

def answer_query(query: str, prompt: str, retrieval_model, processor, device, document_embeddings, images, genai_model):
    best_image, best_index = retrieve_top_document(query, processor, retrieval_model, device, document_embeddings, images)
    answer = get_answer(genai_model, prompt, best_image)
    return answer, best_image, best_index
