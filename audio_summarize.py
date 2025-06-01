import torch
import gc
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_MODEL_NAME = "philschmid/bart-large-cnn-samsum"
OLLAMA_MODEL_NAME = "llama3.1"

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSFORMER_MODEL_NAME).to(DEVICE)


ollama_llm = OllamaLLM(model=OLLAMA_MODEL_NAME)

def clear_gpu_memory(verbose=False):
    """
    Clears GPU memory to prevent memory leaks and fragmentation.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if verbose:
            print("GPU memory cleared.")


# Summarization Pipeline

def summarize_conversation(file_path, batch_size=1000, overlap=200):
    batch_summary = generate_batch_summary(file_path, batch_size, overlap)
    full_summary = generate_full_summary(batch_summary)
    return batch_summary, full_summary

def generate_batch_summary(file_path, batch_size, overlap):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            full_text = file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

    preprocessed_text = preprocess_diarized_conversation(full_text)
    words = preprocessed_text.split()
    total_words = len(words)
    
    if total_words == 0:
        return "The file is empty or preprocessing failed."

    batch_summaries = []
    for i in tqdm(range(0, total_words, batch_size - overlap), desc="Summarizing batches"):
        end_idx = min(i + batch_size, total_words)
        batch_words = words[i:end_idx]
        if len(batch_words) < 50:
            continue

        batch_text = " ".join(batch_words)
        inputs = tokenizer(batch_text, max_length=1024, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            try:
                summary_ids = transformer_model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    min_length=150,
                    max_length=200,
                    early_stopping=True
                )
                batch_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                batch_summaries.append(batch_summary)
            except Exception as e:
                
                raise RuntimeError(f"Error during batch generation: {str(e)}")

    clear_gpu_memory(verbose=True)
    return batch_summaries[0] if len(batch_summaries) == 1 else " ".join(batch_summaries)

def generate_full_summary(batch_summary):
    try:
        prompt_template = PromptTemplate.from_template(
            "Generate a long summary of {summary}. Include all important information in paragraph form."
        )
        final_prompt = prompt_template.format(summary=batch_summary)
        final_summary = ollama_llm.invoke(final_prompt)
        return final_summary
    except Exception as e:
        return f"Error generating full summary: {str(e)}"

def preprocess_diarized_conversation(text):
    pattern = r'([^(]+)\s+\(([^)]+)\):\s+(.*?)(?=\n[^(]+\s+\([^)]+\):|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    formatted_text = ""
    for match in matches:
        speaker = match[0].strip()
        message = re.sub(r'\s+', ' ', match[2].strip())
        formatted_text += f"{speaker}: {message}\n\n"
    return formatted_text
