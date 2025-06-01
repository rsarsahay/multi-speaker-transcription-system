import os
import torch
import logging
import shutil
import tempfile
import datetime
from audio_segmentation import segment_audio
from audio_similarity import generate_speaker_rttm
from segment_transcription_batches import process_diarization
from conversation_format import merge_conversations
from audio_summarize import summarize_conversation
from tone_detection import analyze_tone, read_conversation
import time 
import gc  

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
def setup_device():
    """Configure compute device."""
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "compute_type": "float16" if torch.cuda.is_available() else "int8"
    }

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

setup_logging()



def process_audio(audio_path):
    """Process audio and return transcription, summary, and tone analysis."""
    device_config = setup_device()
    base_output_dir = "local"

    os.makedirs(base_output_dir, exist_ok=True)

    audio_filename = os.path.basename(audio_path)
    filename_without_ext = os.path.splitext(audio_filename)[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_folder_name = f"{filename_without_ext}_{timestamp}"

    output_dir = os.path.join(base_output_dir, unique_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Processing audio in directory: {output_dir}")

    paths = {
        "segments": os.path.join(output_dir, "segments"),
        "rttm": os.path.join(output_dir, "speaker_labels.rttm"),
        "conversation": os.path.join(output_dir, "diarized_conversation.txt")
    }

    try:
        start_time = time.time()

        # Step 1: Segment Audio
        segment_audio(audio_path, paths["segments"])

        # Step 2: Speaker Diarization (RTTM)
        generate_speaker_rttm('voice_sample', paths["segments"], paths["rttm"])
        # Step 3: Transcription
        text_path = process_diarization(paths["rttm"], paths["segments"], output_dir)

        # Step 4: Format Conversation
        conversation_text_path = merge_conversations(text_path, output_dir)

        with open(conversation_text_path, "r", encoding="utf-8") as f:
            conversation_text = f.read()

        # Step 5: Summarization
        _, meta_summary = summarize_conversation(paths["conversation"])

        # Step 6: Tone Analysis
        speakers_data = read_conversation(paths["conversation"])
        tones = {speaker: analyze_tone(lines) for speaker, lines in speakers_data.items()}
        clear_gpu_memory(verbose=True)

        result = {
            "transcription": conversation_text,
            "summary": meta_summary,
            "tone": tones
        }

        end_time = time.time()
        print(f"TOTAL TIME: {end_time - start_time}")

        # Final GPU cleanup before directory deletion
        clear_gpu_memory(verbose=True)
        shutil.rmtree(base_output_dir, ignore_errors=True)

        logging.info(f"Successfully processed audio file: {audio_filename}")
        return result

    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        clear_gpu_memory(verbose=True)
        shutil.rmtree(base_output_dir, ignore_errors=True)
        return {"error": str(e)}
