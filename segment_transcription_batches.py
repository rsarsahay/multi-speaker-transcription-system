import os
import torch
from typing import Dict, Optional, List
import pandas as pd
import faster_whisper
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load model once at the global level
device_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "compute_type": "float16" if torch.cuda.is_available() else "int8"
}
model = faster_whisper.WhisperModel(
    "medium.en",
    device=device_config["device"],
    compute_type=device_config["compute_type"],
)

def process_diarization(
    rttm_path: str,
    audio_segments_folder: str,
    output_file: str = 'local',
    batch_size: int = 4
) -> str:
    try:
        rttm_data = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    rttm_data.append({
                        'filename': parts[1],
                        'start_time': float(parts[3]),
                        'end_time': float(parts[3]) + float(parts[4]),
                        'speaker': parts[6]
                    })

        df = pd.DataFrame(rttm_data).sort_values('start_time')

        def transcribe_batch(batch: List[Dict]):
            batch_results = []
            for row in batch:
                segment_path = os.path.join(audio_segments_folder, row['filename'])
                if not os.path.exists(segment_path):
                    continue
                try:
                    segments, _ = model.transcribe(segment_path)
                    transcript = ' '.join(segment.text for segment in segments).strip()
                    start_time = format_timestamp(row['start_time'])
                    end_time = format_timestamp(row['end_time'])
                    formatted_line = f"{row['speaker']} ({start_time} - {end_time}): {transcript}"
                    batch_results.append(formatted_line)
                except Exception as e:
                    raise RuntimeError(f"Error processing file {segment_path}: {str(e)}")
            return batch_results

        batch_size = min(batch_size, len(df))  # Adjust if fewer items than batch_size
        batches = [df.iloc[i:i+batch_size].to_dict(orient='records') for i in range(0, len(df), batch_size)]

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(tqdm(executor.map(transcribe_batch, batches), total=len(batches), desc="Transcribing batches"))

        conversation = [line for batch in results for line in batch]
        full_conversation = "\n".join(conversation)

        os.makedirs(output_file, exist_ok=True)
        output_file_path = os.path.join(output_file, 'conversation.txt')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(full_conversation)

        return output_file_path

    except Exception as e:
        raise RuntimeError(f"An error occurred during transcription: {str(e)}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"
