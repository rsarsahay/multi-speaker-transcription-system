from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import torchaudio
from typing import List, Dict
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def process_segment(segment, original_audio, sample_rate, output_folder, index):
    """Extracts and saves an audio segment in parallel."""
    try:
        start_time, end_time = segment['start'], segment['end']
        start_sample, end_sample = int(start_time * sample_rate), int(end_time * sample_rate)

        segment_audio = original_audio[:, start_sample:end_sample]
        filename = f"segment_{index:03d}_{start_time:.2f}_{end_time:.2f}.wav"
        output_path = os.path.join(output_folder, filename)
        torchaudio.save(output_path, segment_audio, sample_rate)

        return {
            'start_time': start_time,
            'end_time': end_time,
            'segment_path': output_path
        }
    except Exception as e:
        raise RuntimeError(f"Error processing segment {index}: {str(e)}")

def segment_audio(input_audio_path: str, output_folder: str) -> List[Dict]: 
    try:
        os.makedirs(output_folder, exist_ok=True)

        num_threads = max(2, multiprocessing.cpu_count() - 1)

        model = load_silero_vad()
        wav = read_audio(input_audio_path)
        original_audio, sample_rate = torchaudio.load(input_audio_path)
        speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)

        # Process segments in parallel using ThreadPoolExecutor
        segments_info = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(process_segment, segment, original_audio, sample_rate, output_folder, i)
                for i, segment in enumerate(speech_timestamps)
            ]
            for future in futures:
                result = future.result()
                if result:
                    segments_info.append(result)
        return segments_info

    except Exception as e:
        raise

