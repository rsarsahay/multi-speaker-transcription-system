import re
import os
import gc
import torch
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from deepmultilingualpunctuation import PunctuationModel

@dataclass
class ConversationSegment:
    speaker: str
    start_time: str
    end_time: str
    text: str

# Global: Load punctuation model once
try:
    global_punctuation_model = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilingual-sonar-base")
except Exception as e:
    global_punctuation_model = None

def clear_gpu_memory(verbose=False):
    """
    Clears GPU memory to prevent memory leaks and fragmentation.
    """
    gc.collect()  # Collect unused Python objects
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if verbose:
            print("GPU memory cleared.")

class ConversationFormatter:
    def __init__(self, input_file_path: str, output_folder_path: str):
        self.input_file_path = input_file_path
        self.output_folder_path = output_folder_path
        self.line_pattern = re.compile(r'([\w\s-]+)\s*\((\d+:\d+)\s*-\s*(\d+:\d+)\):\s*(.+)')

        self.segments: List[ConversationSegment] = []
        self.metadata: Dict = {
            "processed_date": datetime.now().isoformat(),
            "total_segments": 0,
            "speakers": set(),
            "duration": "00:00"
        }

        self.punctuation_model = global_punctuation_model

    def correct_punctuation(self, text: str) -> str:
        if not self.punctuation_model:
            return text

        try:
            punctuated_text = self.punctuation_model.restore_punctuation(text.strip()).capitalize()
            return punctuated_text
        except Exception:
            return text

    def merge_consecutive_segments(self) -> List[ConversationSegment]:
        if not self.segments:
            return []

        merged = []
        current = self.segments[0]
        current_texts = [current.text]

        for next_segment in self.segments[1:]:
            if (next_segment.speaker == current.speaker and 
                self._is_time_consecutive(current.end_time, next_segment.start_time)):
                current_texts.append(next_segment.text)
                current.end_time = next_segment.end_time
            else:
                merged_text = ' '.join(current_texts)
                current.text = self.correct_punctuation(merged_text)
                merged.append(current)
                current = next_segment
                current_texts = [current.text]

        merged_text = ' '.join(current_texts)
        current.text = self.correct_punctuation(merged_text)
        merged.append(current)
        
        return merged

    def parse_line(self, line: str) -> Optional[ConversationSegment]:
        match = self.line_pattern.match(line.strip())
        if match:
            speaker, start_time, end_time, text = match.groups()
            return ConversationSegment(
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                text=text.strip()
            )
        return None

    def calculate_duration(self) -> str:
        if not self.segments:
            return "00:00"

        def time_to_minutes(time_str: str) -> int:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds

        last_segment = self.segments[-1]
        end_time = time_to_minutes(last_segment.end_time)
        start_time = time_to_minutes(self.segments[0].start_time)
        total_minutes = (end_time - start_time) // 60
        total_seconds = (end_time - start_time) % 60
        return f"{total_minutes:02d}:{total_seconds:02d}"

    def _is_time_consecutive(self, time1: str, time2: str, threshold_seconds: int = 2) -> bool:
        def time_to_seconds(time_str: str) -> int:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes * 60 + seconds

        t1 = time_to_seconds(time1)
        t2 = time_to_seconds(time2)
        return abs(t2 - t1) <= threshold_seconds

    def format_output(self, segment: ConversationSegment) -> str:
        return f"{segment.speaker} ({segment.start_time} - {segment.end_time}): {segment.text}"

    def save_metadata(self) -> None:
        metadata_path = os.path.join(self.output_folder_path, 'conversation_metadata.json')
        self.metadata["speakers"] = list(self.metadata["speakers"])
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

    def process(self) -> str:
        try:
            if not os.path.exists(self.input_file_path):
                raise FileNotFoundError(f"Input file not found: {self.input_file_path}")

            os.makedirs(self.output_folder_path, exist_ok=True)
            
            with open(self.input_file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if segment := self.parse_line(line):
                        self.segments.append(segment)
                        self.metadata["speakers"].add(segment.speaker)

            self.metadata["total_segments"] = len(self.segments)
            merged_segments = self.merge_consecutive_segments()
            self.metadata["duration"] = self.calculate_duration()
           
            output_text = "\n".join(self.format_output(segment) for segment in merged_segments)
            
            output_file_path = os.path.join(self.output_folder_path, 'diarized_conversation.txt')
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(output_text)

            self.save_metadata()
            return output_file_path

        except Exception as e:
            raise

def merge_conversations(input_file_path: str, output_folder_path: str) -> str:
    try:
        formatter = ConversationFormatter(input_file_path, output_folder_path)
        return formatter.process()
    except Exception as e:
        return f"Error processing file: {str(e)}"
