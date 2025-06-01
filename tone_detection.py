import re
from collections import defaultdict
from transformers import pipeline

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

def read_conversation(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    speakers = {}
    
    pattern = r"^([^()]+)(?:\([^)]*\))?:\s*(.*)$"
    
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            speaker_name = match.group(1).strip()
            text = match.group(2).strip()
            
            if text:
                if speaker_name not in speakers:
                    speakers[speaker_name] = []
                speakers[speaker_name].append(text)
    
    return speakers

def analyze_tone(lines):
    label_scores = defaultdict(float)
    
    for line in lines:
        if line.strip():
            for item in classifier(line)[0]:
                label_scores[item['label']] += item['score']
    
    total_lines = max(1, len(lines))
    overall_results = {label: score / total_lines for label, score in label_scores.items()}
    
    return sorted(overall_results.items(), key=lambda x: x[1], reverse=True)[:5]

# def analyze_tone(lines):
#     label_scores = defaultdict(float)
    
#     for line in lines:
#         if line.strip():
#             for item in classifier(line)[0]:
#                 label_scores[item['label']] += item['score']
    
#     total_lines = max(1, len(lines))
#     overall_results = {label: score / total_lines for label, score in label_scores.items()}
    
#     # Return only the top 5 labels (no scores)
#     return [label for label, _ in sorted(overall_results.items(), key=lambda x: x[1], reverse=True)[:5]]

