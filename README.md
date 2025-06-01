# multi-speaker-audio-intelligence-pipeline

## Overview

**multi-speaker-audio-intelligence-pipeline** is a modular, end-to-end system designed to transform raw multi-speaker audio recordings into structured, speaker-attributed transcripts, comprehensive conversation summaries, and actionable emotional tone analytics. The pipeline is optimized for scalability, leveraging GPU acceleration and batch processing to support high-throughput audio analytics for enterprise applications such as meeting intelligence, call center analytics, and conversational AI.

## Key Features

- **Automated Audio Segmentation:** Utilizes Silero VAD for robust voice activity detection, segmenting continuous audio streams into discrete speech segments.
- **Speaker Diarization:** Employs SpeechBrain’s ResNet-based speaker recognition to generate RTTM files and accurately attribute speech segments to individual speakers.
- **High-Performance Transcription:** Integrates Faster-Whisper for rapid, high-accuracy transcription, preserving speaker and temporal metadata.
- **Advanced Conversation Formatting:** Merges overlapping speaker turns and applies multilingual punctuation correction for enhanced readability and downstream NLP compatibility.
- **AI-Powered Summarization:** Leverages BART-large CNN and Llama3.1 (via Langchain) to generate concise and comprehensive conversation summaries, even for extended transcripts.
- **Emotion and Tone Analysis:** Applies RoBERTa Go-Emotions model to extract and rank dominant emotional tones per speaker, enabling sentiment-driven insights.
- **Optimized for Scale:** Implements batch processing with ThreadPoolExecutor and proactive GPU memory management for efficient operation on CUDA-enabled infrastructure.
- **Plug-and-Play Architecture:** Modular design enables seamless integration into existing analytics pipelines or deployment as a standalone service.

## Technology Stack

- **Languages:** Python
- **Core Libraries:** PyTorch, torchaudio, SpeechBrain, Hugging Face Transformers, Faster-Whisper, Langchain, Pandas
- **Audio Processing:** Librosa, Silero VAD, FFmpeg
- **NLP Models:** 
  - `philschmid/bart-large-cnn-samsum` (Summarization)
  - `SamLowe/roberta-base-go_emotions` (Emotion Classification)
  - `Llama3.1` via Langchain-Ollama (Long-form Summarization)
- **Utilities:** ThreadPoolExecutor, tqdm, regex, multiprocessing
- **Deployment:** Docker-ready structure, device configuration (float16/int8), GPU memory cleanup

## Pipeline Architecture

1. **Audio Segmentation:**  
   Raw audio is segmented using Silero VAD, producing speech-only segments for downstream processing.

2. **Speaker Diarization:**  
   Each segment is analyzed for speaker identity using SpeechBrain, generating RTTM files for accurate speaker labeling.

3. **Transcription:**  
   Segments are batch-transcribed with Faster-Whisper, maintaining speaker and timestamp metadata.

4. **Conversation Formatting:**  
   Transcripts are parsed, merged, and punctuated for clarity and structure, supporting multilingual content.

5. **Summarization:**  
   The formatted conversation is summarized using both extractive (BART) and abstractive (Llama3.1) LLMs for comprehensive meeting insights.

6. **Tone Detection:**  
   Speaker turns are analyzed for emotional tone using the Go-Emotions model, providing ranked sentiment profiles per participant.

## Use Cases

- Enterprise meeting summarization and analytics
- Call center conversation intelligence
- Customer feedback and sentiment analysis
- Voice-based conversational AI systems

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/multi-speaker-audio-intelligence-pipeline.git
   cd multi-speaker-audio-intelligence-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure CUDA for GPU acceleration.

## Usage

Place your audio files and enrollment samples in the appropriate directories, then invoke the main processing function as follows:

```python
from audio_function import process_audio

result = process_audio("path/to/audio_file.wav")
print(result["summary"])
```

## Output

- **Transcription:** Speaker-attributed, time-stamped conversation text
- **Summary:** AI-generated, context-rich summary of the conversation
- **Tone Analysis:** Top emotional tones per speaker

## License

Distributed under the MIT License. See `LICENSE` for details.

## Acknowledgements

- [Silero VAD](https://github.com/snakers4/silero-vad)
- [SpeechBrain](https://speechbrain.github.io/)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Langchain](https://github.com/langchain-ai/langchain)
