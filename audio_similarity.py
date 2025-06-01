import os
import torch
import librosa
from speechbrain.inference import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity

# Load the model once globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    encoder = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-resnet-voxceleb",
        run_opts={"device": device}
    )
except Exception as e:
    raise Exception("Failed to load any speaker recognition model.") from e


def generate_speaker_rttm(enrollment_folder: str, test_folder: str, output_rttm_path: str):
    # Generate embeddings for enrollment speakers
    enrollment_embeddings = {}
    for audio_file in os.listdir(enrollment_folder):
        if audio_file.endswith(('.wav', '.mp3')):
            speaker_name = os.path.splitext(audio_file)[0]
            audio_path = os.path.join(enrollment_folder, audio_file)

            try:
                signal, sr = librosa.load(audio_path, sr=16000)
                signal = torch.tensor(signal).unsqueeze(0).to(device)

                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    embedding = encoder.encode_batch(signal)
                enrollment_embeddings[speaker_name] = embedding.squeeze().cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"Error processing file {audio_path}: {str(e)}") from e

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_segments = len([f for f in os.listdir(test_folder) if f.endswith(('.wav', '.mp3'))])
    processed_segments = 0

    with open(output_rttm_path, 'w') as rttm_file:
        for audio_file in os.listdir(test_folder):
            if audio_file.endswith(('.wav', '.mp3')):
                audio_path = os.path.join(test_folder, audio_file)

                try:
                    signal, sr = librosa.load(audio_path, sr=16000)
                    signal = torch.tensor(signal).unsqueeze(0).to(device)

                    with torch.cuda.amp.autocast():
                        test_embedding = encoder.encode_batch(signal)
                    test_embedding = test_embedding.squeeze().cpu().numpy()

                    similarities = {
                        speaker: cosine_similarity(test_embedding.reshape(1, -1),
                                                   embedding.reshape(1, -1))[0][0]
                        for speaker, embedding in enrollment_embeddings.items()
                    }

                    recognized_speaker = max(similarities.items(), key=lambda x: x[1])[0]
                    duration = librosa.get_duration(y=signal.cpu().numpy().squeeze(), sr=sr)

                    try:
                        parts = os.path.splitext(audio_file)[0].split('_')
                        start_time = float(parts[2]) if len(parts) >= 4 else 0.0
                    except (ValueError, IndexError):
                        start_time = 0.0

                    rttm_entry = f"SPEAKER {audio_file} 1 {start_time:.3f} {duration:.3f} <NA> {recognized_speaker} <NA>\n"
                    rttm_file.write(rttm_entry)

                    processed_segments += 1
                    if torch.cuda.is_available() and processed_segments % 10 == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    raise RuntimeError(f"Error processing file {audio_path}: {str(e)}") from e

    return output_rttm_path
