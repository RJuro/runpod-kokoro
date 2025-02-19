import os
import io
import json
import time
import wave
import torch
import base64
import random
import string
import subprocess
import numpy as np
from datetime import timedelta
from google.cloud import storage
from kokoro import KModel, KPipeline
import runpod
from pydub import AudioSegment

# Initialize CUDA and Kokoro model/pipelines
CUDA_AVAILABLE = torch.cuda.is_available()
device = 'cuda' if CUDA_AVAILABLE else 'cpu'
model = KModel().to(device).eval()

pipelines = {
    'a': KPipeline(lang_code='a', model=False),
    'b': KPipeline(lang_code='b', model=False)
}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

def generate_unique_filename(voice):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"tts_{voice}_{timestamp}_{unique_id}.mp3"

def synthesize_text_to_wav(text, voice, speed, wav_filename):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    audio_parts = []
    
    with torch.no_grad():
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[len(ps) - 1]
            audio = model(ps, ref_s, speed)
            audio_np = audio.cpu().numpy().astype(np.float32)
            audio_parts.append(audio_np)
    
    if not audio_parts:
        raise Exception("No audio generated.")
    
    combined_audio = np.concatenate(audio_parts)
    int_audio = (combined_audio * 32767).clip(-32768, 32767).astype(np.int16)
    
    with wave.open(wav_filename, 'wb') as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)         # 16-bit samples
        wav_file.setframerate(24000)     # Sample rate
        wav_file.writeframes(int_audio.tobytes())

def convert_wav_to_mp3(input_wav, output_mp3):
    command = [
        'ffmpeg',
        '-y',
        '-i', input_wav,
        '-codec:a', 'libmp3lame',
        '-qscale:a', '2',
        output_mp3
    ]
    subprocess.run(command, check=True)

def upload_blob(bucket_name, source_file, destination_blob):
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_json:
        raise Exception("Missing GOOGLE_APPLICATION_CREDENTIALS environment variable")
    credentials_info = json.loads(credentials_json)
    client = storage.Client.from_service_account_info(credentials_info)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_file)
    return blob

def generate_signed_url(blob, expiration_minutes=15):
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="GET"
    )

def handler(job):
    job_input = job.get('input', {})
    mode = job_input.get('mode', 'single')  # 'single' or 'podcast'
    try:
        speed = float(job_input.get('speed', 1.0))
    except Exception:
        speed = 1.0

    bucket_name = "research-digest-tts-out"
    
    if mode == 'podcast':
        segments = job_input.get('segments', [])
        if not segments:
            return {"error": "No segments provided for podcast mode.", "success": False}
        # Voice options can be overridden via API
        host_voice = job_input.get('host_voice', 'am_michael')
        expert_voice = job_input.get('expert_voice', 'af_bella')
        default_voice = job_input.get('default_voice', 'af_heart')
        silence_duration_ms = int(job_input.get('silence_duration_ms', 300))
        
        segment_wav_files = []
        try:
            for idx, seg in enumerate(segments):
                text = seg.get('text', '')
                if not text:
                    continue
                speaker = seg.get('speaker', 'host').lower()
                if speaker == 'host':
                    voice = host_voice
                elif speaker == 'expert':
                    voice = expert_voice
                else:
                    voice = seg.get('voice', default_voice)
                temp_wav = f"/tmp/tts_segment_{idx}_{int(time.time())}.wav"
                synthesize_text_to_wav(text, voice, speed, temp_wav)
                segment_wav_files.append(temp_wav)
            
            if not segment_wav_files:
                raise Exception("No valid segments synthesized.")
            
            combined_audio = None
            silence = AudioSegment.silent(duration=silence_duration_ms)
            for wav_file in segment_wav_files:
                segment_audio = AudioSegment.from_wav(wav_file)
                if combined_audio is None:
                    combined_audio = segment_audio
                else:
                    combined_audio += silence + segment_audio
            
            combined_wav = f"/tmp/tts_combined_{int(time.time())}.wav"
            combined_audio.export(combined_wav, format="wav")
            output_filename = generate_unique_filename("podcast")
            combined_mp3 = f"/tmp/tts_combined_{int(time.time())}.mp3"
            convert_wav_to_mp3(combined_wav, combined_mp3)
            blob = upload_blob(bucket_name, combined_mp3, output_filename)
            download_url = generate_signed_url(blob)
            
            for f in segment_wav_files:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(combined_wav):
                os.remove(combined_wav)
            if os.path.exists(combined_mp3):
                os.remove(combined_mp3)
            
            return {
                "download_url": download_url,
                "filename": output_filename,
                "success": True
            }
        except Exception as e:
            return {"error": f"Podcast TTS processing failed: {str(e)}", "success": False}
    
    else:
        text = job_input.get('text', '')
        if not text:
            return {"error": "No text provided.", "success": False}
        voice = job_input.get('voice', 'af_heart')
        try:
            temp_wav = f"/tmp/tts_temp_{int(time.time())}.wav"
            temp_mp3 = f"/tmp/tts_temp_{int(time.time())}.mp3"
            output_filename = generate_unique_filename(voice)
            
            synthesize_text_to_wav(text, voice, speed, temp_wav)
            convert_wav_to_mp3(temp_wav, temp_mp3)
            blob = upload_blob(bucket_name, temp_mp3, output_filename)
            download_url = generate_signed_url(blob)
            
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            
            return {
                "download_url": download_url,
                "filename": output_filename,
                "success": True
            }
        except Exception as e:
            return {"error": f"TTS processing failed: {str(e)}", "success": False}

runpod.serverless.start({"handler": handler})