from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import numpy

# load model and processor
processor = WhisperProcessor.from_pretrained("ZeeshanGeoPk/haitian-speech-to-text")
model = WhisperForConditionalGeneration.from_pretrained("ZeeshanGeoPk/haitian-speech-to-text")

# read audio files
sample_path = "bumn_x11.wav"
# load audio file using torchaudio
waveform, sample_rate = torchaudio.load(sample_path)

# resample if needed (Whisper model requires 16kHz)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)
    sample_rate = 16000

# ensure mono channel
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# process audio using Whisper processor
input_features = processor(waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
