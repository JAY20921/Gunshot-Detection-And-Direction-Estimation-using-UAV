import os
import sys
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import pickle
from scipy import signal
from scipy.fft import rfft, rfftfreq
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = r"C:\Users\acer\gunshot_direction_model.pkl"
SAMPLE_RATE = 48000       # Hz 
CHUNK_DURATION = 0.1         # seconds per chunk (100ms)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_DURATION = 3.0        # seconds to analyze when gunshot detected
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# Gunshot detection thresholds
GUNSHOT_THRESHOLD_DB = -30   # dB level to trigger detection
CREST_FACTOR_MIN = 5         # Minimum crest factor for gunshot
COOLDOWN_TIME = 2.0          # Seconds between detections

# ============================================================
# LOAD MODEL
# ============================================================
def load_model(model_path):
    """Load the trained direction detection model."""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please run wav_to_csv.py.py first to train the model.")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)

    print(f"Model loaded: {model_path}")
    print(f"Classes: {bundle['label_encoder'].classes_}")
    return bundle


# ============================================================
# LIST AVAILABLE MICROPHONES
# ============================================================
def list_microphones():
    """List all available input devices."""
    print("\n" + "="*60)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("="*60)

    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append(i)
            print(f"  [{i}] {dev['name']}")
            print(f"      Channels: {dev['max_input_channels']}, "
                  f"Sample Rate: {dev['default_samplerate']}")

    return input_devices


def select_microphones():
    """Let user select 4 microphones for the drone array."""
    input_devices = list_microphones()

    if len(input_devices) < 4:
        print(f"\nWARNING: Only {len(input_devices)} input devices found.")
        print("You need 4 microphones for direction detection.")

    print("\n" + "="*60)
    print("SELECT 4 MICROPHONES (in order: mic1, mic2, mic3, mic4)")
    print("="*60)
    print("Refer to your drone setup:")
    print("  mic1 = Front-Right motor")
    print("  mic2 = Back-Right motor")
    print("  mic3 = Back-Left motor")
    print("  mic4 = Front-Left motor")

    selected = []
    for i in range(4):
        while True:
            try:
                idx = int(input(f"\nEnter device index for mic{i+1}: "))
                if idx in input_devices:
                    selected.append(idx)
                    break
                else:
                    print("Invalid device index. Try again.")
            except ValueError:
                print("Please enter a number.")

    return selected


# ============================================================
# AUDIO FEATURE COMPUTATION (same as training)
# ============================================================
def compute_tdoa(audio1, audio2, fs):
    """Compute time delay between two signals using cross-correlation."""
    audio1 = audio1 / (np.max(np.abs(audio1)) + 1e-12)
    audio2 = audio2 / (np.max(np.abs(audio2)) + 1e-12)

    corr = signal.correlate(audio1, audio2, mode='full')
    lags = signal.correlation_lags(len(audio1), len(audio2), mode='full')

    peak_idx = np.argmax(np.abs(corr))
    delay_samples = lags[peak_idx]
    delay_seconds = delay_samples / fs

    return delay_seconds


def compute_onset_time(audio, fs, threshold_ratio=0.1):
    """Find the time when the signal first exceeds threshold."""
    envelope = np.abs(audio)
    peak = np.max(envelope)
    threshold = threshold_ratio * peak

    for i, val in enumerate(envelope):
        if val > threshold:
            return i / fs
    return 0


def compute_spectral_centroid(audio, fs):
    """Compute spectral centroid."""
    fft_mag = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1/fs)
    if np.sum(fft_mag) < 1e-12:
        return 0
    return np.sum(freqs * fft_mag) / np.sum(fft_mag)


def compute_spectral_rolloff(audio, fs, percentile=0.85):
    """Compute spectral rolloff."""
    fft_mag = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1/fs)
    cumsum = np.cumsum(fft_mag)
    total = cumsum[-1]
    if total < 1e-12:
        return 0
    idx = np.searchsorted(cumsum, percentile * total)
    return freqs[min(idx, len(freqs)-1)]


def extract_features(audios, fs):
    """Extract all features needed for prediction."""
    # TDOA between mic pairs
    tdoa_12 = compute_tdoa(audios[1], audios[2], fs)
    tdoa_13 = compute_tdoa(audios[1], audios[3], fs)
    tdoa_14 = compute_tdoa(audios[1], audios[4], fs)
    tdoa_23 = compute_tdoa(audios[2], audios[3], fs)
    tdoa_24 = compute_tdoa(audios[2], audios[4], fs)
    tdoa_34 = compute_tdoa(audios[3], audios[4], fs)

    # Onset times
    onset_1 = compute_onset_time(audios[1], fs)
    onset_2 = compute_onset_time(audios[2], fs)
    onset_3 = compute_onset_time(audios[3], fs)
    onset_4 = compute_onset_time(audios[4], fs)

    # Peaks and energies
    peaks = {i: np.max(np.abs(audios[i])) for i in [1, 2, 3, 4]}
    energies = {i: np.sum(audios[i]**2) for i in [1, 2, 3, 4]}
    total_energy = sum(energies.values())

    # Spectral features from mic1
    spectral_centroid = compute_spectral_centroid(audios[1], fs)
    spectral_rolloff = compute_spectral_rolloff(audios[1], fs)
    rms = np.sqrt(np.mean(audios[1]**2))
    db = 20 * np.log10(rms + 1e-12)

    features = {
        'tdoa_12': tdoa_12, 'tdoa_13': tdoa_13, 'tdoa_14': tdoa_14,
        'tdoa_23': tdoa_23, 'tdoa_24': tdoa_24, 'tdoa_34': tdoa_34,
        'onset_diff_12': onset_1 - onset_2,
        'onset_diff_13': onset_1 - onset_3,
        'onset_diff_14': onset_1 - onset_4,
        'intensity_ratio_12': peaks[1] / (peaks[2] + 1e-12),
        'intensity_ratio_13': peaks[1] / (peaks[3] + 1e-12),
        'intensity_ratio_14': peaks[1] / (peaks[4] + 1e-12),
        'intensity_ratio_24': peaks[2] / (peaks[4] + 1e-12),
        'energy_ratio_1': energies[1] / (total_energy + 1e-12),
        'energy_ratio_2': energies[2] / (total_energy + 1e-12),
        'energy_ratio_3': energies[3] / (total_energy + 1e-12),
        'energy_ratio_4': energies[4] / (total_energy + 1e-12),
        'rms': rms,
        'db': db,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff
    }

    return features


# ============================================================
# GUNSHOT DETECTION
# ============================================================
def is_gunshot(audio, fs):
    """Check if audio contains a gunshot-like impulsive sound."""
    rms = np.sqrt(np.mean(audio**2))
    db = 20 * np.log10(rms + 1e-12)

    if db < GUNSHOT_THRESHOLD_DB:
        return False, db, 0

    # Check crest factor (peak/RMS ratio)
    peak = np.max(np.abs(audio))
    crest_factor = peak / (rms + 1e-12)

    if crest_factor < CREST_FACTOR_MIN:
        return False, db, crest_factor

    return True, db, crest_factor


# ============================================================
# PREDICT DIRECTION
# ============================================================
def predict_direction(audios, fs, model_bundle):
    """Predict gunshot direction from 4-channel audio."""
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    le = model_bundle['label_encoder']
    feature_cols = model_bundle['feature_cols']

    # Extract features
    features = extract_features(audios, fs)

    # Create feature vector in correct order
    X = np.array([[features[col] for col in feature_cols]])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    direction = le.inverse_transform([prediction])[0]
    confidence = np.max(probabilities)

    return direction, confidence, dict(zip(le.classes_, probabilities))


# ============================================================
# MULTI-MICROPHONE RECORDER
# ============================================================
class MultiMicRecorder:
    """Record from 4 microphones simultaneously."""

    def __init__(self, device_indices, sample_rate=48000, chunk_size=48000):
        self.device_indices = device_indices
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.streams = []
        self.buffers = {i: deque(maxlen=BUFFER_SIZE) for i in [1, 2, 3, 4]}
        self.running = False
        self.audio_queues = {i: queue.Queue() for i in [1, 2, 3, 4]}

    def _audio_callback(self, mic_num):
        """Create callback for a specific microphone."""
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Mic{mic_num} status: {status}")
            # Add audio to buffer
            audio = indata[:, 0].copy()
            self.buffers[mic_num].extend(audio)
            self.audio_queues[mic_num].put(audio)
        return callback

    def start(self):
        """Start recording from all microphones."""
        print("\nStarting microphone streams...")

        for i, dev_idx in enumerate(self.device_indices, 1):
            try:
                stream = sd.InputStream(
                    device=dev_idx,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    callback=self._audio_callback(i)
                )
                stream.start()
                self.streams.append(stream)
                print(f"  Mic{i}: Started (device {dev_idx})")
            except Exception as e:
                print(f"  Mic{i}: FAILED - {e}")
                self.stop()
                return False

        self.running = True
        return True

    def stop(self):
        """Stop all microphone streams."""
        self.running = False
        for stream in self.streams:
            stream.stop()
            stream.close()
        self.streams = []
        print("Microphone streams stopped.")

    def get_buffers(self):
        """Get current audio buffers as numpy arrays."""
        return {
            i: np.array(list(self.buffers[i]))
            for i in [1, 2, 3, 4]
        }

    def get_latest_chunk(self):
        """Get the latest chunk from each microphone."""
        chunks = {}
        for i in [1, 2, 3, 4]:
            try:
                chunks[i] = self.audio_queues[i].get_nowait()
            except queue.Empty:
                chunks[i] = None
        return chunks


# ============================================================
# DIRECTION VISUALIZER
# ============================================================
def print_direction_indicator(direction, confidence):
    """Print ASCII art direction indicator."""
    angle = int(direction.replace('°', ''))

    # Direction arrows
    arrows = {
        0: "    ↑\n    |\n    *",
        45: "      ↗\n     /\n    *",
        90: "    * ── →",
        135: "    *\n     \\\n      ↘",
        180: "    *\n    |\n    ↓",
        225: "      *\n     /\n    ↙",
        270: "← ── *",
        315: "    ↖\n     \\\n      *"
    }

    # Find closest arrow
    closest = min(arrows.keys(), key=lambda x: min(abs(x-angle), 360-abs(x-angle)))

    print("\n" + "="*40)
    print(f"GUNSHOT DETECTED!")
    print("="*40)
    print(f"\nDirection: {direction}")
    print(f"Confidence: {confidence:.1%}")
    print(f"\n{arrows.get(closest, '???')}")
    print("="*40 + "\n")


# ============================================================
# MAIN LIVE DETECTION LOOP
# ============================================================
def run_live_detection(mic_indices, model_bundle):
    """Main loop for live gunshot detection."""

    print("\n" + "="*60)
    print("LIVE GUNSHOT DIRECTION DETECTION")
    print("="*60)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Detection Threshold: {GUNSHOT_THRESHOLD_DB} dB")
    print(f"Cooldown Time: {COOLDOWN_TIME} seconds")
    print("\nPress Ctrl+C to stop\n")

    # Initialize recorder
    recorder = MultiMicRecorder(
        mic_indices,
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK_SIZE
    )

    if not recorder.start():
        print("Failed to start microphones!")
        return

    last_detection_time = 0

    try:
        print("Listening for gunshots...\n")

        while True:
            time.sleep(CHUNK_DURATION)

            # Get latest audio chunks
            chunks = recorder.get_latest_chunk()

            # Check if we have data from all mics
            if any(c is None for c in chunks.values()):
                continue

            # Check for gunshot in any channel
            for mic_num, audio in chunks.items():
                is_shot, db, crest = is_gunshot(audio, SAMPLE_RATE)

                if is_shot:
                    current_time = time.time()

                    # Check cooldown
                    if current_time - last_detection_time < COOLDOWN_TIME:
                        continue

                    last_detection_time = current_time

                    print(f"[{time.strftime('%H:%M:%S')}] Potential gunshot detected!")
                    print(f"  Trigger: mic{mic_num}, dB={db:.1f}, crest={crest:.1f}")

                    # Wait a moment to capture full event
                    time.sleep(0.2)

                    # Get full buffers for analysis
                    buffers = recorder.get_buffers()

                    # Ensure we have enough data
                    min_len = min(len(b) for b in buffers.values())
                    if min_len < SAMPLE_RATE:  # Need at least 1 second
                        print("  Not enough audio data, skipping...")
                        continue

                    # Trim to same length
                    audios = {i: buffers[i][-min_len:] for i in [1, 2, 3, 4]}

                    # Predict direction
                    try:
                        direction, confidence, probs = predict_direction(
                            audios, SAMPLE_RATE, model_bundle
                        )

                        print_direction_indicator(direction, confidence)

                        # Show all probabilities
                        print("All directions:")
                        for d, p in sorted(probs.items(), key=lambda x: -x[1]):
                            bar = "█" * int(p * 20)
                            print(f"  {d:>6}: {bar} {p:.1%}")
                        print()

                    except Exception as e:
                        print(f"  Prediction error: {e}")

                    break  # Only detect once per cycle

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        recorder.stop()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("GUNSHOT DIRECTION DETECTION - LIVE MODE")
    print("="*60)

    # Load model
    model_bundle = load_model(MODEL_PATH)

    # Check for command line arguments
    if len(sys.argv) == 5:
        # Mic indices provided as arguments
        try:
            mic_indices = [int(sys.argv[i]) for i in range(1, 5)]
            print(f"\nUsing microphones: {mic_indices}")
        except ValueError:
            print("Invalid arguments. Usage: python live_detection.py mic1_idx mic2_idx mic3_idx mic4_idx")
            sys.exit(1)
    else:
        # Interactive selection
        mic_indices = select_microphones()

    print(f"\nSelected microphones: {mic_indices}")
    print("  mic1 (Front-Right): Device {0}".format(mic_indices[0]))
    print("  mic2 (Back-Right):  Device {0}".format(mic_indices[1]))
    print("  mic3 (Back-Left):   Device {0}".format(mic_indices[2]))
    print("  mic4 (Front-Left):  Device {0}".format(mic_indices[3]))

    input("\nPress Enter to start live detection...")

    # Run live detection
    run_live_detection(mic_indices, model_bundle)
