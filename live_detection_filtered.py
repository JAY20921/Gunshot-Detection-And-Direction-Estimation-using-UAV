"""
Enhanced Gunshot Detection with Propeller Noise Filtering
- High-pass filter to remove low-frequency propeller noise
- Adaptive noise profiling during calibration phase
- Spectral subtraction for noise removal
- Improved gunshot detection with spectral analysis
"""
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
CHUNK_DURATION = 0.1      # seconds per chunk (100ms)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
BUFFER_DURATION = 3.0     # seconds to analyze when gunshot detected
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)

# Noise filtering parameters
HIGHPASS_CUTOFF = 500     # Hz - propellers are typically below this
NOISE_PROFILE_DURATION = 3.0  # seconds to capture noise profile at startup
SPECTRAL_SUBTRACTION_FACTOR = 1.5  # How aggressively to subtract noise

# Gunshot detection thresholds (adjusted for filtered audio)
GUNSHOT_THRESHOLD_DB = -30   # dB level to trigger detection (higher after noise removal)
CREST_FACTOR_MIN = 6         # Minimum crest factor for gunshot
COOLDOWN_TIME = 2.0          # Seconds between detections

# Additional gunshot characteristics
RISE_TIME_MAX = 0.005        # Max 5ms rise time for gunshot
HIGH_FREQ_RATIO_MIN = 0.3    # Min ratio of energy above 2kHz
GUNSHOT_DURATION_MAX = 0.3   # Gunshots are typically < 300ms


# ============================================================
# NOISE FILTER CLASS
# ============================================================
class PropellerNoiseFilter:
    """Filter out propeller noise using multiple techniques."""
    
    def __init__(self, sample_rate=48000, highpass_cutoff=500):
        self.sample_rate = sample_rate
        self.highpass_cutoff = highpass_cutoff
        
        # Design high-pass Butterworth filter
        nyquist = sample_rate / 2
        normalized_cutoff = highpass_cutoff / nyquist
        self.b_hp, self.a_hp = signal.butter(4, normalized_cutoff, btype='high')
        
        # Noise profile (spectral)
        self.noise_profile = None
        self.noise_floor_db = -60
        self.is_calibrated = False
        
        # Filter states for continuous filtering
        self.filter_states = {}
    
    def calibrate(self, audio_samples):
        """
        Build noise profile from propeller-only audio.
        Call this during a calibration phase when only propellers are running.
        """
        # Compute average spectrum of noise
        fft_mag = np.abs(rfft(audio_samples))
        
        if self.noise_profile is None:
            self.noise_profile = fft_mag
        else:
            # Running average
            self.noise_profile = 0.9 * self.noise_profile + 0.1 * fft_mag
        
        # Update noise floor
        rms = np.sqrt(np.mean(audio_samples**2))
        self.noise_floor_db = 20 * np.log10(rms + 1e-12)
        self.is_calibrated = True
    
    def apply_highpass(self, audio, mic_id=1):
        """Apply high-pass filter to remove low-frequency noise."""
        if mic_id not in self.filter_states:
            self.filter_states[mic_id] = signal.lfilter_zi(self.b_hp, self.a_hp)
        
        filtered, self.filter_states[mic_id] = signal.lfilter(
            self.b_hp, self.a_hp, audio, zi=self.filter_states[mic_id]
        )
        return filtered
    
    def spectral_subtract(self, audio):
        """Remove noise using spectral subtraction."""
        if self.noise_profile is None:
            return audio
        
        # Compute FFT
        fft = rfft(audio)
        fft_mag = np.abs(fft)
        fft_phase = np.angle(fft)
        
        # Ensure noise profile matches size
        if len(self.noise_profile) != len(fft_mag):
            # Resize noise profile
            noise = np.interp(
                np.linspace(0, 1, len(fft_mag)),
                np.linspace(0, 1, len(self.noise_profile)),
                self.noise_profile
            )
        else:
            noise = self.noise_profile
        
        # Subtract noise (with floor to avoid negative values)
        cleaned_mag = np.maximum(
            fft_mag - SPECTRAL_SUBTRACTION_FACTOR * noise,
            0.01 * fft_mag  # Keep at least 1% to avoid artifacts
        )
        
        # Reconstruct signal
        cleaned_fft = cleaned_mag * np.exp(1j * fft_phase)
        cleaned_audio = np.fft.irfft(cleaned_fft, n=len(audio))
        
        return cleaned_audio.astype(np.float32)
    
    def filter(self, audio, mic_id=1):
        """Apply full noise filtering pipeline."""
        # Step 1: High-pass filter
        filtered = self.apply_highpass(audio, mic_id)
        
        # Step 2: Spectral subtraction (if calibrated)
        if self.is_calibrated:
            filtered = self.spectral_subtract(filtered)
        
        return filtered


# ============================================================
# IMPROVED GUNSHOT DETECTION
# ============================================================
def compute_high_freq_ratio(audio, fs, cutoff=2000):
    """Compute ratio of energy above cutoff frequency."""
    fft_mag = np.abs(rfft(audio))
    freqs = rfftfreq(len(audio), 1/fs)
    
    total_energy = np.sum(fft_mag**2)
    if total_energy < 1e-12:
        return 0
    
    high_freq_mask = freqs > cutoff
    high_freq_energy = np.sum(fft_mag[high_freq_mask]**2)
    
    return high_freq_energy / total_energy


def compute_rise_time(audio, fs):
    """Compute time from 10% to 90% of peak amplitude."""
    envelope = np.abs(audio)
    peak = np.max(envelope)
    
    if peak < 1e-12:
        return float('inf')
    
    threshold_10 = 0.1 * peak
    threshold_90 = 0.9 * peak
    
    idx_10 = None
    idx_90 = None
    
    for i, val in enumerate(envelope):
        if idx_10 is None and val > threshold_10:
            idx_10 = i
        if idx_10 is not None and val > threshold_90:
            idx_90 = i
            break
    
    if idx_10 is None or idx_90 is None:
        return float('inf')
    
    return (idx_90 - idx_10) / fs


def compute_duration(audio, fs, threshold_ratio=0.1):
    """Compute duration of the impulse above threshold."""
    envelope = np.abs(audio)
    peak = np.max(envelope)
    threshold = threshold_ratio * peak
    
    above_threshold = envelope > threshold
    if not np.any(above_threshold):
        return 0
    
    indices = np.where(above_threshold)[0]
    duration = (indices[-1] - indices[0]) / fs
    return duration


def is_gunshot_enhanced(audio, fs, noise_floor_db=-60):
    """
    Enhanced gunshot detection with multiple criteria.
    Returns: (is_gunshot, details_dict)
    """
    details = {}
    
    # Basic RMS and dB
    rms = np.sqrt(np.mean(audio**2))
    db = 20 * np.log10(rms + 1e-12)
    details['db'] = db
    
    # Must be significantly above noise floor
    db_above_noise = db - noise_floor_db
    details['db_above_noise'] = db_above_noise
    
    if db < GUNSHOT_THRESHOLD_DB:
        return False, details
    
    # Crest factor
    peak = np.max(np.abs(audio))
    crest_factor = peak / (rms + 1e-12)
    details['crest_factor'] = crest_factor
    
    if crest_factor < CREST_FACTOR_MIN:
        return False, details
    
    # Rise time (gunshots have very fast attack)
    rise_time = compute_rise_time(audio, fs)
    details['rise_time'] = rise_time
    
    if rise_time > RISE_TIME_MAX:
        return False, details
    
    # High frequency content (gunshots have broadband energy)
    high_freq_ratio = compute_high_freq_ratio(audio, fs)
    details['high_freq_ratio'] = high_freq_ratio
    
    if high_freq_ratio < HIGH_FREQ_RATIO_MIN:
        return False, details
    
    # Duration check
    duration = compute_duration(audio, fs)
    details['duration'] = duration
    
    if duration > GUNSHOT_DURATION_MAX:
        return False, details
    
    return True, details


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(model_path):
    """Load the trained direction detection model."""
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please run wav_to_csv.py first to train the model.")
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
# AUDIO FEATURE COMPUTATION
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
# MULTI-MICROPHONE RECORDER WITH FILTERING
# ============================================================
class MultiMicRecorderFiltered:
    """Record from 4 microphones with noise filtering."""

    def __init__(self, device_indices, sample_rate=48000, chunk_size=4800):
        self.device_indices = device_indices
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.streams = []
        self.buffers = {i: deque(maxlen=BUFFER_SIZE) for i in [1, 2, 3, 4]}
        self.filtered_buffers = {i: deque(maxlen=BUFFER_SIZE) for i in [1, 2, 3, 4]}
        self.running = False
        self.audio_queues = {i: queue.Queue() for i in [1, 2, 3, 4]}
        self.filtered_queues = {i: queue.Queue() for i in [1, 2, 3, 4]}
        
        # Noise filters for each microphone
        self.filters = {i: PropellerNoiseFilter(sample_rate, HIGHPASS_CUTOFF) 
                       for i in [1, 2, 3, 4]}
        self.is_calibrating = False

    def _audio_callback(self, mic_num):
        """Create callback for a specific microphone."""
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Mic{mic_num} status: {status}")
            
            # Get raw audio
            audio = indata[:, 0].copy()
            self.buffers[mic_num].extend(audio)
            self.audio_queues[mic_num].put(audio)
            
            # During calibration, update noise profile
            if self.is_calibrating:
                self.filters[mic_num].calibrate(audio)
            
            # Apply filtering
            filtered = self.filters[mic_num].filter(audio, mic_num)
            self.filtered_buffers[mic_num].extend(filtered)
            self.filtered_queues[mic_num].put(filtered)
            
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

    def calibrate_noise(self, duration=3.0):
        """
        Calibrate noise profile. 
        Run this when only propellers are making noise (no gunshots).
        """
        print(f"\n{'='*60}")
        print("NOISE CALIBRATION")
        print(f"{'='*60}")
        print(f"Recording propeller noise for {duration} seconds...")
        print("Make sure propellers are running but NO other sounds!")
        
        self.is_calibrating = True
        time.sleep(duration)
        self.is_calibrating = False
        
        # Report calibration results
        for mic_num, filt in self.filters.items():
            print(f"  Mic{mic_num}: Noise floor = {filt.noise_floor_db:.1f} dB")
        
        print("Calibration complete!\n")

    def stop(self):
        """Stop all microphone streams."""
        self.running = False
        for stream in self.streams:
            stream.stop()
            stream.close()
        self.streams = []
        print("Microphone streams stopped.")

    def get_filtered_buffers(self):
        """Get filtered audio buffers."""
        return {
            i: np.array(list(self.filtered_buffers[i]))
            for i in [1, 2, 3, 4]
        }

    def get_latest_filtered_chunk(self):
        """Get the latest filtered chunk from each microphone."""
        chunks = {}
        for i in [1, 2, 3, 4]:
            try:
                chunks[i] = self.filtered_queues[i].get_nowait()
            except queue.Empty:
                chunks[i] = None
        return chunks
    
    def get_noise_floor(self):
        """Get average noise floor across all mics."""
        floors = [self.filters[i].noise_floor_db for i in [1, 2, 3, 4]]
        return np.mean(floors)


# ============================================================
# DIRECTION VISUALIZER
# ============================================================
def print_direction_indicator(direction, confidence):
    """Print ASCII art direction indicator."""
    angle = int(direction.replace('°', ''))

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
    """Main loop for live gunshot detection with noise filtering."""

    print("\n" + "="*60)
    print("LIVE GUNSHOT DIRECTION DETECTION (FILTERED)")
    print("="*60)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"High-pass Cutoff: {HIGHPASS_CUTOFF} Hz")
    print(f"Detection Threshold: {GUNSHOT_THRESHOLD_DB} dB")
    print(f"Cooldown Time: {COOLDOWN_TIME} seconds")
    print("\nPress Ctrl+C to stop\n")

    # Initialize recorder with filtering
    recorder = MultiMicRecorderFiltered(
        mic_indices,
        sample_rate=SAMPLE_RATE,
        chunk_size=CHUNK_SIZE
    )

    if not recorder.start():
        print("Failed to start microphones!")
        return

    # CALIBRATION PHASE
    print("\n" + "="*60)
    print("STARTING PROPELLER NOISE CALIBRATION")
    print("="*60)
    input("Start your drone propellers, then press Enter to begin calibration...")
    recorder.calibrate_noise(NOISE_PROFILE_DURATION)

    last_detection_time = 0

    try:
        print("Listening for gunshots (with noise filtering)...\n")

        while True:
            time.sleep(CHUNK_DURATION)

            # Get filtered audio chunks
            chunks = recorder.get_latest_filtered_chunk()

            if any(c is None for c in chunks.values()):
                continue

            # Get current noise floor
            noise_floor = recorder.get_noise_floor()

            # Check for gunshot in any channel
            for mic_num, audio in chunks.items():
                is_shot, details = is_gunshot_enhanced(audio, SAMPLE_RATE, noise_floor)

                if is_shot:
                    current_time = time.time()

                    if current_time - last_detection_time < COOLDOWN_TIME:
                        continue

                    last_detection_time = current_time

                    print(f"[{time.strftime('%H:%M:%S')}] GUNSHOT DETECTED!")
                    print(f"  Trigger: mic{mic_num}")
                    print(f"  Level: {details['db']:.1f} dB (noise floor: {noise_floor:.1f} dB)")
                    print(f"  Crest factor: {details['crest_factor']:.1f}")
                    print(f"  Rise time: {details['rise_time']*1000:.2f} ms")
                    print(f"  High-freq ratio: {details['high_freq_ratio']:.1%}")

                    time.sleep(0.2)

                    # Get filtered buffers for direction analysis
                    buffers = recorder.get_filtered_buffers()

                    min_len = min(len(b) for b in buffers.values())
                    if min_len < SAMPLE_RATE:
                        print("  Not enough audio data, skipping...")
                        continue

                    audios = {i: buffers[i][-min_len:] for i in [1, 2, 3, 4]}

                    try:
                        direction, confidence, probs = predict_direction(
                            audios, SAMPLE_RATE, model_bundle
                        )

                        print_direction_indicator(direction, confidence)

                        print("All directions:")
                        for d, p in sorted(probs.items(), key=lambda x: -x[1]):
                            bar = "█" * int(p * 20)
                            print(f"  {d:>6}: {bar} {p:.1%}")
                        print()

                    except Exception as e:
                        print(f"  Prediction error: {e}")

                    break

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        recorder.stop()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("GUNSHOT DIRECTION DETECTION - NOISE FILTERED MODE")
    print("="*60)

    # Load model
    model_bundle = load_model(MODEL_PATH)

    # Check for command line arguments
    if len(sys.argv) == 5:
        try:
            mic_indices = [int(sys.argv[i]) for i in range(1, 5)]
            print(f"\nUsing microphones: {mic_indices}")
        except ValueError:
            print("Invalid arguments. Usage: python live_detection_filtered.py mic1 mic2 mic3 mic4")
            sys.exit(1)
    else:
        mic_indices = select_microphones()

    print(f"\nSelected microphones: {mic_indices}")
    print("  mic1 (Front-Right): Device {0}".format(mic_indices[0]))
    print("  mic2 (Back-Right):  Device {0}".format(mic_indices[1]))
    print("  mic3 (Back-Left):   Device {0}".format(mic_indices[2]))
    print("  mic4 (Front-Left):  Device {0}".format(mic_indices[3]))

    input("\nPress Enter to start live detection...")

    run_live_detection(mic_indices, model_bundle)
