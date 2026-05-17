import os
import re
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from scipy.fft import rfft, rfftfreq
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_DIR = r"D:\Dataset"
INPUT_CSV = r"c:\Users\acer\gunshot_features.csv"  
OUTPUT_CSV = "gunshot_direction_dataset.csv"
MODEL_PATH = "gunshot_direction_model.pkl"


# ============================================================
# STEP 1: COMPUTE TDOA BETWEEN MICROPHONE PAIRS
# ============================================================
def compute_tdoa(audio1, audio2, fs):
    """Compute time delay between two signals using cross-correlation."""
    # Normalize
    audio1 = audio1 / (np.max(np.abs(audio1)) + 1e-12)
    audio2 = audio2 / (np.max(np.abs(audio2)) + 1e-12)

    # Cross-correlation
    corr = signal.correlate(audio1, audio2, mode='full')
    lags = signal.correlation_lags(len(audio1), len(audio2), mode='full')

    # Find peak
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


def estimate_direction_from_tdoa(tdoa_12, tdoa_13, tdoa_14):
    """
    Estimate direction using TDOA values for X-configuration drone.

    Mic layout (looking down at drone):
        mic1 (FL)     mic2 (FR)
              \\     //
               \\   //
                \\ //
                 //\\
                //  \\
        mic4 (BL)     mic3 (BR)

    TDOA > 0 means sound reached mic1 BEFORE mic2 (source closer to mic1)
    """
    # For X-config: use diagonal pairs for better accuracy
    # tdoa_13: mic1 (front-left) vs mic3 (back-right) - main diagonal
    # tdoa_24: would be mic2 (front-right) vs mic4 (back-left) - other diagonal

    # Front-back component: average of front mics vs back mics
    # If tdoa_12 is small but tdoa_13 and tdoa_14 are large negative, sound is from front
    front_back = (tdoa_13 + tdoa_14) / 2  # Positive = from back, Negative = from front

    # Left-right component
    # mic1,mic4 are left; mic2,mic3 are right
    # tdoa_12: if positive, sound reached mic1 first (from left)
    # tdoa_34 would tell us about back left-right
    left_right = tdoa_12 - tdoa_14  # Combines front and back L-R info

    # Calculate angle using arctan2 (y=front-back, x=left-right)
    angle_rad = np.arctan2(-left_right, -front_back)
    angle_deg = np.degrees(angle_rad)

    # Normalize to 0-360 (0=front, 90=right, 180=back, 270=left)
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg


# ============================================================
# STEP 2: PROCESS EXISTING CSV AND ADD DIRECTION FEATURES
# ============================================================
def process_csv_for_direction(input_csv, output_csv, dataset_dir):
    """Process the existing CSV and compute TDOA-based direction features."""

    print("Loading existing CSV...")
    df = pd.read_csv(input_csv)

    # Files named by recording session (e.g., "0_10" from "0_10_mic1.wav")
    # Filename format: {angle}_{shot_number}_mic{mic_number}.wav
    # Example: 0_10_mic2.wav = angle 0°, shot #10, mic 2
    pattern = r'(\d+)_(\d+)_mic(\d+)\.wav'

    # Create mapping: session_id -> {mic_num: row_data, 'angle': angle}
    sessions = defaultdict(dict)

    for idx, row in df.iterrows():
        filename = row['file']
        match = re.match(pattern, filename)
        if match:
            angle = int(match.group(1))      # Direction angle (0, 90, 180, 270, etc.)
            shot_num = match.group(2)         # Shot number
            mic_num = int(match.group(3))     # Microphone number
            session_id = f"{angle}_{shot_num}"
            sessions[session_id][mic_num] = row
            sessions[session_id]['angle'] = angle  # Store the angle as label

    print(f"Found {len(sessions)} recording sessions with 4 mics each")

    # Process each session
    results = []

    for session_id, session_data in sessions.items():
        # Count only mic entries (1, 2, 3, 4), not 'angle' key
        mic_count = sum(1 for k in session_data.keys() if isinstance(k, int))
        if mic_count != 4:
            print(f"  Skipping {session_id}: only {mic_count} mics found")
            continue

        mics = session_data  # Contains both mic data and angle

        try:
            # Load all 4 audio files for TDOA calculation
            audios = {}
            fs = None

            # Extract angle and shot_num from session_id
            angle, shot_num = session_id.split('_')

            for mic_num in [1, 2, 3, 4]:
                filename = f"{angle}_{shot_num}_mic{mic_num}.wav"
                filepath = os.path.join(dataset_dir, filename)

                if os.path.exists(filepath):
                    audio, sample_rate = sf.read(filepath)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    audios[mic_num] = audio
                    fs = sample_rate

            if len(audios) != 4 or fs is None:
                continue

            # Compute TDOA between mic pairs
            tdoa_12 = compute_tdoa(audios[1], audios[2], fs)  # mic1 vs mic2
            tdoa_13 = compute_tdoa(audios[1], audios[3], fs)  # mic1 vs mic3
            tdoa_14 = compute_tdoa(audios[1], audios[4], fs)  # mic1 vs mic4
            tdoa_23 = compute_tdoa(audios[2], audios[3], fs)  # mic2 vs mic3
            tdoa_24 = compute_tdoa(audios[2], audios[4], fs)  # mic2 vs mic4
            tdoa_34 = compute_tdoa(audios[3], audios[4], fs)  # mic3 vs mic4

            # Compute onset times for each mic
            onset_1 = compute_onset_time(audios[1], fs)
            onset_2 = compute_onset_time(audios[2], fs)
            onset_3 = compute_onset_time(audios[3], fs)
            onset_4 = compute_onset_time(audios[4], fs)

            # Onset time differences (relative to mic1)
            onset_diff_12 = onset_1 - onset_2
            onset_diff_13 = onset_1 - onset_3
            onset_diff_14 = onset_1 - onset_4

            # Estimate direction
            estimated_direction = estimate_direction_from_tdoa(tdoa_12, tdoa_13, tdoa_14)

            # Intensity ratios (louder mic = closer to source)
            peak_1 = float(mics[1]['peak'])
            peak_2 = float(mics[2]['peak'])
            peak_3 = float(mics[3]['peak'])
            peak_4 = float(mics[4]['peak'])

            intensity_ratio_12 = peak_1 / (peak_2 + 1e-12)
            intensity_ratio_13 = peak_1 / (peak_3 + 1e-12)
            intensity_ratio_14 = peak_1 / (peak_4 + 1e-12)
            intensity_ratio_24 = peak_2 / (peak_4 + 1e-12)

            # Energy ratios
            energy_1 = float(mics[1]['energy'])
            energy_2 = float(mics[2]['energy'])
            energy_3 = float(mics[3]['energy'])
            energy_4 = float(mics[4]['energy'])
            total_energy = energy_1 + energy_2 + energy_3 + energy_4

            energy_ratio_1 = energy_1 / (total_energy + 1e-12)
            energy_ratio_2 = energy_2 / (total_energy + 1e-12)
            energy_ratio_3 = energy_3 / (total_energy + 1e-12)
            energy_ratio_4 = energy_4 / (total_energy + 1e-12)

            # Find which mic detected sound first (indicates direction)
            first_mic = np.argmin([onset_1, onset_2, onset_3, onset_4]) + 1
            loudest_mic = np.argmax([peak_1, peak_2, peak_3, peak_4]) + 1

            # Get label from filename (angle is already extracted)
            label = sessions[session_id].get('angle', 'unknown')

            results.append({
                'session_id': session_id,
                # TDOA features (most important for direction)
                'tdoa_12': tdoa_12,
                'tdoa_13': tdoa_13,
                'tdoa_14': tdoa_14,
                'tdoa_23': tdoa_23,
                'tdoa_24': tdoa_24,
                'tdoa_34': tdoa_34,
                # Onset differences
                'onset_diff_12': onset_diff_12,
                'onset_diff_13': onset_diff_13,
                'onset_diff_14': onset_diff_14,
                # Intensity ratios
                'intensity_ratio_12': intensity_ratio_12,
                'intensity_ratio_13': intensity_ratio_13,
                'intensity_ratio_14': intensity_ratio_14,
                'intensity_ratio_24': intensity_ratio_24,
                # Energy distribution
                'energy_ratio_1': energy_ratio_1,
                'energy_ratio_2': energy_ratio_2,
                'energy_ratio_3': energy_ratio_3,
                'energy_ratio_4': energy_ratio_4,
                # Categorical features
                'first_mic': first_mic,
                'loudest_mic': loudest_mic,
                # Estimated direction (can be used as feature or target)
                'estimated_direction_deg': estimated_direction,
                # Original features from mic1 (representative)
                'duration': float(mics[1]['duration']),
                'rms': float(mics[1]['rms']),
                'db': float(mics[1]['db']),
                'spectral_centroid': float(mics[1]['spectral_centroid']),
                'spectral_rolloff': float(mics[1]['spectral_rolloff']),
                # Label from filename (angle)
                'label': label
            })

            print(f"  Processed: {session_id} -> Est. direction: {estimated_direction:.1f}°")

        except Exception as e:
            print(f"  Error processing {session_id}: {e}")

    # Save to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"\nDirection dataset saved: {output_csv}")
    print(f"Total sessions processed: {len(results)}")

    return result_df


# ============================================================
# STEP 3: TRAIN ML MODEL FOR DIRECTION DETECTION
# ============================================================
def train_direction_model(df, model_path):
    """Train a model to predict gunshot direction."""

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import pickle

    print("\n" + "="*60)
    print("TRAINING DIRECTION DETECTION MODEL")
    print("="*60)

    # Labels are angles extracted from filenames (0,30,60, 90, 180, 270, etc.)
    # Convert to direction classes
    df['label'] = df['label'].astype(int)

    # Get unique angles and create direction mapping
    unique_angles = sorted(df['label'].unique())
    print(f"\nDetected angles in data: {unique_angles}")

    # Map angles to direction names (optional, for readability)
    angle_to_direction = {
        0: 'front',
        45: 'front_right',
        90: 'right',
        135: 'back_right',
        180: 'back',
        225: 'back_left',
        270: 'left',
        315: 'front_left'
    }

    # Use angle as label directly (works for any angles in your data)
    df['direction_class'] = df['label'].astype(str) + '°'

    # Feature columns
    feature_cols = [
        'tdoa_12', 'tdoa_13', 'tdoa_14', 'tdoa_23', 'tdoa_24', 'tdoa_34',
        'onset_diff_12', 'onset_diff_13', 'onset_diff_14',
        'intensity_ratio_12', 'intensity_ratio_13', 'intensity_ratio_14', 'intensity_ratio_24',
        'energy_ratio_1', 'energy_ratio_2', 'energy_ratio_3', 'energy_ratio_4',
        'rms', 'db', 'spectral_centroid', 'spectral_rolloff'
    ]

    X = df[feature_cols].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['direction_class'].astype(str))

    print(f"\nClasses: {le.classes_}")
    print(f"Samples per class: {np.bincount(y)}")

    # Handle NaN/Inf values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Split data
    # Use stratify only if we have enough samples per class
    min_samples_per_class = np.bincount(y).min()
    if min_samples_per_class >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("Warning: Some classes have very few samples, disabling stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    # Test accuracy
    y_pred = rf_model.predict(X_test_scaled)
    print(f"\nTest set accuracy: {rf_model.score(X_test_scaled, y_test):.3f}")

    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=le.classes_,
        cmap='Blues'
    )
    plt.show()
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10).to_string(index=False))

    # Save model
    model_bundle = {
        'model': rf_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_cols': feature_cols
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f)

    print(f"\nModel saved: {model_path}")

    return model_bundle


# ============================================================
# STEP 4: PREDICT DIRECTION FOR NEW AUDIO
# ============================================================
def predict_direction(session_files, model_path, dataset_dir):
    """Predict gunshot direction for new 4-mic recording."""
    import pickle

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)

    model = bundle['model']
    scaler = bundle['scaler']
    le = bundle['label_encoder']
    feature_cols = bundle['feature_cols']

    # Load audio files
    audios = {}
    fs = None
    peaks = {}
    energies = {}

    for mic_num in [1, 2, 3, 4]:
        filepath = session_files.get(mic_num)
        if filepath and os.path.exists(filepath):
            audio, sample_rate = sf.read(filepath)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            audios[mic_num] = audio
            fs = sample_rate
            peaks[mic_num] = np.max(np.abs(audio))
            energies[mic_num] = np.sum(audio**2)

    if len(audios) != 4:
        raise ValueError("Need all 4 microphone files")

    # Compute features
    tdoa_12 = compute_tdoa(audios[1], audios[2], fs)
    tdoa_13 = compute_tdoa(audios[1], audios[3], fs)
    tdoa_14 = compute_tdoa(audios[1], audios[4], fs)
    tdoa_23 = compute_tdoa(audios[2], audios[3], fs)
    tdoa_24 = compute_tdoa(audios[2], audios[4], fs)
    tdoa_34 = compute_tdoa(audios[3], audios[4], fs)

    onset_1 = compute_onset_time(audios[1], fs)
    onset_2 = compute_onset_time(audios[2], fs)
    onset_3 = compute_onset_time(audios[3], fs)
    onset_4 = compute_onset_time(audios[4], fs)

    total_energy = sum(energies.values())

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
        'rms': np.sqrt(np.mean(audios[1]**2)),
        'db': 20 * np.log10(np.sqrt(np.mean(audios[1]**2)) + 1e-12),
        'spectral_centroid': 2000,  
        'spectral_rolloff': 4000 
    }

    X = np.array([[features[col] for col in feature_cols]])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    direction = le.inverse_transform([prediction])[0]
    confidence = np.max(probabilities)

    return {
        'direction': direction,
        'confidence': confidence,
        'all_probabilities': dict(zip(le.classes_, probabilities))
    }


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("GUNSHOT DIRECTION DETECTION SYSTEM")
    print("="*60)

    # Step 1: Process CSV and compute direction features
    if os.path.exists(INPUT_CSV):
        direction_df = process_csv_for_direction(INPUT_CSV, OUTPUT_CSV, DATASET_DIR)

        # Step 2: Train the model
        if len(direction_df) > 10:
            model_bundle = train_direction_model(direction_df, MODEL_PATH)

            print("\n" + "="*60)
            print("SETUP COMPLETE!")
            print("="*60)
            print(f"\n1. Direction dataset: {OUTPUT_CSV}")
            print(f"2. Trained model: {MODEL_PATH}")
            print("\nTO USE FOR PREDICTION:")
            print("------------------------")
            print("session_files = {")
            print("    1: 'path/to/mic1.wav',")
            print("    2: 'path/to/mic2.wav',")
            print("    3: 'path/to/mic3.wav',")
            print("    4: 'path/to/mic4.wav'")
            print("}")
            print(f"result = predict_direction(session_files, '{MODEL_PATH}', '{DATASET_DIR}')")
            print("print(f'Direction: {result[\"direction\"]} (confidence: {result[\"confidence\"]:.1%})')")
        else:
            print(f"\nNot enough data to train model. Found {len(direction_df)} sessions.")
    else:
        print(f"\nInput CSV not found: {INPUT_CSV}")
        print("Please run feature extraction first or update the INPUT_CSV path.")