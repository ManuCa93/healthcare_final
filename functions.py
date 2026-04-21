import pickle 
import numpy as np

# === Load all subjects ===
def load_subject_data(DATA_PATH, subject_id):
    """Load a single subject's .pkl file."""
    filepath = DATA_PATH / f'S{subject_id}' / f'S{subject_id}.pkl'
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

# === Windowing function ===
def create_windows(subject_data, window_size_sec, LABEL_SAMPLING_RATE, WRIST_SAMPLING_RATES, VALID_LABELS):
    """Segment wrist signals into fixed-size time windows."""
    labels = subject_data['label'].flatten()
    wrist = subject_data['signal']['wrist']

    total_duration = len(labels) / LABEL_SAMPLING_RATE
    n_windows = int(total_duration // window_size_sec)

    windows = []
    window_labels = []

    for i in range(n_windows):
        # Majority vote for window label
        l_start = int(i * window_size_sec * LABEL_SAMPLING_RATE)
        l_end = int((i + 1) * window_size_sec * LABEL_SAMPLING_RATE)
        seg = labels[l_start:l_end]
        unique_l, counts_l = np.unique(seg, return_counts=True)
        majority = int(unique_l[np.argmax(counts_l)])

        if majority not in VALID_LABELS:
            continue
        if np.max(counts_l) / len(seg) < 0.8:
            continue

        # Extract signals at native rates
        window_data = {}
        valid = True
        for sig_name, sr in WRIST_SAMPLING_RATES.items():
            s_start = int(i * window_size_sec * sr)
            s_end = int((i + 1) * window_size_sec * sr)
            if s_end <= len(wrist[sig_name]):
                window_data[sig_name] = wrist[sig_name][s_start:s_end]
            else:
                valid = False
                break

        if valid:
            windows.append(window_data)
            window_labels.append(majority)

    return windows, np.array(window_labels)

