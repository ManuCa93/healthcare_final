import pickle 

# === Load all subjects ===
def load_subject_data(DATA_PATH, subject_id):
    """Load a single subject's .pkl file."""
    filepath = DATA_PATH / f'S{subject_id}' / f'S{subject_id}.pkl'
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


