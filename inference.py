import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

def moving_average_smoothing(array, kernel_size=5):
    if kernel_size < 2:
        return array
    pad_size = kernel_size // 2
    padded = np.pad(array, (pad_size, pad_size), mode='edge')
    smoothed = np.convolve(padded, np.ones(kernel_size)/kernel_size, mode='valid')
    return smoothed

def compute_frac_curve(H_d, H_u, epsilon=1e-8):
    numerator = np.abs(H_d * np.conjugate(H_u)) ** 2
    denominator = (np.abs(H_d)**2) * (np.abs(H_u)**2) + epsilon
    return numerator / denominator

def select_frequency_bands(freq, mag_db, phase_deg, band_ranges):
    mask = np.zeros_like(freq, dtype=bool)
    for (low, high) in band_ranges:
        mask |= (freq >= low) & (freq <= high)
    return freq[mask], mag_db[mask], phase_deg[mask]

def process_file(file_path,
                 downsample_factor=6,
                 unwrap_phase=True,
                 smoothing=False,
                 smoothing_kernel=5,
                 band_ranges=None):
    try:
        df = pd.read_csv(file_path, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(file_path, sep=r"\s+", engine="python")
    freq = df["Frequency"].values.astype(np.float32)
    mag_db = df["Magnitude"].values.astype(np.float32)
    phase_deg = df["Phase"].values.astype(np.float32)

    if unwrap_phase:
        phase_rad = np.deg2rad(phase_deg)
        phase_rad_unwrapped = np.unwrap(phase_rad)
        phase_deg = np.rad2deg(phase_rad_unwrapped)

    if downsample_factor > 1:
        freq = freq[::downsample_factor]
        mag_db = mag_db[::downsample_factor]
        phase_deg = phase_deg[::downsample_factor]

    if band_ranges is not None:
        freq, mag_db, phase_deg = select_frequency_bands(freq, mag_db, phase_deg, band_ranges)

    if smoothing and len(freq) >= smoothing_kernel:
        mag_db = moving_average_smoothing(mag_db, kernel_size=smoothing_kernel)
        phase_deg = moving_average_smoothing(phase_deg, kernel_size=smoothing_kernel)

    return freq, mag_db, phase_deg

class FRFDataset(Dataset):
    def __init__(self,
                 root_dir,
                 condition,
                 healthy_dir=None,
                 freq_max=2000.0,
                 downsample_factor=6,
                 smoothing=False,
                 smoothing_kernel=5,
                 unwrap_phase=True,
                 mag_mean=0.0, mag_std=1.0,
                 phase_mean=0.0, phase_std=1.0,
                 band_ranges=None):
        super().__init__()
        self.root_dir = root_dir
        self.condition = condition
        self.freq_max = freq_max
        self.downsample_factor = downsample_factor
        self.smoothing = smoothing
        self.smoothing_kernel = smoothing_kernel
        self.unwrap_phase = unwrap_phase
        self.band_ranges = band_ranges

        self.mag_mean = mag_mean
        self.mag_std = mag_std
        self.phase_mean = phase_mean
        self.phase_std = phase_std

        self.label_map = {
            "Beam_healthy": 0,
            "Beam_damage_2.96": 1,
            "Beam_damage_5.92": 2,
            "Beam_damage_8.87": 3
        }
        self.label = self.label_map[condition]

        self.folder = os.path.join(root_dir, condition)
        self.files = sorted(glob.glob(os.path.join(self.folder, "P-*.txt")))

        if condition != "Beam_healthy":
            if healthy_dir is None:
                self.healthy_dir = os.path.join(root_dir, "Beam_healthy")
            else:
                self.healthy_dir = healthy_dir
            self.healthy_files = sorted(glob.glob(os.path.join(self.healthy_dir, "P-*.txt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        freq, mag_db, phase_deg = process_file(file_path,
                                                downsample_factor=self.downsample_factor,
                                                unwrap_phase=self.unwrap_phase,
                                                smoothing=self.smoothing,
                                                smoothing_kernel=self.smoothing_kernel,
                                                band_ranges=self.band_ranges)

        freq_scaled = freq / self.freq_max
        mag_norm = (mag_db - self.mag_mean) / (self.mag_std + 1e-8)
        phase_norm = (phase_deg - self.phase_mean) / (self.phase_std + 1e-8)

        tokens = np.stack([freq_scaled, mag_norm, phase_norm], axis=-1)

        # Compute FRAC
        H_d = 10 ** (mag_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))
        if self.condition == "Beam_healthy":
            H_u = H_d
        else:
            healthy_file = self.healthy_files[idx]
            freq_h, mag_h, phase_h = process_file(
                healthy_file,
                downsample_factor=self.downsample_factor,
                unwrap_phase=self.unwrap_phase,
                smoothing=self.smoothing,
                smoothing_kernel=self.smoothing_kernel,
                band_ranges=self.band_ranges
            )
            H_u = 10 ** (mag_h / 20.0) * np.exp(1j * np.deg2rad(phase_h))

        frac_curve = compute_frac_curve(H_d, H_u)
        frac_channel = frac_curve[:, np.newaxis]
        tokens = np.concatenate([tokens, frac_channel], axis=-1)

        tokens = torch.tensor(tokens, dtype=torch.float32)
        label = torch.tensor(self.label, dtype=torch.long)
        return tokens, label

class HybridFRFTransformer(nn.Module):
    def __init__(self,
                 input_dim=4,
                 cnn_channels=32,
                 cnn_kernel=3,
                 cnn_stride=1,
                 d_model=64,
                 nhead=4,
                 num_layers=1,
                 num_classes=4,
                 dropout=0.1,
                 max_seq_len=600):
        super(HybridFRFTransformer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim,
                               out_channels=cnn_channels,
                               kernel_size=cnn_kernel,
                               stride=cnn_stride,
                               padding=cnn_kernel//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.input_proj = nn.Linear(cnn_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=d_model*4,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  
        x = self.input_proj(x)  
        seq_length = x.shape[1]
        pos_emb = self.pos_embedding[:seq_length, :].unsqueeze(0)
        x = x + pos_emb
        x = self.encoder(x)  
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print("Test Accuracy: {:.4f}".format(accuracy))

    print("Classification Report:")
    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    return all_labels, all_preds, cm



def plot_confusion_matrix_counts(cm, class_names, save_folder="training_graph"):
    plt.figure(figsize=(6,5))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Raw Counts)")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"confusion_matrix_counts_{timestamp}.png"
    plt.savefig(os.path.join(save_folder, filename), bbox_inches="tight")
    plt.close()
    print("Raw counts confusion matrix saved as", filename)

def plot_confusion_matrix_percent_total(cm, class_names, save_folder="training_graph"):
    total = cm.sum()
    cm_perc = (cm.astype(float) / total) * 100

    plt.figure(figsize=(6,5))
    ax = sns.heatmap(cm_perc, annot=True, fmt=".2f", cmap="Blues",
                     xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (% of Total Samples)")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

    plt.tight_layout()
    os.makedirs(save_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"confusion_matrix_totalperc_{timestamp}.png"
    plt.savefig(os.path.join(save_folder, filename), bbox_inches="tight")
    plt.close()
    print("Percentage-of-total confusion matrix saved as", filename)

def inference_main():
    root_dir = "./Data"  
    freq_max = 600.0
    downsample_factor = 1
    smoothing = False
    smoothing_kernel = 5
    unwrap_phase = False
    band_ranges = [(10, 40), (120, 160), (350, 450)] 

    mag_mean, mag_std = 0.0, 1.0
    phase_mean, phase_std = 0.0, 1.0

    conditions = ["Beam_healthy", "Beam_damage_2.96", "Beam_damage_5.92", "Beam_damage_8.87"]
    datasets = []
    labels_list = []
    for cond in conditions:
        ds = FRFDataset(root_dir=root_dir,
                        condition=cond,
                        freq_max=freq_max,
                        downsample_factor=downsample_factor,
                        smoothing=smoothing,
                        smoothing_kernel=smoothing_kernel,
                        unwrap_phase=unwrap_phase,
                        mag_mean=mag_mean,
                        mag_std=mag_std,
                        phase_mean=phase_mean,
                        phase_std=phase_std,
                        band_ranges=band_ranges)
        datasets.append(ds)
        labels_list.extend([ds.label] * len(ds))

    test_dataset = ConcatDataset(datasets)
    print("Total test samples:", len(test_dataset))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=2)

    class_names = ["Healthy", "Damage 2.96", "Damage 5.92", "Damage 8.87"]

    # Model hyperparameters
    cnn_channels = 32
    cnn_kernel = 3
    cnn_stride = 1
    d_model = 64
    nhead = 4
    num_layers = 3
    dropout = 0.1
    max_seq_len = 550

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = HybridFRFTransformer(
        input_dim=4,
        cnn_channels=cnn_channels,
        cnn_kernel=cnn_kernel,
        cnn_stride=cnn_stride,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=len(conditions),
        dropout=dropout,
        max_seq_len=max_seq_len
    ).to(device)

    model_path = "./best_models/best_hybrid_model.pth"  # Adjust path
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {model_path}")
    else:
        print("Model weights not found at", model_path)
        return

    true_labels, preds, cm = evaluate_model(model, test_loader, device, class_names)

    plot_confusion_matrix_counts(cm, class_names)
    plot_confusion_matrix_percent_total(cm, class_names)

if __name__ == "__main__":
    inference_main()
