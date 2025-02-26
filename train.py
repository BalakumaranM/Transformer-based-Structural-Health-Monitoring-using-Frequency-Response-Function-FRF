import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime


def moving_average_smoothing(array, kernel_size=5):
    if kernel_size < 2:
        return array
    pad_size = kernel_size // 2
    padded = np.pad(array, (pad_size, pad_size), mode='edge')
    smoothed = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode='valid')
    return smoothed

def compute_frac_curve(H_d, H_u, epsilon=1e-8):
    numerator = np.abs(H_d * np.conjugate(H_u)) ** 2
    denominator = (np.abs(H_d) ** 2) * (np.abs(H_u) ** 2) + epsilon
    frac_curve = numerator / denominator
    return frac_curve

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
    """
    Processes FRF files with phase unwrapping, downsampling, frequency scaling,smoothing, 
    and z-score normalization. Computes the per-frequency FRAC
    (by comparing with a healthy reference) and concatenates it as a 4th channel.
    Each token becomes a 4-dim vector:
         [scaled_frequency, norm_magnitude, norm_phase, FRAC]
    """
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

        # Compute FRAC curve
        H_d = 10 ** (mag_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))
        if self.condition == "Beam_healthy":
            H_u = H_d
        else:
            healthy_file = self.healthy_files[idx]
            freq_h, mag_h, phase_h = process_file(healthy_file,
                                                   downsample_factor=self.downsample_factor,
                                                   unwrap_phase=self.unwrap_phase,
                                                   smoothing=self.smoothing,
                                                   smoothing_kernel=self.smoothing_kernel,
                                                   band_ranges=self.band_ranges)
            H_u = 10 ** (mag_h / 20.0) * np.exp(1j * np.deg2rad(phase_h))
        frac_curve = compute_frac_curve(H_d, H_u)  # shape (L,)
        frac_channel = frac_curve[:, np.newaxis]  # shape (L, 1)
        tokens = np.concatenate([tokens, frac_channel], axis=-1)  # Now shape (L, 4)

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
        # Using batch_first=True in the transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, input_dim, seq_length) for CNN
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # reduces sequence length by factor of 2
        x = x.transpose(1, 2)  # (B, new_seq_length, cnn_channels)
        x = self.input_proj(x)  # (B, new_seq_length, d_model)
        seq_length = x.shape[1]
        pos_emb = self.pos_embedding[:seq_length, :].unsqueeze(0)  # (1, seq_length, d_model)
        x = x + pos_emb
        # Transformer encoder with batch_first=True expects (B, seq_length, d_model)
        x = self.encoder(x)  # (B, seq_length, d_model)
        x = x.mean(dim=1)  # (B, d_model)
        logits = self.classifier(x)
        return logits


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=50, patience=10):
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_train_loss = 10
    best_epoch = 0
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            print("Best model saved!")
            epochs_no_improve = 0
        
        elif train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), "best_hybrid_model_train_acc.pth")

        elif train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), "best_hybrid_model_train_loss.pth")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Plot and save training graphs
    os.makedirs("training_graph", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Acc")
    plt.plot(range(1, len(val_accs) + 1), val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("training_graph", f"training_graph_{timestamp}.png"))
    plt.close()

    print("Training complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    return model


def main():
    root_dir = "./Data" 
    freq_max = 600.0
    downsample_factor = 1
    smoothing = False
    smoothing_kernel = 5
    unwrap_phase = False
    band_ranges = [(10, 40), (120, 160), (350, 450)]  #frequency bands; 

    mag_mean, mag_std = 0.0, 1.0
    phase_mean, phase_std = 0.0, 1.0

    conditions = ["Beam_healthy", "Beam_damage_2.96", "Beam_damage_5.92", "Beam_damage_8.87"]
    datasets = []
    labels_list = []
    for cond in conditions:
        ds = FRFDataset(
            root_dir=root_dir,
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
            band_ranges=band_ranges
        )
        datasets.append(ds)
        labels_list.extend([ds.label] * len(ds))

    full_dataset = ConcatDataset(datasets)
    total_samples = len(full_dataset)
    print(f"Total samples: {total_samples}")

    indices = np.arange(total_samples)
    labels_array = np.array(labels_list)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels_array,
        random_state=57653
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    batch_size = 2 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Hybrid Model Hyperparameters
    cnn_channels = 32
    cnn_kernel = 3
    cnn_stride = 1
    d_model = 64
    nhead = 4
    num_layers = 4
    dropout = 0.1
    max_seq_len = 550  # As the input is (10-40 Hz, 120-160 Hz, 350-450 Hz, with interval of 0.35 Hz)

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

    load_last_model = True
    best_model_path = "./best_models/best_hybrid_model.pth"
    if load_last_model and os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {best_model_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 100
    patience = 20

    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs,
        patience=patience
    )

    torch.save(trained_model.state_dict(), "final_hybrid_model.pth")
    print("Final model saved as final_hybrid_model.pth.")

if __name__ == "__main__":
    main()


# Training scores

# For ranges (10-40) (120-160), (350-450) (max_seq -560) (early stopping 10)
    # without any preprocessing , batch_size =8
    # Epoch [8/50] Train Loss: 1.3267 | Train Acc: 0.4062 || Val Loss: 1.3624 | Val Acc: 0.3571
    # Best model saved!


    # with phase unwrapping , batch_size =8
    # Epoch [2/50] Train Loss: 1.3926 | Train Acc: 0.2321 || Val Loss: 1.3906 | Val Acc: 0.3036
    # Best model saved!

    # num_layers = 2 , batch_size =8
    # Epoch [11/50] Train Loss: 1.2303 | Train Acc: 0.5312 || Val Loss: 1.3131 | Val Acc: 0.5000
    # Best model saved!

    # num_layers = 2 , batch_size =4 (slow improvement in the model accuracy, so beautiful to see. It was not as sudden as batch_size 8, batch_size 8 was sudden but then went into different area where training accuracy was increasing but validation accuracy was not). But here after 11 th epoch every epoch producing the best model)
    # Epoch [43/50] Train Loss: 0.4239 | Train Acc: 0.8259 || Val Loss: 0.5343 | Val Acc: 0.7321

    # num_layers = 2 , batch_size =4 (early stopping 20 )
    # Epoch [50/100] Train Loss: 0.2403 | Train Acc: 0.9018 || Val Loss: 0.6712 | Val Acc: 0.8036

    # num_layers = 3, batch_size=2
    # Epoch [14/100] Train Loss: 0.4673 | Train Acc: 0.7946 || Val Loss: 0.4368 | Val Acc: 0.8571

    # Fine-tuned on top of the last best model (num_layers = 3, batch_size=2)
    # Epoch [17/100] Train Loss: 0.1222 | Train Acc: 0.9598 || Val Loss: 0.0526 | Val Acc: 1.0000

# For ranges (0, 200), (201, 400), (401, 1000) (max_seq -3200)
    # without any preprocessing
    # Epoch [2/50] Train Loss: 1.3918 | Train Acc: 0.2232 || Val Loss: 1.3963 | Val Acc: 0.3214
    # Best model saved!