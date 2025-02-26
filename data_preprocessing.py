import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings 

class FRFDataset(Dataset):
    def __init__(self, root_dir, condition, transform=None):
        """
        root_dir: Parent directory containing folders for each condition.
        condition: Folder name (e.g., "Beam_healthy", "Beam_damage_2.96", etc.)
        """
        self.folder = os.path.join(root_dir, condition)
        self.files = sorted(glob.glob(os.path.join(self.folder, "P-*.txt")))
        self.transform = transform
        # Map condition to label (example mapping)
        self.label_map = {"Beam_healthy": 0, "Beam_damage_2.96": 1, 
                          "Beam_damage_5.92": 2, "Beam_damage_8.87": 3}
        self.label = self.label_map[condition]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        print(file_path)
        with warnings.catch_warnings(record=True) as w: 
            warnings.simplefilter("always", pd.errors.ParserWarning) 
            df = pd.read_csv(file_path, sep="\t", engine="python", encoding='latin1', on_bad_lines='warn') 
            if w: 
                first_warning = w[0].message 
                print("First parser warning:", first_warning)
        # try:
        #     df = pd.read_csv(file_path, sep="\t", engine="python",encoding='latin1', on_bad_lines='warn')
        # except Exception:
        #     df = pd.read_csv(file_path, delim_whitespace=True, engine="python")
        # print(df.shape)
        # Extract columns
        # freq = df["Frequency"].values.astype(np.float32)
        # mag = df["Magnitude"].values.astype(np.float32)
        # phase = df["Phase"].values.astype(np.float32)
        # Form token: [Frequency, Magnitude, Phase]
        # tokens = np.stack([freq, mag, phase], axis=-1)  # shape: (L, 3)
        # if self.transform:
        #     tokens = self.transform(tokens)
        # # Convert to tensor
        # tokens = torch.tensor(tokens, dtype=torch.float32)
        # label = torch.tensor([self.label], dtype=torch.long)
        # return tokens, label
        return 0,1


def main():
    root_dir = "./Data"
    conditions = ['Beam_healthy', "Beam_damage_2.96", "Beam_damage_5.92", "Beam_damage_8.87"]
    for condition in conditions:
        # print(condition)
        frf_dataset = FRFDataset(root_dir, condition)
        for i in range(70):
            # print(i)
            tokens, label = frf_dataset[i]
        # print(tokens.shape, label.shape)

if __name__=='__main__':
    main()