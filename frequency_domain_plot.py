import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

plot_folder = "plots"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

folder_pattern = "./Data/*Beam*"
folders = sorted([f for f in glob.glob(folder_pattern) if os.path.isdir(f)])
print(folders)
files_dict = {}
for folder in folders:
    txt_files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    files_dict[folder] = txt_files

num_tests = 70

for i in range(num_tests):
    plt.figure()
    for folder, file_list in files_dict.items():
        if i < len(file_list):
            file_path = file_list[i]
            try:
                data = pd.read_csv(file_path, sep="\t", engine="python")
                data = data.iloc[:1500]
            except Exception:
                data = pd.read_csv(file_path, delim_whitespace=True, engine="python")
                data = data.iloc[:1500]
            freq = data["Frequency"]
            mag = data["Magnitude"]
            plt.plot(freq, mag, label=os.path.basename(folder))
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title(f"Test {i+1}: Frequency vs. Magnitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_folder, f"test_{i+1}_magnitude.png"))
    plt.close()

    plt.figure()
    for folder, file_list in files_dict.items():
        if i < len(file_list):
            file_path = file_list[i]
            try:
                data = pd.read_csv(file_path, sep="\t", engine="python")
                data = data.iloc[:1500]
            except Exception:
                data = pd.read_csv(file_path, delim_whitespace=True, engine="python")
                data = data.iloc[:1500]
            freq = data["Frequency"]
            phase = data["Phase"]
            plt.plot(freq, phase, label=os.path.basename(folder))
    plt.xlabel("Frequency")
    plt.ylabel("Phase (degrees)")
    plt.title(f"Test {i+1}: Frequency vs. Phase")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_folder, f"test_{i+1}_phase.png"))
    plt.close()
