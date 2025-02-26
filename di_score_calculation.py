#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd

# ----------------------------
# Helper Functions
# ----------------------------
def load_frf_data(filename, freq_col="Frequency", mag_col="Magnitude", phase_col="Phase"):
    """
    Load FRF data from a file (CSV or TXT) with columns Frequency, Magnitude, and Phase.
    """
    try:
        df = pd.read_csv(filename, sep="\t", engine="python")
    except Exception:
        df = pd.read_csv(filename, sep="\s+", engine="python")
    freq = df[freq_col].values.astype(np.float32)
    mag_db = df[mag_col].values.astype(np.float32)
    phase_deg = df[phase_col].values.astype(np.float32)
    return freq, mag_db, phase_deg

def frf_to_complex(mag_db, phase_deg):
    """
    Convert FRF data from magnitude in dB and phase in degrees to its complex representation.
    H(ω) = 10^(mag_db/20) * exp(j * deg2rad(phase_deg))
    """
    amplitude = 10 ** (mag_db / 20.0)
    # print(amplitude, phase_deg)
    phase_rad = np.deg2rad(phase_deg)
    phase_rad = phase_deg
    # print(phase_rad)
    return amplitude * np.exp(1j * phase_rad)

def select_frequency_bands(freq, mag_db, phase_deg, band_ranges):
    """
    Given full FRF data, return only those points whose frequencies lie in any of the specified bands.
    band_ranges: list of tuples [(low1, high1), (low2, high2), ...]
    """
    mask = np.zeros_like(freq, dtype=bool)
    for (low, high) in band_ranges:
        mask |= (freq >= low) & (freq <= high)
    return freq[mask], mag_db[mask], phase_deg[mask]

def compute_frac_curve(H_d, H_u, epsilon=1e-8):
    """
    Compute the FRAC curve using the formula:
      FRAC(ω) = |H_d(ω)*conjugate(H_u(ω))|^2 / (|H_d(ω)|^2 * |H_u(ω)|^2 + epsilon)
    """
    numerator = np.abs(H_d * np.conjugate(H_u)) ** 2
    denominator = (np.abs(H_d) ** 2) * (np.abs(H_u) ** 2) + epsilon
    return numerator / denominator

# ----------------------------
# Main Processing Script
# ----------------------------
def main():
    # --- User Configuration ---
    # Damage conditions and corresponding folders
    damage_conditions = ["./Data/Beam_damage_2.96", "./Data/Beam_damage_5.92", "./Data/Beam_damage_8.87"]
    # Healthy folder (assumes healthy files have same names as in each damage folder)
    healthy_folder = "./Data/Beam_healthy"
    
    # Define the frequency bands (in Hz) to focus on (e.g., mode 1, mode 2, mode 3)
    band_ranges = [(0, 5),(6,12),(13,20),(20,30),(31,35),(36,40)] #(10, 40), (120, 160), 
    band_labels = [f"{low}-{high}" for (low, high) in band_ranges]

    # Prepare an Excel writer (CSV cannot have multiple sheets)
    output_excel = "FRAC_summary_metrics.xlsx"
    writer = pd.ExcelWriter(output_excel, engine="xlsxwriter")

    # Loop through each damage condition
    for condition in damage_conditions:
        print(f"Processing condition: {condition}")
        # Folder for this damage condition
        damage_folder = condition
        # List all damaged files in the folder (assume naming pattern "P-*.txt")
        damaged_files = sorted(glob.glob(os.path.join(damage_folder, "P-*.txt")))
        healthy_files = sorted(glob.glob(os.path.join(healthy_folder, "P-*.txt")))
        # Prepare list to collect results for this condition
        results = []

        # Loop through damaged files for the current condition
        for d_file, h_file in zip(damaged_files,healthy_files):
            base_name = os.path.basename(h_file)
            healthy_file = os.path.join(healthy_folder, base_name)
            
            if not os.path.exists(healthy_file):
                print(f"Healthy file {healthy_file} not found. Skipping {base_name}.")
                continue

            # Load FRF data for damaged and healthy
            freq_d, mag_db_d, phase_deg_d = load_frf_data(d_file)
            freq_h, mag_db_h, phase_deg_h = load_frf_data(healthy_file)

            # Dictionary to hold metrics for the file
            file_metrics = {"filename": base_name}

            # Process each band separately
            for band, label in zip(band_ranges, band_labels):
                low, high = band
                # Filter the FRF data for the current frequency band
                freq_d_f, mag_db_d_f, phase_deg_d_f = select_frequency_bands(freq_d, mag_db_d, phase_deg_d, [band])
                freq_h_f, mag_db_h_f, phase_deg_h_f = select_frequency_bands(freq_h, mag_db_h, phase_deg_h, [band])
                
                # If no data points are present in this band, store NaN values
                if len(freq_d_f) == 0 or len(freq_h_f) == 0:
                    print(f"No data in band {label} for file {base_name}.")
                    file_metrics[f"{label}_avg_FRAC"] = np.nan
                    file_metrics[f"{label}_min_FRAC"] = np.nan
                    continue

                # print(band)
                # Convert the filtered data to complex form
                H_d_f = frf_to_complex(mag_db_d_f, phase_deg_d_f)
                # print(H_d_f);exit()
                H_h_f = frf_to_complex(mag_db_h_f, phase_deg_h_f)

                # Compute the FRAC curve for this band
                frac_curve = compute_frac_curve(H_d_f, H_h_f)

                # Compute summary metrics: average and minimum FRAC in the band
                avg_frac = np.mean(frac_curve)
                min_frac = np.min(frac_curve)

                file_metrics[f"{label}_avg_FRAC"] = avg_frac
                file_metrics[f"{label}_min_FRAC"] = min_frac

            results.append(file_metrics)
            print(f"Processed {base_name} for condition {condition}.")

        # Convert the results for this condition into a DataFrame
        df_results = pd.DataFrame(results)
        # Write to a separate sheet in the Excel file (sheet name as damage percentage)
        df_results.to_excel(writer, sheet_name=os.path.split(condition)[-1], index=False)
        print(f"Finished processing {condition}.")

    # Save the Excel workbook
    writer.close()
    print(f"FRAC summary metrics saved to {output_excel}")

if __name__ == "__main__":
    main()
