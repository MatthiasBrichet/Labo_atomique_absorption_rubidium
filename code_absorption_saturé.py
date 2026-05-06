import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ========== PARAMÈTRES PHYSIQUES ==========
HFS_85_HZ = 3036e6
HFS_87_HZ = 6835e6

# ========== PARAMÈTRES UTILISATEUR ==========
CSV_FILE = "SDS824X_HD_CSV_C2_1.csv"   # fichier saturé (C2_1)
SKIP_ROWS = 2
DOWNSAMPLE = 5
SMOOTH_WINDOW = 20
PEAK_PROMINENCE = 0.002
PEAK_DISTANCE = 50

# ----- CALIBRATION MANUELLE -----
# Indices des 4 pics dans l'ordre chronologique (0,1,2,3)
# À ajuster selon tes données (ex: 0=87F1, 1=85F2, 2=85F3, 3=87F2)
# Spécifie la paire pour 85Rb (deux pics proches) et pour 87Rb (deux pics éloignés)
PAIR_85Rb = (1, 2)   # indices des deux pics du 85Rb dans t_peaks
PAIR_87Rb = (0, 3)   # indices des deux pics du 87Rb (extrêmes)

# Fenêtre d'analyse autour du 3ème pic (en MHz)
REGION_HALF_WIDTH_MHZ = 2000
PEAK_SAT_PROMINENCE = 0.0005
PEAK_SAT_DISTANCE = 10

# ========== FONCTIONS ==========
def load_data(filepath, skip_rows=2, downsample=1):
    df = pd.read_csv(filepath, header=None, skiprows=skip_rows, names=["time", "voltage"])
    t = df["time"].values.astype(float)
    v = df["voltage"].values.astype(float)
    if downsample > 1:
        t = t[::downsample]
        v = v[::downsample]
    return t, v

def find_four_deepest_peaks(t, v, smooth=20, prominence=0.002, distance=50):
    v_smooth = uniform_filter1d(v, size=smooth)
    neg_v = -v_smooth
    neg_v -= neg_v.min()
    peaks, _ = find_peaks(neg_v, prominence=prominence, distance=distance)
    if len(peaks) < 4:
        peaks, _ = find_peaks(neg_v, prominence=prominence/2, distance=distance//2)
    depths = v_smooth[peaks]
    deepest_idx = np.argsort(depths)[:4]
    deepest_peaks = np.sort(peaks[deepest_idx])
    return deepest_peaks, v_smooth

def calibrate_freq_manual(t_peaks, idx85, idx87):
    """
    idx85 = (i, j) indices des pics du 85Rb
    idx87 = (i, j) indices des pics du 87Rb
    Retourne t_to_freq_hz et slope_hz_s
    """
    i85, j85 = idx85
    i87, j87 = idx87
    dt85 = abs(t_peaks[j85] - t_peaks[i85])
    dt87 = abs(t_peaks[j87] - t_peaks[i87])
    slope85 = HFS_85_HZ / dt85
    slope87 = HFS_87_HZ / dt87
    slope = (slope85 + slope87) / 2
    t0 = t_peaks[0]   # origine au premier pic
    def t_to_freq_hz(t):
        return (t - t0) * slope
    print(f"Pente de calibration : {slope/1e6:.2f} MHz/s")
    print(f"  Utilisation : écart {dt85*1e3:.3f} ms -> {HFS_85_HZ/1e6} MHz (85Rb)")
    print(f"               écart {dt87*1e3:.3f} ms -> {HFS_87_HZ/1e6} MHz (87Rb)")
    return t_to_freq_hz, slope

def detect_peaks_in_region(t, v, freq_hz, f_center_hz, half_width_hz,
                           prominence=0.0005, distance=10):
    fmin = f_center_hz - half_width_hz
    fmax = f_center_hz + half_width_hz
    mask = (freq_hz >= fmin) & (freq_hz <= fmax)
    t_win = t[mask]
    v_win = v[mask]
    freq_win = freq_hz[mask]
    v_smooth = uniform_filter1d(v_win, size=5)
    neg_v = -v_smooth
    neg_v -= neg_v.min()
    peaks, _ = find_peaks(neg_v, prominence=prominence, distance=distance)
    return t_win[peaks], v_win[peaks], freq_win[peaks]

# ========== MAIN ==========
def main():
    t, v = load_data(CSV_FILE, SKIP_ROWS, DOWNSAMPLE)
    print(f"Données chargées : {len(t)} points")

    # Graphique temporel
    plt.figure(figsize=(12,5))
    plt.plot(t*1e3, v, 'b-', lw=0.5, alpha=0.7)
    plt.xlabel("Temps (ms)")
    plt.ylabel("Tension (V)")
    plt.title("Spectre brut - Absorption saturée")
    plt.grid(alpha=0.3)
    plt.show()

    # Détection des 4 pics principaux
    peak_indices, v_smooth = find_four_deepest_peaks(t, v, SMOOTH_WINDOW,
                                                     PEAK_PROMINENCE, PEAK_DISTANCE)
    t_peaks = t[peak_indices]
    print("4 pics principaux (ms) :", t_peaks*1e3)

    # Calibration manuelle avec les paires spécifiées
    t_to_freq_hz, slope = calibrate_freq_manual(t_peaks, PAIR_85Rb, PAIR_87Rb)
    freq_hz = t_to_freq_hz(t)
    freq_mhz = freq_hz / 1e6

    # Spectre fréquentiel
    plt.figure(figsize=(12,5))
    plt.plot(freq_mhz, v, 'b-', lw=0.5, alpha=0.7)
    plt.xlabel("Fréquence relative (MHz)")
    plt.ylabel("Tension (V)")
    plt.title("Spectre complet - Échelle fréquentielle")
    plt.grid(alpha=0.3)
    plt.show()

    # 3ème pic
    pic_index = 2
    t_pic = t_peaks[pic_index]
    f_pic_hz = t_to_freq_hz(t_pic)
    f_pic_mhz = f_pic_hz / 1e6
    print(f"3ème pic : t = {t_pic*1e3:.2f} ms, f = {f_pic_mhz:.1f} MHz")

    # Détection des raies de saturation dans la région
    half_width_hz = REGION_HALF_WIDTH_MHZ * 1e6
    t_sat, v_sat, f_sat_hz = detect_peaks_in_region(t, v, freq_hz, f_pic_hz,
                                                    half_width_hz,
                                                    PEAK_SAT_PROMINENCE,
                                                    PEAK_SAT_DISTANCE)
    f_sat_mhz = f_sat_hz / 1e6
    print(f"Nombre de raies de saturation : {len(t_sat)}")
    if len(f_sat_mhz) > 0:
        print("Fréquences (MHz) :", np.sort(f_sat_mhz))

    # Zoom
    mask = (freq_mhz >= f_pic_mhz - REGION_HALF_WIDTH_MHZ) & \
           (freq_mhz <= f_pic_mhz + REGION_HALF_WIDTH_MHZ)
    plt.figure(figsize=(12,6))
    plt.plot(freq_mhz[mask], v[mask], 'k-', lw=1, label='Spectre')
    plt.plot(f_sat_mhz, v_sat, 'ro', markersize=5, label='Raies détectées')
    plt.axvline(f_pic_mhz, color='orange', ls='--', lw=1.5, label='Centre 3ème creux')
    plt.xlabel("Fréquence relative (MHz)")
    plt.ylabel("Tension (V)")
    plt.title("Zoom sur le 3ème pic")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()