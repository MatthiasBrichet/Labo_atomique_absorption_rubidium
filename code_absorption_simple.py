"""
Analyse de l'absorption simple - Spectroscopie du Rubidium (780 nm)
LPHYS1347 - Laboratoire 2

Version corrigée :
- Ordre réel des pics : 87Rb(F=1), 85Rb(F=2), 85Rb(F=3), 87Rb(F=2)
- Calibration fréquence basée sur les deux séparations hyperfines :
    85Rb : pic2 ↔ pic3 = 3036 MHz
    87Rb : pic1 ↔ pic4 = 6835 MHz
- Calcul de température avec les masses appropriées
- Abondance isotopique : 85Rb = pics 2+3, 87Rb = pics 1+4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES PHYSIQUES
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_NM   = 780e-9          # longueur d'onde de la transition (m)
C           = 3e8             # vitesse de la lumière (m/s)
KB          = 1.380649e-23    # constante de Boltzmann (J/K)
M85         = 85 * 1.66054e-27  # masse atomique 85Rb (kg)
M87         = 87 * 1.66054e-27  # masse atomique 87Rb (kg)
SIGMA_RB    = 2.9e-13         # section efficace du Rb à 780 nm (m²)
NU0         = C / LAMBDA_NM   # fréquence centrale de la transition (Hz)

# Écarts hyperfins du fondamental (en Hz)
HFS_85 = 3036e6   # 85Rb : F=2 → F=3
HFS_87 = 6835e6   # 87Rb : F=1 → F=2

# Ordre réel des 4 pics principaux (balayage fréquence croissante) :
#   Pic 1 : 87Rb F=2 → F'
#   Pic 2 : 85Rb F=3 → F'
#   Pic 3 : 85Rb F=2 → F'
#   Pic 4 : 87Rb F=1 → F'
PEAK_LABELS = [
    r"$^{87}$Rb  $F=2 \to F'$",
    r"$^{85}$Rb  $F=3 \to F'$",
    r"$^{85}$Rb  $F=2 \to F'$",
    r"$^{87}$Rb  $F=1 \to F'$",
]

# Masse correspondante pour chaque pic (pour le calcul de T)
PEAK_MASSES = [M87, M85, M85, M87]

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES UTILISATEUR  ← modifier si besoin
# ─────────────────────────────────────────────────────────────────────────────
CSV_FILE        = "SDS824X_HD_CSV_C2_2.csv"   # chemin vers ton fichier CSV
SKIP_ROWS       = 2             # nombre de lignes d'en-tête à ignorer
DOWNSAMPLE      = 10            # facteur de sous-échantillonnage
SMOOTH_WINDOW   = 50            # fenêtre de lissage pour détection
PEAK_PROMINENCE = 0.005
PEAK_DISTANCE   = 500
FIT_WINDOW_PTS  = 3000

# Positions manuelles des 4 pics (ms) dans l'ordre réel (87,85,85,87)
MANUAL_PEAKS_MS = [15.8, 19.1, 27.2, 33.8]

# Longueur du chemin d'absorption dans la cellule (m)
CELL_LENGTH_M   = 0.05          # 5 cm

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def gaussian(x, A, x0, sigma, offset):
    return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + offset

def load_data(filepath, skip_rows=2, downsample=1):
    print(f"Chargement du fichier : {filepath}")
    df = pd.read_csv(filepath, header=None, skiprows=skip_rows,
                     names=["time", "voltage"])
    t = df["time"].values.astype(float)
    v = df["voltage"].values.astype(float)
    if downsample > 1:
        t = t[::downsample]
        v = v[::downsample]
    print(f"  {len(t):,} points chargés (facteur ×{downsample}).")
    return t, v

def detect_peaks(t, v, prominence=0.005, distance=500, smooth=50):
    v_smooth = uniform_filter1d(v, size=smooth)
    neg_v = -v_smooth
    neg_v -= neg_v.min()
    peaks, props = find_peaks(neg_v, prominence=prominence * 0.3, distance=distance)
    if len(peaks) < 4:
        peaks, props = find_peaks(neg_v, prominence=0.001, distance=distance // 2)
    peak_voltages = v_smooth[peaks]
    order = np.argsort(peak_voltages)
    top4 = sorted(peaks[order[:4]])
    return np.array(top4), v_smooth

def calibrate_freq_two_pairs(t_peaks, delta_nu1, idx_pair1, delta_nu2, idx_pair2):
    """
    Calibration linéaire temps → fréquence en utilisant deux paires de pics.
    delta_nu1, delta_nu2 : écarts de fréquence réels (Hz)
    idx_pair1 = (i, j) indices des deux pics dans t_peaks pour la première paire
    Retourne la fonction t_to_freq(t) et la pente (Hz/s).
    """
    i1, j1 = idx_pair1
    i2, j2 = idx_pair2
    dt1 = t_peaks[j1] - t_peaks[i1]
    dt2 = t_peaks[j2] - t_peaks[i2]
    slope1 = delta_nu1 / dt1
    slope2 = delta_nu2 / dt2
    slope = (slope1 + slope2) / 2
    t_ref = t_peaks[0]   # origine à pic1 (87Rb F=1)
    def t_to_freq(t):
        return (t - t_ref) * slope
    print(f"Calibration : pente = {slope/1e6:.2f} MHz/s")
    print(f"  basée sur : {delta_nu1/1e6:.0f} MHz pour pic{idx_pair1} et {delta_nu2/1e6:.0f} MHz pour pic{idx_pair2}")
    return t_to_freq, slope

def four_gaussians_baseline(x, A1,x1,s1, A2,x2,s2, A3,x3,s3, A4,x4,s4, a, b):
    g1 = A1 * np.exp(-0.5 * ((x - x1) / s1) ** 2)
    g2 = A2 * np.exp(-0.5 * ((x - x2) / s2) ** 2)
    g3 = A3 * np.exp(-0.5 * ((x - x3) / s3) ** 2)
    g4 = A4 * np.exp(-0.5 * ((x - x4) / s4) ** 2)
    return g1 + g2 + g3 + g4 + a * x + b

def fit_peaks(t, v, peak_indices, window_pts=3000, t_to_freq=None):
    """
    Fit simultané de 4 gaussiennes + baseline linéaire.
    Retourne une liste de résultats pour chaque pic.
    """
    margin = window_pts
    lo = max(0, peak_indices[0] - margin)
    hi = min(len(t) - 1, peak_indices[-1] + margin)
    t_fit = t[lo:hi]
    v_fit = v[lo:hi]

    # Baseline linéaire estimée aux bords
    n = len(t_fit)
    n_edge = max(10, n // 20)
    v_l = np.mean(v_fit[:n_edge]); t_l = np.mean(t_fit[:n_edge])
    v_r = np.mean(v_fit[-n_edge:]); t_r = np.mean(t_fit[-n_edge:])
    a0 = (v_r - v_l) / (t_r - t_l)
    b0 = v_l - a0 * t_l

    sig0 = 1.5e-3  # largeur temporelle initiale (s)
    p0 = []
    for idx in peak_indices:
        A0 = v[idx] - (a0 * t[idx] + b0)
        p0 += [A0, t[idx], sig0]
    p0 += [a0, b0]

    lb, ub = [], []
    for idx in peak_indices:
        lb += [-np.inf, t[idx] - 2e-3, 0.05e-3]
        ub += [0,       t[idx] + 2e-3, 2e-3]
    lb += [-np.inf, -np.inf]
    ub += [np.inf,   np.inf]

    try:
        popt, pcov = curve_fit(four_gaussians_baseline, t_fit, v_fit,
                               p0=p0, bounds=(lb, ub), maxfev=100000)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError as e:
        print(f"  ⚠ Fit global échoué : {e}")
        popt = p0 + [0]*2
        perr = np.zeros(14)

    # Reconstruction des résultats individuels
    results = []
    for i, idx in enumerate(peak_indices):
        win = window_pts
        lo_i = max(0, idx - win)
        hi_i = min(len(t) - 1, idx + win)
        t_win = t[lo_i:hi_i]
        v_win = v[lo_i:hi_i]
        baseline_win = popt[12] * t_win + popt[13]
        v_corr = v_win - baseline_win
        Ai, xi, si = popt[3*i], popt[3*i+1], popt[3*i+2]
        popt_gauss = np.array([Ai, xi, si, 0.0])
        perr_gauss = np.array([perr[3*i], perr[3*i+1], perr[3*i+2], 0.0])
        results.append((popt_gauss, perr_gauss, t_win, v_corr, baseline_win))
    return results

# ─────────────────────────────────────────────────────────────────────────────
# PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Chargement
    t, v = load_data(CSV_FILE, skip_rows=SKIP_ROWS, downsample=DOWNSAMPLE)

    # 2. Détection des 4 pics (manuelle ou auto)
    print("\nDétection des 4 pics d'absorption...")
    v_smooth = uniform_filter1d(v, size=SMOOTH_WINDOW)

    if MANUAL_PEAKS_MS is not None:
        peak_idx = []
        for t_ms in MANUAL_PEAKS_MS:
            idx = np.argmin(np.abs(t * 1e3 - t_ms))
            lo = max(0, idx - 200)
            hi = min(len(t) - 1, idx + 200)
            idx_local = lo + np.argmin(v_smooth[lo:hi])
            peak_idx.append(idx_local)
        peak_idx = np.array(peak_idx)
        print("  Pics définis manuellement :")
    else:
        peak_idx, _ = detect_peaks(t, v, prominence=PEAK_PROMINENCE,
                                   distance=PEAK_DISTANCE, smooth=SMOOTH_WINDOW)

    if len(peak_idx) != 4:
        print(f"  ⚠ {len(peak_idx)} pics trouvés au lieu de 4. Vérifie MANUAL_PEAKS_MS.")
        return
    else:
        print("  4 pics détectés :")
        for i, idx in enumerate(peak_idx):
            print(f"    Pic {i+1} : t = {t[idx]*1e3:.3f} ms,  V = {v[idx]:.4f} V")

    # 3. Calibration fréquence avec deux paires
    # Ordre des pics : 0:87F1, 1:85F2, 2:85F3, 3:87F2
    # Paires : (1,2) pour 85Rb (3036 MHz), (0,3) pour 87Rb (6835 MHz)
    t_peaks = t[peak_idx]
    t_to_freq, slope_hz_s = calibrate_freq_two_pairs(
        t_peaks,
        HFS_85, (1, 2),   # 85Rb : pic2 ↔ pic3
        HFS_87, (0, 3)    # 87Rb : pic1 ↔ pic4
    )
    freq = t_to_freq(t)   # fréquence relative (Hz), référence pic1

    # 4. Fit gaussien
    print("\nFit gaussien des 4 pics...")
    fit_results = fit_peaks(t, v_smooth, peak_idx, window_pts=FIT_WINDOW_PTS)

    # 5. Températures
    print("\n--- Températures déduites de la largeur Doppler ---")
    temperatures = []
    for i, (res, label, mass) in enumerate(zip(fit_results, PEAK_LABELS, PEAK_MASSES)):
        popt, perr, _, _, _ = res
        if popt is None:
            continue
        sigma_t = abs(popt[2])          # largeur temporelle (s)
        sigma_nu = sigma_t * abs(slope_hz_s)   # largeur en fréquence (Hz)
        T = mass * (C * sigma_nu / NU0) ** 2 / KB
        temperatures.append(T)
        sigma_nu_MHz = sigma_nu / 1e6
        label_clean = label.replace("$", "").replace("\\", "")
        print(f"  Pic {i+1} ({label_clean}) : "
              f"σ_ν = {sigma_nu_MHz:.1f} MHz  →  T = {T:.1f} K")

    if temperatures:
        T_mean = np.mean(temperatures)
        print(f"\n  Température moyenne : {T_mean:.1f} K  ({T_mean-273.15:.1f} °C)")

    # 6. Abondance isotopique
    print("\n--- Abondance isotopique ---")
    if len(fit_results) == 4 and all(r[0] is not None for r in fit_results):
        areas = [abs(r[0][0]) * abs(r[0][2]) * np.sqrt(2 * np.pi) for r in fit_results]
        # 85Rb : pics 2 et 3 (indices 1 et 2)
        area_85 = areas[1] + areas[2]
        # 87Rb : pics 1 et 4 (indices 0 et 3)
        area_87 = areas[0] + areas[3]
        total = area_85 + area_87
        abund_85 = area_85 / total * 100
        abund_87 = area_87 / total * 100
        print(f"  85Rb : {abund_85:.1f}%  (référence naturelle : 72.17%)")
        print(f"  87Rb : {abund_87:.1f}%  (référence naturelle : 27.83%)")
    else:
        print("  Fit insuffisant pour calculer l'abondance.")

    # 7. Densité de Rb (loi de Beer-Lambert)
    print("\n--- Densité de Rb ---")
    if len(fit_results) >= 1 and fit_results[0][0] is not None:
        # On prend le pic le plus profond (amplitude absolue la plus grande)
        depths = [abs(r[0][0]) for r in fit_results if r[0] is not None]
        best_idx = np.argmax(depths)
        popt_best = fit_results[best_idx][0]
        V_offset = popt_best[3]            # baseline locale = I₀
        V_bottom = V_offset + popt_best[0] # tension au creux = I
        ratio = V_bottom / V_offset
        if 0 < ratio < 1:
            n_rb = -np.log(ratio) / (SIGMA_RB * CELL_LENGTH_M)
            print(f"  I/I₀ = {ratio:.3f}  (pic {best_idx+1})")
            print(f"  n_Rb = {n_rb:.2e} m⁻³  =  {n_rb/1e6:.2e} cm⁻³")
        else:
            print("  ⚠ Ratio I/I₀ hors plage, vérifie le fit.")

    # 8. Figures
    plt.style.use("default")
    colors_peak = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a"]

    # Figure 1 : temps
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(t * 1e3, v, color="#adb5bd", lw=0.5, alpha=0.6, label="Signal brut")
    ax1.plot(t * 1e3, v_smooth, color="#343a40", lw=1.2, label="Signal lissé")
    for i, idx in enumerate(peak_idx):
        ax1.axvline(t[idx] * 1e3, color=colors_peak[i], ls="--", lw=1.2)
        ax1.annotate(f"Pic {i+1}", xy=(t[idx]*1e3, v[idx]),
                     xytext=(0, -25), textcoords="offset points",
                     ha="center", fontsize=9, color=colors_peak[i],
                     arrowprops=dict(arrowstyle="->", color=colors_peak[i]))
    ax1.set_xlabel("Temps (ms)")
    ax1.set_ylabel("Tension (V)")
    ax1.set_title("Spectre d'absorption simple — échelle temporelle")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    plt.show()

    # Figure 2 : fréquence
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    freq_MHz = freq / 1e6
    ax2.plot(freq_MHz, v_smooth, color="#343a40", lw=1.2)
    for i, idx in enumerate(peak_idx):
        ax2.axvline(freq_MHz[idx], color=colors_peak[i], ls="--", lw=1.2)
        ax2.annotate(PEAK_LABELS[i], xy=(freq_MHz[idx], v_smooth[idx]),
                     xytext=(0, -30), textcoords="offset points",
                     ha="center", fontsize=9, color=colors_peak[i],
                     arrowprops=dict(arrowstyle="->", color=colors_peak[i]))
    ax2.set_xlabel("Fréquence relative (MHz)")
    ax2.set_ylabel("Tension (V)")
    ax2.set_title("Spectre d'absorption simple — échelle en fréquence (calibrée)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    plt.show()

    # Figures individuelles par pic
    for i, (res, label, color) in enumerate(zip(fit_results, PEAK_LABELS, colors_peak)):
        fig_i, ax_i = plt.subplots(figsize=(8, 5))
        popt, perr, t_win, v_win, baseline = res
        freq_win = t_to_freq(t_win) / 1e6
        ax_i.plot(freq_win, v_win, "o", color=color, ms=1.5, alpha=0.5, label="Données")
        if popt is not None:
            freq_fit = np.linspace(freq_win[0], freq_win[-1], 1000)
            t_fit = freq_fit * 1e6 / slope_hz_s + t[peak_idx[0]]
            v_fit = gaussian(t_fit, *popt)
            ax_i.plot(freq_fit, v_fit, "-", color="black", lw=2, label="Fit gaussien")
            sigma_nu_MHz = abs(popt[2]) * abs(slope_hz_s) / 1e6
            FWHM_MHz = 2 * np.sqrt(2 * np.log(2)) * sigma_nu_MHz
            T_pic = PEAK_MASSES[i] * (C * sigma_nu_MHz * 1e6 / NU0) ** 2 / KB
            textstr = f"σ_ν = {sigma_nu_MHz:.1f} MHz\nFWHM = {FWHM_MHz:.1f} MHz\nT = {T_pic:.0f} K"
            ax_i.text(0.97, 0.05, textstr, transform=ax_i.transAxes, fontsize=10,
                      va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
        ax_i.set_title(f"Pic {i+1} : {label}  —  Fit gaussien (Doppler)", color=color, fontweight="bold")
        ax_i.set_xlabel("Fréquence relative (MHz)")
        ax_i.set_ylabel("Tension (V)")
        ax_i.legend()
        ax_i.grid(True, alpha=0.3)
        fig_i.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()