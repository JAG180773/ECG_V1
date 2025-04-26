
# -*- coding: utf-8 -*-
"""ECG Viewer – Stage 4 (sin PDF)
Basado en Stage 3 pero con UI de pestañas y sin generación de PDF.
"""

import streamlit as st
st.set_page_config(
    page_title="Visor ECG IA",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"   # opcional
)
st.set_page_config(layout="wide", page_title="ECG Analyzer", page_icon="🫀")

from typing import Optional, List, Tuple, Dict
import os, re, numpy as np, pandas as pd, wfdb, plotly.graph_objs as go
import neurokit2 as nk
import torch, torch.nn as nn

DB_NAME, DOWNLOAD_DIR = "ecg-arrhythmia", "./ecg-arrhythmia-1.0.0"
DEFAULT_SNOMED = {0: "Bradicardia", 1: "Normal", 2: "Fibrilación auricular", 3: "Taquicardia"}

# ---------------------------------------------------------------------
# 0. Utilidades de etiquetas SNOMED
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_condition_map(csv: str = "ConditionNames_SNOMED-CT.csv") -> dict[str, str]:
    """Lee el CSV proporcionado por el usuario y devuelve {code: label}."""
    if not os.path.exists(csv):
        return {}
    try:
        df = pd.read_csv(csv)
        # Intentamos adivinar las columnas relevantes de forma flexible
        cols = {c.lower(): c for c in df.columns}
        code_col = cols.get("snomed_code") or cols.get("code") or list(df.columns)[0]
        name_col = cols.get("condition") or cols.get("name") or list(df.columns)[1]
        return dict(zip(df[code_col].astype(str), df[name_col].astype(str)))
    except Exception as e:
        st.sidebar.warning(f"No se pudo leer el CSV de etiquetas: {e}")
        return {}

COND_MAP: dict[str, str] = load_condition_map()

# Si el CSV trae las etiquetas básicas (0‑3) aprovechamos para remplazar
SNOMED_LABELS = {int(k): v for k, v in COND_MAP.items() if k.isdigit() and int(k) in DEFAULT_SNOMED}
if len(SNOMED_LABELS) < 4:
    SNOMED_LABELS = DEFAULT_SNOMED  # fallback

# ------------------------------------------------------------------
# 1. Carga PhysioNet / Local
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _download_subset(db: str, folder: str):
    from wfdb import dl_database, get_record_list
    base = f"WFDBRecords/{folder}"
    subset = [r for r in get_record_list(db) if r.startswith(base)]
    dl_database(db, dl_dir=DOWNLOAD_DIR, records=subset)

def load_physionet() -> Tuple[Optional[wfdb.Record], Optional[str]]:
    ds = st.sidebar.selectbox("Dataset PhysioNet", [f"{i:02}" for i in range(1, 47)])
    sub = [f"{ds}/460", f"{ds}/461"] if ds == "46" else [f"{ds}/{ds}{i}" for i in range(10)]
    folder = st.sidebar.selectbox("Carpeta", sub)
    path = os.path.join(DOWNLOAD_DIR, "WFDBRecords", folder)
    if not (os.path.isdir(path) and any(f.endswith(".hea") for f in os.listdir(path))):
        if st.sidebar.button("Descargar carpeta"):
            _download_subset(DB_NAME, folder)
            st.sidebar.success("Descarga completada")
    recs = sorted(f[:-4] for f in os.listdir(path)) if os.path.isdir(path) else []
    rec = st.sidebar.selectbox("Registro", recs) if recs else None
    return (wfdb.rdrecord(os.path.join(path, rec)), rec, path) if rec else (None, None, None)

def load_local() -> Tuple[Optional[wfdb.Record], Optional[str]]:
    base = st.sidebar.text_input("Ruta base", "ECG_DB/WFDBRecords")
    ds = st.sidebar.selectbox("Dataset local", [f"{i:02}" for i in range(1, 47)])
    sub = [f"{ds}/460", f"{ds}/461"] if ds == "46" else [f"{ds}/{ds}{i}" for i in range(10)]
    folder = st.sidebar.selectbox("Carpeta", sub)
    path = os.path.join(base, folder)
    recs = sorted(f[:-4] for f in os.listdir(path)) if os.path.isdir(path) else []
    rec = st.sidebar.selectbox("Registro", recs) if recs else None
    return (wfdb.rdrecord(os.path.join(path, rec)), rec, path) if rec else (None, None, None)

# ------------------------------------------------------------------
# 2. Procesamiento ECG
# ------------------------------------------------------------------

def best_lead(rec):
    return rec.sig_name[np.argmax([np.std(rec.p_signal[:, i]) for i in range(rec.n_sig)])]

def get_sig(rec, lead):
    fs = rec.fs
    idx = rec.sig_name.index(lead)
    sig = rec.p_signal[:, idx]
    t = np.arange(len(sig)) / fs
    return sig, t, fs

def zoom(t, s):
    rng = st.slider("⏱️ Zoom (s)", 0.0, float(t[-1]), (0.0, min(10.0, float(t[-1]))), 0.04)
    m = (t >= rng[0]) & (t <= rng[1])
    return t[m], s[m]

def detect_r(sz, tz, fs):
    clean = nk.ecg_clean(sz, sampling_rate=fs)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs)
    r = info["ECG_R_Peaks"]
    return r, 60 / np.diff(tz[r]).mean() if len(r) > 1 else ([], None)

def hrv_metrics(peaks, fs):
    return nk.hrv_time(peaks=peaks, sampling_rate=fs, show=False)[["HRV_SDNN", "HRV_RMSSD", "HRV_pNN50"]] if len(peaks) >= 3 else pd.DataFrame()


def compute_hr(peaks_idx: np.ndarray, time_vec: np.ndarray):
    """Devuelve HR (media RR) y HR (mediana RR) en bpm, y clasificación."""
    if len(peaks_idx) < 2:
        return None, None, "Sin datos"
    rr = np.diff(time_vec[peaks_idx])
    hr_mean = 60 / np.mean(rr)
    hr_med  = 60 / np.median(rr)
    # Clasificación simple
    if hr_mean < 60:
        state = "Bradicardia"
    elif hr_mean > 100:
        # Detectar irregularidad → posible FA (coeficiente variación RR > 0.15)
        cv = np.std(rr) / np.mean(rr)
        state = "Fibrilación auricular" if cv > 0.15 else "Taquicardia"
    else:
        state = "Normal"
    return hr_mean, hr_med, state

# ------------------------------------------------------------------
# 3. MLP dummy
# ------------------------------------------------------------------
class ECGClassifierMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(10, 20), nn.Linear(20, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# ---- entrenamiento sintético ----
def _synthetic_dataset(n=500):
    X, y = [], []
    for _ in range(n):
        label = np.random.choice(4)
        base  = np.random.uniform(1.1,1.4) if label==0 else \
                np.random.uniform(0.4,0.55) if label==3 else \
                np.random.uniform(0.6,1.0)
        seq   = np.random.normal(base, 0.05, 10)
        X.append(seq); y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def _real_dataset(records: list[wfdb.Record]):
    """Convierte registros reales en vectores RR (10) + etiqueta."""
    X, y = [], []
    for rec in records:
        lead = best_lead(rec)
        sig, t, fs = get_sig(rec, lead)
        clean = nk.ecg_clean(sig, sampling_rate=fs)
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        r = info["ECG_R_Peaks"]
        if len(r) < 11:
            continue
        for i in range(len(r) - 10):
            rr = np.diff(r[i:i+11]) / fs      # s
            hr = 60 / rr.mean()
            cv = rr.std() / rr.mean()
            if hr < 60:
                lbl = 0                      # Bradicardia
            elif hr > 100 and cv <= 0.15:
                lbl = 3                      # Taquicardia
            elif cv > 0.15:
                lbl = 2                      # FA
            else:
                lbl = 1                      # Normal
            X.append(rr.astype(np.float32))
            y.append(lbl)
    return (np.asarray(X, np.float32), np.asarray(y, np.int64)) if X else (None, None)


def train_mlp(model: ECGClassifierMLP,
              real: bool = False,
              records: Optional[List[wfdb.Record]] = None,
              epochs: int = 30,
              lr: float = 1e-3):
#def train_mlp(model: ECGClassifierMLP, epochs: int = 30, lr: float = 1e-3,
              #real: bool = False, recs: list | None = None):
    """Entrena con datos reales o sintéticos."""
    if real and records:
        X, y = _real_dataset(records)
        if X is None:
            st.sidebar.warning("Dataset real vacío; uso sintético.")
            X, y = _synthetic_dataset()
    else:
        X, y = _synthetic_dataset()

    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(torch.tensor(X)), torch.tensor(y))
        loss.backward(); opt.step()
    torch.save(model.state_dict(), "mlp_weights.pth")
    st.sidebar.success("MLP entrenado y pesos guardados.")


def load_mlp(p: str = "mlp_weights.pth") -> ECGClassifierMLP:
    m = ECGClassifierMLP()
    if os.path.exists(p):
        m.load_state_dict(torch.load(p, map_location="cpu"))
        st.sidebar.success("Pesos MLP cargados")
    else:
        st.sidebar.info("MLP dummy (sin entrenamiento)")
    m.eval()
    return m

def launch_training(model: ECGClassifierMLP, current_rec: wfdb.Record):
    """Botón Sidebar para lanzar el entrenamiento."""
    src = st.sidebar.selectbox("Dataset entrenamiento", ["Sintético", "Registro actual"])
    if st.sidebar.button("Entrenar MLP"):
        train_mlp(model, real=(src == "Registro actual"), recs=[current_rec])
# ------------------------------------------------------------------
# 4. APP PRINCIPAL
# ------------------------------------------------------------------

def main() -> None:
    mlp = load_mlp()
    st.title("🫀 Visor ECG con IA")
    st.sidebar.title("Fuente de datos")
    mode = st.sidebar.radio("Origen", ["PhysioNet", "Local"])
    rec, rec_name, rec_path = (load_physionet() if mode == "PhysioNet" else load_local())
    if rec is None:
        st.info("Selecciona un registro")
        return

    manual = st.sidebar.checkbox("Derivación manual")
    lead = st.sidebar.selectbox("Derivación", rec.sig_name) if manual else best_lead(rec)

    sig, t, fs = get_sig(rec, lead)
    tz, sz = zoom(t, sig)
    r_idx, freq = detect_r(sz, tz, fs)

    tab_ecg, tab_hrv, tab_cls, tab_info = st.tabs(["ECG", "HRV", "Clasificación", "Info"])

    # ------- TAB ECG -------
    
    with tab_ecg:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tz, y=sz * 0.1, mode="lines", name=lead))
        if len(r_idx):
            fig.add_trace(go.Scatter(x=tz[r_idx], y=sz[r_idx] * 0.1, mode="markers",
                                     marker=dict(color="red", size=6), name="R"))
        fig.update_layout(height=470, plot_bgcolor="white", xaxis_title="Tiempo (s)", yaxis_title="Amplitud (mV)")
        st.plotly_chart(fig, use_container_width=True)

        # ----- HR metrics & state
        hr_mean, hr_med, hr_state = compute_hr(r_idx, tz)
        if hr_mean is None:
            st.info("No se detectaron suficientes picos R para HR.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("HR media", f"{hr_mean:.1f} bpm")
            c2.metric("HR mediana", f"{hr_med:.1f} bpm")
            c3.metric("Estado", hr_state)
            if hr_state != "Normal":
                st.error(f"⚠️ {hr_state} detectada")
    #with tab_ecg:
     #   fig = go.Figure()
      #  fig.add_trace(go.Scatter(x=tz, y=sz * 0.1, mode="lines", name=lead))
       # if len(r_idx):
        #    fig.add_trace(go.Scatter(x=tz[r_idx], y=sz[r_idx] * 0.1, mode="markers",
         #                            marker=dict(color="red", size=6), name="R"))
       # fig.update_layout(height=480, plot_bgcolor="white", xaxis_title="Tiempo (s)", yaxis_title="Amplitud (mV)")
       # st.plotly_chart(fig, use_container_width=True)
        
        
        # ---- HR analysis -------------------------------------------------
        #hr_mean, hr_med = compute_hr(r_idx, tz)
        #if hr_mean is None:
        #    st.info("No se detectaron suficientes picos R para calcular HR.")
        #else:
           # c1, c2 = st.columns(2)
            #c1.metric("HR (media RR)",   f"{hr_mean:.1f} bpm")
            #c2.metric("HR (mediana RR)", f"{hr_med:.1f} bpm")
        
            # Alerta fuera de rango clínico
            #if hr_mean < 60 or hr_mean > 100:
             #   st.error("⚠️ Frecuencia cardíaca fuera del rango 60-100 lpm")
    # ------- TAB HRV -------
    with tab_hrv:
        metr = hrv_metrics(r_idx, fs)
        if metr.empty:
            st.info("Se necesitan ≥3 picos R para HRV")
        else:
            st.dataframe(metr.T, use_container_width=True)

    # ------- TAB Clasificación -------
    with tab_cls:
 
         # ------------- Entrenamiento ---------------------------------
        st.markdown("### Entrenar MLP")
        col1, col2 = st.columns(2)
        if col1.button("Sintético"):
            train_mlp(mlp, real=False)
        if col2.button("Registro actual"):
            train_mlp(mlp, real=True, records=[rec])
        st.divider()       
 
        if st.button("Clasificar ritmo"):
            if len(r_idx) >= 11:
                rr = np.diff(r_idx) / fs
                lbl_idx = mlp(torch.tensor(rr[:10], dtype=torch.float32).unsqueeze(0)).argmax().item()
                lbl = SNOMED_LABELS.get(lbl_idx, f"Clase {lbl_idx}")
                st.success(f"Resultado MLP: **{lbl}**")
            else:
                st.error("Se requieren ≥11 picos R")
        #if st.sidebar.button("Entrenar MLP (sintético)"):
            train_mlp(mlp, epochs=20)
    # ------- TAB Info -------
    with tab_info:
        st.subheader("Metadatos del registro")
        st.write("**Archivo:**", rec_name)
        st.write("**Ruta:**", rec_path)
        st.write("**Frecuencia de muestreo:**", f"{fs} Hz")
        st.write("**Muestras:**", len(sig))
        st.write("**Duración:**", f"{len(sig)/fs:.2f} s")
        st.write("**Derivaciones:**", ", ".join(rec.sig_name))

        if rec.comments:
            st.write("**Comentarios (header):**")
            for c in rec.comments:
                st.markdown(f"- {c}")
        else:
            st.info("Sin comentarios en el header")

        # Extraemos códigos SNOMED (≥6 dígitos) de los comentarios
        codes = set(re.findall(r"\b\d{6,}\b", " ".join(rec.comments))) if rec.comments else set()
        if codes:
            st.markdown("**Etiquetas detectadas:**")
            for c in sorted(codes):
                st.markdown(f"- `{c}` → {COND_MAP.get(c, 'Etiqueta no encontrada')}")
        elif COND_MAP:
            st.info("No se encontraron códigos SNOMED en los comentarios del header.")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
