import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile
import io
from pathlib import Path
from PIL import Image
import base64
import pickle
import os
import re
import zipfile
import shutil
import warnings
import json
import textwrap
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------- Optional deps (graceful fallbacks) ----------
# Audio
try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

# Albumentations
try:
    import albumentations as A
    HAS_ALB = True
except Exception:
    HAS_ALB = False

# Embeddings
HAS_SENTENCE_T = False
HAS_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_T = True
except Exception:
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        HAS_TRANSFORMERS = True
    except Exception:
        pass

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except Exception:
    HAS_TF = False

# Missingness viz
try:
    import missingno as msno
    HAS_MISSINGNO = True
except Exception:
    HAS_MISSINGNO = False

# Profilers
try:
    from ydata_profiling import ProfileReport
    HAS_PROFILING = True
except Exception:
    HAS_PROFILING = False

try:
    import sweetviz as sv
    HAS_SWEETVIZ = True
except Exception:
    HAS_SWEETVIZ = False

import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ---------- Sklearn ----------
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, mean_absolute_error,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------- NLTK (robust & cached) ----------
import nltk
from nltk.data import find
from nltk.stem import PorterStemmer, WordNetLemmatizer

@st.cache_resource(show_spinner=False)
def ensure_nltk():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, pkg in resources.items():
        try:
            find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, ToktokTokenizer

def safe_word_tokenize(text: str):
    try:
        return word_tokenize(text)
    except LookupError:
        try:
            return ToktokTokenizer().tokenize(text)
        except Exception:
            return re.findall(r"\w+", text or "")

# ---------- Streamlit meta ----------
st.set_page_config(page_title="DataMentor - Universal Preprocessor", layout="wide")
st.title("🧠 DataMentor: Universal Data Profiling, Preprocessing & Training")

# ---------- Utilities ----------
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file)
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return None

def basic_df_profile(df: pd.DataFrame):
    st.subheader("📊 Dataset Preview")
    st.write(df.head())
    st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Column Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Duplicate Rows")
    st.write(df.duplicated().sum())

    st.subheader("Numeric Summary")
    st.write(df.describe())

def np_download(name: str, arr: np.ndarray):
    bio = io.BytesIO()
    np.save(bio, arr)
    st.download_button(f"Download {name}", bio.getvalue(), file_name=f"{name}.npy")

def allow_download_bytes(label: str, b: bytes, fname: str):
    st.download_button(label, data=b, file_name=fname)

def save_model_download_button(pipe, fname: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fp:
        pickle.dump(pipe, fp)
        path = fp.name
    with open(path, "rb") as f:
        st.download_button(f"Download model: {fname}", f.read(), file_name=fname)

# ---- Target auto-detection ----
def autodetect_target_column(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None
    cand_names = [c for c in df.columns]
    lower_map = {c: c.lower() for c in cand_names}
    common = ["target", "label", "labels", "class", "y"]
    for c in cand_names:
        if lower_map[c] in common:
            y = df[c]
            is_clf = (y.dtype == "O") or (y.nunique() <= max(20, int(0.05 * len(y))))
            return c, is_clf
    # fallback: last column
    last = df.columns[-1]
    y = df[last]
    is_clf = (y.dtype == "O") or (y.nunique() <= max(20, int(0.05 * len(y))))
    return last, is_clf

# ---- Missing value viz ----
def show_missing_value_viz(df: pd.DataFrame):
    st.subheader("🕳️ Missing-Value Visualizations")
    if df.isna().sum().sum() == 0:
        st.success("No missing values found 🎉")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.write("Per-column Missing Count")
        st.bar_chart(df.isna().sum())
    with c2:
        st.write("Per-row Missing Count (first 200 rows)")
        row_miss = df.isna().sum(axis=1).head(200)
        st.line_chart(row_miss)
    st.write("Matrix/Heatmap")
    if HAS_MISSINGNO:
        msno.matrix(df, figsize=(6, 3))
        st.pyplot(plt.gcf()); plt.clf()
        msno.heatmap(df, figsize=(6, 3))
        st.pyplot(plt.gcf()); plt.clf()
    else:
        st.caption("Install `missingno` for richer plots: pip install missingno")
        sample = df.head(200).isna().astype(int)
        plt.figure(figsize=(6, 3))
        plt.imshow(sample, aspect="auto", interpolation="nearest")
        plt.title("Missingness Matrix (1=missing)")
        plt.xlabel("Columns"); plt.ylabel("Rows (first 200)")
        st.pyplot(plt.gcf()); plt.clf()

# ---- Profiling reports ----
def render_profiling_report(df: pd.DataFrame):
    st.subheader("📋 Data Profiling Report (ydata-profiling / Sweetviz)")
    choice = st.selectbox("Choose a profiler", ["ydata-profiling (Pandas-Profiling)", "Sweetviz"])
    if st.button("Generate Report"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if choice.startswith("ydata"):
            if not HAS_PROFILING:
                st.error("Install: pip install ydata-profiling")
                return
            profile = ProfileReport(df, title="DataMentor Profile", minimal=True)
            out_html = f"profile_{ts}.html"
            profile.to_file(out_html)
            with open(out_html, "rb") as f:
                st.download_button("Download Profile (HTML)", f, file_name=out_html)
            components.html(open(out_html, "r", encoding="utf-8").read(), height=600, scrolling=True)
        else:
            if not HAS_SWEETVIZ:
                st.error("Install: pip install sweetviz")
                return
            report = sv.analyze(df)
            out_html = f"sweetviz_{ts}.html"
            report.show_html(out_html, open_browser=False)
            with open(out_html, "rb") as f:
                st.download_button("Download Sweetviz (HTML)", f, file_name=out_html)
            components.html(open(out_html, "r", encoding="utf-8").read(), height=600, scrolling=True)

# ---- Preprocessor builder / saver ----
def build_preprocessor_from_df(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median")),
                                   ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                   ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )
    return pre, num_cols, cat_cols

def save_object_as_pickle(obj, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fp:
        pickle.dump(obj, fp)
        path = fp.name
    with open(path, "rb") as f:
        st.download_button(f"Download {filename}", f, file_name=filename)

# ---- AutoML tabular ----
def run_automl_tabular(X_train, X_test, y_train, y_test):
    is_classification = (pd.Series(y_train).dtype == "O") or (pd.Series(y_train).nunique() <= max(20, int(0.05*len(y_train))))
    pre, _, _ = build_preprocessor_from_df(pd.concat([X_train, X_test], axis=0))

    candidates = {}
    if is_classification:
        candidates = {
            "LogReg": LogisticRegression(max_iter=400),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        }
        if HAS_XGB:
            candidates["XGBoost"] = XGBClassifier(
                n_estimators=400, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42
            )
    else:
        candidates = {
            "Linear": LinearRegression(),
            "RandomForestReg": RandomForestRegressor(n_estimators=300, random_state=42),
        }
        if HAS_XGB:
            candidates["XGBReg"] = XGBRegressor(
                n_estimators=400, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42
            )

    results, trained = [], {}
    for name, mdl in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", mdl)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        if is_classification:
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")
            results.append({"Model": name, "Accuracy": acc, "F1_weighted": f1})
        else:
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            results.append({"Model": name, "R2": r2, "MAE": mae})
        trained[name] = pipe

    res_df = pd.DataFrame(results)
    st.subheader("📊 AutoML Leaderboard")
    sort_col = "Accuracy" if is_classification else "R2"
    res_df = res_df.sort_values(by=sort_col, ascending=False)
    st.dataframe(res_df)

    best_name = res_df.iloc[0]["Model"]
    best_pipe = trained[best_name]
    st.success(f"🏆 Best model: **{best_name}**")

    # Plots
    if is_classification:
        preds = best_pipe.predict(X_test)
        labels = sorted(pd.Series(y_test).unique())
        cm = confusion_matrix(y_test, preds, labels=labels)
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix"); plt.colorbar()
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
        plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Pred")
        st.pyplot(plt.gcf()); plt.clf()
        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))
    else:
        preds = best_pipe.predict(X_test)
        plt.figure()
        plt.scatter(y_test, preds, s=12)
        plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Regression: True vs Pred")
        st.pyplot(plt.gcf()); plt.clf()

    save_object_as_pickle(best_pipe, f"automl_best_{best_name}.pkl")
    st.caption("Save standalone preprocessing pipeline:")
    save_object_as_pickle(best_pipe.named_steps["pre"], "preprocessing_pipeline.pkl")
    return best_pipe

# ---- FastAPI bundle generator ----
def generate_fastapi_bundle(example_columns):
    example_cols_json = json.dumps(list(example_columns))
    fastapi_code = f'''\
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

class Payload(BaseModel):
    features: dict

app = FastAPI()
with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

COLUMNS = {example_cols_json}

@app.post("/predict")
def predict(payload: Payload):
    x = payload.features
    row = [x.get(c, None) for c in COLUMNS]
    pred = MODEL.predict([row])[0]
    return {{"prediction": str(pred)}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    streamlit_client = f'''\
import streamlit as st
import requests

st.title("DataMentor Inference Client")
st.caption("FastAPI + Streamlit")

cols = {example_cols_json}
st.subheader("Input features")
vals = {{}}
for c in cols:
    vals[c] = st.text_input(c, "")

if st.button("Predict"):
    payload = {{"features": vals}}
    r = requests.post("http://localhost:8000/predict", json=payload, timeout=30)
    st.write("Response:", r.json())
'''

    tmpd = tempfile.mkdtemp()
    api_path = os.path.join(tmpd, "server_fastapi.py")
    cli_path = os.path.join(tmpd, "client_streamlit.py")
    with open(api_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(fastapi_code))
    with open(cli_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(streamlit_client))

    st.success("Generated FastAPI + Streamlit client files.")
    with open(api_path, "rb") as f:
        st.download_button("Download FastAPI server_fastapi.py", f, file_name="server_fastapi.py")
    with open(cli_path, "rb") as f:
        st.download_button("Download Streamlit client_streamlit.py", f, file_name="client_streamlit.py")

    st.caption("How to deploy:")
    st.code("""\
# Put model.pkl (best pipeline) alongside server_fastapi.py
pip install fastapi uvicorn pydantic requests
uvicorn server_fastapi:app --reload --port 8000

# In another terminal:
streamlit run client_streamlit.py
""")

# ---------- Tabular Preprocess ----------
def preprocess_tabular(df: pd.DataFrame):
    st.header("🧹 Tabular Data Preprocessing")

    missing_method = st.selectbox("Missing Values", ["None", "Mean", "Median", "Mode"])
    scale_method = st.selectbox("Scaling", ["None", "Standard", "MinMax", "Robust"])
    encoding = st.selectbox("Categorical Encoding", ["None", "Label", "One-Hot"])

    if st.button("Apply Preprocessing"):
        df_proc = df.copy()

        # Missing values
        if missing_method != "None":
            numeric = df_proc.select_dtypes(include=['number'])
            if not numeric.empty:
                strategy = "mean" if missing_method == "Mean" else "median" if missing_method == "Median" else "most_frequent"
                imputer = SimpleImputer(strategy=strategy)
                df_proc[numeric.columns] = imputer.fit_transform(numeric)

        # Encoding
        if encoding == "Label":
            le = LabelEncoder()
            for col in df_proc.select_dtypes(include="object"):
                try:
                    df_proc[col] = le.fit_transform(df_proc[col].astype(str))
                except Exception:
                    pass
        elif encoding == "One-Hot":
            df_proc = pd.get_dummies(df_proc, drop_first=False)

        # Scaling
        numeric_cols = df_proc.select_dtypes(include="number").columns
        scaler = None
        if scale_method == "Standard": scaler = StandardScaler()
        elif scale_method == "MinMax": scaler = MinMaxScaler()
        elif scale_method == "Robust": scaler = RobustScaler()
        if scaler and len(numeric_cols) > 0:
            df_proc[numeric_cols] = scaler.fit_transform(df_proc[numeric_cols])

        st.success("✅ Tabular preprocessing complete")
        st.dataframe(df_proc.head())

        allow_download_bytes("Download Preprocessed CSV", df_proc.to_csv(index=False).encode(), "processed_data.csv")
        st.session_state["df_processed"] = df_proc

    # Save preprocessing pipeline quickly
    if st.session_state.get("df_processed", None) is not None:
        st.subheader("📦 Save Preprocessing-Only Pipeline")
        dfp = st.session_state["df_processed"]
        if st.button("Save Pipeline (impute + scale + encode)"):
            pre, _, _ = build_preprocessor_from_df(dfp)
            pre.fit(dfp.copy())
            save_object_as_pickle(pre, "preprocessing_pipeline.pkl")

# ---------- Text Preprocess + Embeddings ----------
def preprocess_text(text: str):
    st.header("✍️ Text Preprocessing")

    col_cases, col_stop, col_stem, col_lemma = st.columns(4)
    lowercase = col_cases.checkbox("Lowercase", True)
    remove_stop = col_stop.checkbox("Stopwords Removal", True)
    do_stem = col_stem.checkbox("Stemming")
    do_lemma = col_lemma.checkbox("Lemmatization")

    if st.button("Clean Text"):
        try:
            stop_words = set(stopwords.words("english")) if remove_stop else set()
        except LookupError:
            stop_words = set()

        stemmer = PorterStemmer()
        lemmer = WordNetLemmatizer()

        tokens = safe_word_tokenize(text or "")

        if lowercase:
            tokens = [t.lower() for t in tokens]

        tokens = [re.sub(r"[^a-zA-Z0-9]", "", t) for t in tokens]
        tokens = [t for t in tokens if t]

        if remove_stop and stop_words:
            tokens = [t for t in tokens if t not in stop_words]

        if do_stem:
            tokens = [stemmer.stem(t) for t in tokens]

        if do_lemma:
            tokens = [lemmer.lemmatize(t) for t in tokens]

        res = " ".join(tokens)
        st.text_area("Cleaned Text", res, height=300)
        st.download_button("Download Cleaned Text", res, "cleaned_text.txt")
        st.session_state["last_text_clean"] = res

    # Embeddings (optional)
    st.subheader("🧠 (Optional) HuggingFace Sentence Embeddings")
    embed_src = st.radio("Embed:", ["Use cleaned text above", "Paste new text"], horizontal=True)
    text_for_embed = st.session_state.get("last_text_clean", text or "") if embed_src == "Use cleaned text above" else st.text_area("Text to embed", height=150, value=text or "")

    @st.cache_resource(show_spinner=False)
    def load_embedder():
        if HAS_SENTENCE_T:
            try:
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                return ("sbert", model)
            except Exception:
                pass
        if HAS_TRANSFORMERS:
            try:
                tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                return ("hf", (tok, mdl))
            except Exception:
                pass
        return ("none", None)

    kind, emb = load_embedder()

    def _mean_pool(last_hidden, attn_mask):
        mask = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    if st.button("Compute Embeddings"):
        if not text_for_embed.strip():
            st.warning("Please provide text.")
        else:
            if kind == "sbert":
                vec = emb.encode([text_for_embed])[0]
            elif kind == "hf":
                tok, mdl = emb
                import torch
                mdl.eval()
                with torch.no_grad():
                    batch = tok([text_for_embed], return_tensors="pt", truncation=True, max_length=256)
                    out = mdl(**batch)
                    vec = _mean_pool(out.last_hidden_state, batch["attention_mask"]).squeeze(0).numpy()
            else:
                st.error("Install `sentence-transformers` or `transformers` for embeddings.")
                return
            st.success(f"Embedding shape: {np.array(vec).shape}")
            st.code(np.array2string(np.array(vec), precision=4, suppress_small=True, max_line_width=100))
            np_download("text_embedding", np.array(vec))

# ---------- Image Preprocess ----------
def preprocess_image(img: Image.Image):
    st.header("🖼️ Image Preprocessing")

    resize = st.checkbox("Resize to 224×224")
    gray = st.checkbox("Convert to Grayscale")
    normalize = st.checkbox("Normalize (0–1)")
    st.markdown("---")

    st.subheader("🖼️ Image Augmentations (Albumentations)")
    aug_flip = st.checkbox("Horizontal Flip")
    aug_rotate = st.checkbox("Rotate ±15°")
    aug_brightness = st.checkbox("Random Brightness/Contrast")
    aug_noise = st.checkbox("Gaussian Noise")

    st.subheader("😎 Face Detection & Blur (Privacy)")
    do_face_blur = st.checkbox("Detect faces and blur")

    img_np = np.array(img.convert("RGB"))

    if resize:
        img_np = cv2.resize(img_np, (224, 224))
    if gray:
        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    if normalize:
        img_np = img_np.astype(np.float32) / 255.0

    if aug_flip or aug_rotate or aug_brightness or aug_noise:
        if not HAS_ALB:
            st.warning("Albumentations not installed. Falling back to basic OpenCV ops.")
            if aug_flip: img_np = cv2.flip(img_np, 1)
            if aug_rotate:
                h, w = img_np.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), 15, 1.0)
                img_np = cv2.warpAffine(img_np, M, (w, h))
            if aug_brightness and img_np.dtype != np.float32:
                img_np = cv2.convertScaleAbs(img_np, alpha=1.1, beta=10)
            if aug_noise:
                noise = np.random.normal(0, 5, size=img_np.shape)
                img_np = np.clip(img_np + noise, 0, 255).astype(img_np.dtype)
        else:
            aug_list = []
            if aug_flip: aug_list.append(A.HorizontalFlip(p=1.0))
            if aug_rotate: aug_list.append(A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_REFLECT_101))
            if aug_brightness: aug_list.append(A.RandomBrightnessContrast(p=1.0))
            if aug_noise: aug_list.append(A.GaussNoise(var_limit=(10.0, 50.0), p=1.0))
            transform = A.Compose(aug_list)
            if img_np.dtype != np.uint8:
                img_aug_in = (np.clip(img_np, 0, 1) * 255).astype(np.uint8) if img_np.dtype == np.float32 else img_np.astype(np.uint8)
            else:
                img_aug_in = img_np
            if img_aug_in.ndim == 2:
                img_aug_in = cv2.cvtColor(img_aug_in, cv2.COLOR_GRAY2RGB)
            out = transform(image=img_aug_in)
            img_np = out["image"]

    if do_face_blur:
        det_vis = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB) if img_np.ndim == 2 else img_np.copy()
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray_det = cv2.cvtColor(det_vis, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_det, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            roi = det_vis[y:y+h, x:x+w]
            roi = cv2.GaussianBlur(roi, (51, 51), 30)
            det_vis[y:y+h, x:x+w] = roi
        img_np = det_vis

    st.image(img_np, caption="Processed Image")
    out_img = img_np
    if out_img.dtype == np.float32:
        out_img = (np.clip(out_img, 0, 1) * 255).astype(np.uint8)
    if out_img.ndim == 2:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2RGB)
    _, out_buf = cv2.imencode(".png", cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    allow_download_bytes("Download Processed Image (PNG)", out_buf.tobytes(), "processed_image.png")

# ---------- Video Preprocess ----------
def preprocess_video(file):
    st.header("🎥 Video Preprocessing")

    extract_frames = st.checkbox("Extract Frames (1 FPS)")

    t = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    t.write(file.read())
    t.close()
    vid = cv2.VideoCapture(t.name)

    fps = int(vid.get(cv2.CAP_PROP_FPS)) or 1
    frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    st.info(f"FPS: {fps}, Total Frames: {frames}")

    count = 0
    frame_list = []

    if extract_frames:
        frame_idx = 0
        while True:
            ret, frame = vid.read()
            if not ret: break
            if frame_idx % fps == 0:
                frame_list.append(frame[:, :, ::-1])
                count += 1
            frame_idx += 1
        st.success(f"Extracted {count} frames")
        for f in frame_list[:5]:
            st.image(f)

    vid.release()

# ---------- Audio Preprocess ----------
def preprocess_audio(file):
    st.header("🎤 Audio Preprocessing")
    if not HAS_AUDIO:
        st.error("Audio libs not available. Please install: `pip install librosa soundfile`")
        return

    target_sr = st.selectbox("Target sample rate", [16000, 22050, 32000, 44100], index=0)
    do_trim = st.checkbox("Trim leading/trailing silence", True)
    do_normalize = st.checkbox("Normalize peak to -1..1", True)
    show_spec = st.checkbox("Show Mel-Spectrogram", True)

    suffix = Path(file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        in_path = tmp.name

    y, sr = librosa.load(in_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr); sr = target_sr
    if do_trim:
        y, _ = librosa.effects.trim(y, top_db=20)
    if do_normalize and np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    st.success(f"Audio: {len(y)} samples @ {sr} Hz")
    st.line_chart(pd.DataFrame({"amplitude": y[: min(len(y), 20000)]}))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_wav:
        sf.write(out_wav.name, y, sr)
        with open(out_wav.name, "rb") as f:
            allow_download_bytes("Download Processed WAV", f.read(), "processed_audio.wav")

    if show_spec:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr//2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig = plt.figure()
        plt.imshow(S_dB, aspect="auto", origin="lower")
        plt.title("Mel-Spectrogram"); plt.xlabel("Frames"); plt.ylabel("Mel bins")
        st.pyplot(fig)

    np_download("audio_waveform", y.astype(np.float32))

# --------------------- FEATURE ENGINEERING & TRAIN/TEST SPLIT ---------------------
st.header("Feature Engineering & Train/Test Split")

df_processed = st.session_state.get("df_processed", None)

if df_processed is not None:
    data = df_processed

    st.subheader("Target Column Selection")
    auto_tgt, auto_is_clf = autodetect_target_column(data)
    st.caption(f"Auto-detected target: **{auto_tgt}** (task: {'classification' if auto_is_clf else 'regression'})")
    target_col = st.selectbox("Select the target column", options=list(data.columns),
                              index=list(data.columns).index(auto_tgt) if auto_tgt in data.columns else 0)

    st.subheader("Feature Engineering Options")
    poly_feat = st.checkbox("Add Polynomial Features (Numeric only)")
    pca_feat = st.checkbox("Apply PCA (Dimensionality Reduction)")
    text_embed = st.checkbox("Convert Text to TF-IDF vectors (if text exists)")

    test_size = st.slider("Test Split Percentage", min_value=10, max_value=50, value=20)

    if st.button("Apply Feature Engineering + Split Data"):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.decomposition import PCA
        from sklearn.feature_extraction.text import TfidfVectorizer
        import joblib

        X = data.drop(columns=[target_col]).copy()
        y = data[target_col].copy()

        if poly_feat:
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = poly.fit_transform(X[num_cols])
                X = pd.concat([
                    pd.DataFrame(poly_features, columns=poly.get_feature_names_out(num_cols), index=X.index),
                    X.drop(columns=num_cols)
                ], axis=1)

        if pca_feat:
            numeric_data = X.select_dtypes(include=np.number)
            if numeric_data.shape[1] > 0:
                pca = PCA(n_components=min(5, numeric_data.shape[1]))
                pca_data = pca.fit_transform(numeric_data)
                pca_df = pd.DataFrame(pca_data, columns=[f"PCA{i+1}" for i in range(pca_data.shape[1])], index=X.index)
                X = pd.concat([pca_df, X.select_dtypes(exclude=np.number)], axis=1)

        if text_embed:
            from sklearn.feature_extraction.text import TfidfVectorizer
            text_cols = X.select_dtypes(include='object').columns
            for col in text_cols:
                if X[col].notna().sum() == 0:
                    X = X.drop(columns=[col]); continue
                tfidf = TfidfVectorizer(max_features=500)
                tfidf_matrix = tfidf.fit_transform(X[col].fillna(""))
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                                        columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                                        index=X.index)
                X = pd.concat([X.drop(columns=[col]), tfidf_df], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        Path("data/processed/").mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump((X_train, X_test, y_train, y_test), "data/processed/split_data.pkl")

        st.success("✅ Feature Engineering + Train/Test Split Completed")
        st.write("Training Data:", X_train.shape)
        st.write("Test Data:", X_test.shape)
        st.caption("Saved at: data/processed/split_data.pkl")

        st.session_state["split"] = (X_train, X_test, y_train, y_test, target_col)
else:
    st.info("ℹ️ Preprocess and set a dataframe first (Tabular section) to enable Feature Engineering.")

# --------------------- TRAINERS & EVALUATION ---------------------
# Jump shortcuts from media sections
default_index = 0
if st.session_state.pop("_jump_to_image_trainer", False):
    default_index = 2
elif st.session_state.pop("_jump_to_video_trainer", False):
    default_index = 3

st.markdown("### Model Training Options")
opt = st.radio(
    "Choose an option",
    [
        "1️⃣ Train ML models (tabular) – LR, RF, XGBoost",
        "2️⃣ Train NLP model – TF-IDF + Naive Bayes / LinearSVC",
        "3️⃣ Train Image model – CNN auto-trainer",
        "4️⃣ Train Video model – extract frames + CNN",
        "5️⃣ Evaluation Dashboard with metrics & plots",
    ],
    index=default_index
)

# ---------- 1) TABULAR ML TRAINER ----------
if opt.startswith("1️⃣"):
    if "split" not in st.session_state:
        st.warning("Please run Feature Engineering + Split first.")
    else:
        X_train, X_test, y_train, y_test, target_col = st.session_state["split"]
        is_classification = (y_train.dtype == "O") or (pd.Series(y_train).nunique() <= max(20, int(0.05*len(y_train))))
        st.write(f"Detected problem type: **{'Classification' if is_classification else 'Regression'}**")

        # Manual quick baseline leaderboard (optional)
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in X_train.columns if c not in num_cols]

        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), num_cols),
                ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ],
            remainder="drop"
        )

        results, models = [], {}

        if is_classification:
            candidates = {
                "LogReg": LogisticRegression(max_iter=300),
                "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
            }
            if HAS_XGB:
                candidates["XGBoost"] = XGBClassifier(
                    n_estimators=400, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42
                )
            for name, mdl in candidates.items():
                pipe = Pipeline([("pre", pre), ("model", mdl)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="weighted")
                results.append({"Model": name, "Accuracy": acc, "F1_weighted": f1})
                models[name] = pipe
        else:
            candidates = {
                "Linear": LinearRegression(),
                "RandomForestReg": RandomForestRegressor(n_estimators=300, random_state=42),
            }
            if HAS_XGB:
                candidates["XGBReg"] = XGBRegressor(
                    n_estimators=400, learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42
                )
            for name, mdl in candidates.items():
                pipe = Pipeline([("pre", pre), ("model", mdl)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                results.append({"Model": name, "R2": r2, "MAE": mae})
                models[name] = pipe

        if results:
            res_df = pd.DataFrame(results)
            sort_col = "Accuracy" if is_classification else "R2"
            res_df = res_df.sort_values(by=sort_col, ascending=False)
            st.dataframe(res_df)
            best = res_df.iloc[0]["Model"]
            st.success(f"✅ Best model: **{best}**")
            save_model_download_button(models[best], f"best_{best}.pkl")

        # 🚀 One-click AutoML + report + save preprocessing
        if st.button("🚀 Run AutoML & Report"):
            best_pipe = run_automl_tabular(X_train, X_test, y_train, y_test)
            st.subheader("🧩 Deploy")
            if st.button("Create FastAPI + Streamlit bundle"):
                cols = list(X_train.columns)
                generate_fastapi_bundle(cols)

# ---------- 2) NLP TRAINER ----------
elif opt.startswith("2️⃣"):
    dfp = st.session_state.get("df_processed", None)
    if dfp is None or len(dfp.select_dtypes(include="object").columns) == 0:
        st.warning("Please preprocess your tabular data and ensure a text + target column exist.")
    else:
        text_col = st.selectbox("Text column", dfp.select_dtypes(include="object").columns)
        target_col = st.selectbox("Target column", [c for c in dfp.columns if c != text_col])

        if st.button("Train TF-IDF + Classifier"):
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.svm import LinearSVC

            df_nn = dfp.dropna(subset=[text_col, target_col]).copy()
            X_text = df_nn[text_col].astype(str).fillna("")
            y = df_nn[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X_text, y, test_size=0.2, random_state=42,
                stratify=y if y.nunique()>1 else None
            )

            tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
            Xtr = tfidf.fit_transform(X_train)
            Xte = tfidf.transform(X_test)

            clf = MultinomialNB() if y_train.nunique() > 2 else LinearSVC()

            clf.fit(Xtr, y_train)
            preds = clf.predict(Xte)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            st.success(f"✅ Trained. Accuracy={acc:.4f}, F1(weighted)={f1:.4f}")

            labels = sorted(y_test.unique())
            cm = confusion_matrix(y_test, preds, labels=labels)
            fig = plt.figure()
            plt.imshow(cm, interpolation='nearest')
            plt.title("Confusion Matrix"); plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45); plt.yticks(tick_marks, labels)
            plt.tight_layout(); plt.ylabel('True'); plt.xlabel('Pred')
            st.pyplot(fig)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fp:
                pickle.dump({"vectorizer": tfidf, "model": clf}, fp)
                pth = fp.name
            with open(pth, "rb") as f:
                st.download_button("Download TFIDF+Model (.pkl)", f, file_name="nlp_tfidf_model.pkl")

# ---------- 3) IMAGE TRAINER ----------
elif opt.startswith("3️⃣"):
    st.info("Upload a ZIP with structure: train/<class>/*.jpg and optionally val/<class>/*.jpg")
    zipf = st.file_uploader("ZIP dataset", type=["zip"])
    epochs = st.slider("Epochs", 1, 20, 5)
    img_size = st.selectbox("Image size", [128, 160, 192, 224], index=3)
    batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=2)

    if zipf is not None:
        if not HAS_TF:
            st.error("TensorFlow not available. Install: `pip install tensorflow` (or tensorflow-cpu).")
        else:
            with tempfile.TemporaryDirectory() as tmpd:
                zpath = os.path.join(tmpd, "data.zip")
                with open(zpath, "wb") as f: f.write(zipf.read())
                with zipfile.ZipFile(zpath) as z: z.extractall(tmpd)

                train_dir = os.path.join(tmpd, "train")
                val_dir = os.path.join(tmpd, "val") if os.path.isdir(os.path.join(tmpd, "val")) else None

                if not os.path.isdir(train_dir):
                    st.error("ZIP must contain 'train/<class>/*.jpg'.")
                else:
                    seed = 123
                    train_ds = keras.utils.image_dataset_from_directory(
                        train_dir, image_size=(img_size, img_size), batch_size=batch_size, label_mode="categorical", seed=seed
                    )
                    if val_dir:
                        val_ds = keras.utils.image_dataset_from_directory(
                            val_dir, image_size=(img_size, img_size), batch_size=batch_size, label_mode="categorical", seed=seed
                        )
                    else:
                        val_ds = train_ds.take(1); train_ds = train_ds.skip(1)

                    AUTOTUNE = tf.data.AUTOTUNE
                    train_ds = train_ds.prefetch(AUTOTUNE)
                    val_ds = val_ds.prefetch(AUTOTUNE)

                    num_classes = len(train_ds.class_names)

                    try:
                        base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                                              include_top=False, weights="imagenet")
                        base.trainable = False
                        inputs = keras.Input(shape=(img_size, img_size, 3))
                        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
                        x = base(x, training=False)
                        x = layers.GlobalAveragePooling2D()(x)
                        x = layers.Dropout(0.2)(x)
                        outputs = layers.Dense(num_classes, activation="softmax")(x)
                        model = keras.Model(inputs, outputs)
                    except Exception:
                        model = keras.Sequential([
                            layers.Input((img_size, img_size, 3)),
                            layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
                            layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
                            layers.Conv2D(128, 3, activation="relu"), layers.GlobalAveragePooling2D(),
                            layers.Dense(num_classes, activation="softmax"),
                        ])

                    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

                    st.success("✅ Training complete")
                    st.write("Classes:", train_ds.class_names)
                    st.line_chart(pd.DataFrame({"train_acc": hist.history["accuracy"], "val_acc": hist.history["val_accuracy"]}))
                    st.line_chart(pd.DataFrame({"train_loss": hist.history["loss"], "val_loss": hist.history["val_loss"]}))

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as fp:
                        model.save(fp.name)
                        with open(fp.name, "rb") as f:
                            st.download_button("Download Keras Model (.h5)", f, file_name="image_model.h5")

# ---------- 4) VIDEO TRAINER ----------
elif opt.startswith("4️⃣"):
    st.info("Upload a ZIP with videos in class folders: train/<class>/*.mp4 (and optional val/...)")
    zipv = st.file_uploader("ZIP of video dataset", type=["zip"])
    epochs = st.slider("Epochs", 1, 10, 3)
    img_size = st.selectbox("Frame size", [128, 160, 192, 224], index=3)
    batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)
    fps_extract = st.selectbox("Extract FPS", [1, 2, 4], index=0)

    if zipv is not None:
        if not HAS_TF:
            st.error("TensorFlow not available. Install: `pip install tensorflow`.")
        else:
            with tempfile.TemporaryDirectory() as tmpd:
                zpath = os.path.join(tmpd, "vid.zip")
                with open(zpath, "wb") as f: f.write(zipv.read())
                with zipfile.ZipFile(zpath) as z: z.extractall(tmpd)

                train_dir_v = os.path.join(tmpd, "train")
                val_dir_v = os.path.join(tmpd, "val") if os.path.isdir(os.path.join(tmpd, "val")) else None
                if not os.path.isdir(train_dir_v):
                    st.error("ZIP must contain 'train/<class>/*.mp4'")
                else:
                    def extract_frames_dir(src_dir, out_dir, fps_target=1):
                        Path(out_dir).mkdir(parents=True, exist_ok=True)
                        for cls in os.listdir(src_dir):
                            src_cls = os.path.join(src_dir, cls)
                            if not os.path.isdir(src_cls): continue
                            dst_cls = os.path.join(out_dir, cls)
                            Path(dst_cls).mkdir(exist_ok=True)
                            for vid_name in os.listdir(src_cls):
                                if not vid_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")): continue
                                vpath = os.path.join(src_cls, vid_name)
                                cap = cv2.VideoCapture(vpath)
                                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
                                step = max(1, fps // fps_target)
                                idx = 0
                                while True:
                                    ret, frame = cap.read()
                                    if not ret: break
                                    if idx % step == 0:
                                        fr = cv2.resize(frame[:, :, ::-1], (img_size, img_size))  # BGR->RGB
                                        out_name = f"{Path(vid_name).stem}_{idx}.jpg"
                                        cv2.imwrite(os.path.join(dst_cls, out_name), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                                    idx += 1
                                cap.release()
                        return out_dir

                    frames_train = extract_frames_dir(train_dir_v, os.path.join(tmpd, "frames_train"), fps_extract)
                    frames_val = extract_frames_dir(val_dir_v, os.path.join(tmpd, "frames_val"), fps_extract) if val_dir_v else None

                    seed = 123
                    train_ds = keras.utils.image_dataset_from_directory(
                        frames_train, image_size=(img_size, img_size), batch_size=batch_size, label_mode="categorical", seed=seed
                    )
                    if frames_val and os.path.isdir(frames_val):
                        val_ds = keras.utils.image_dataset_from_directory(
                            frames_val, image_size=(img_size, img_size), batch_size=batch_size, label_mode="categorical", seed=seed
                        )
                    else:
                        val_ds = train_ds.take(1); train_ds = train_ds.skip(1)

                    AUTOTUNE = tf.data.AUTOTUNE
                    train_ds = train_ds.prefetch(AUTOTUNE)
                    val_ds = val_ds.prefetch(AUTOTUNE)

                    num_classes = len(train_ds.class_names)
                    model = keras.Sequential([
                        layers.Input((img_size, img_size, 3)),
                        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
                        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
                        layers.Conv2D(128, 3, activation="relu"), layers.GlobalAveragePooling2D(),
                        layers.Dense(num_classes, activation="softmax"),
                    ])
                    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

                    st.success("✅ Training complete (frame-based baseline)")
                    st.write("Classes:", train_ds.class_names)
                    st.line_chart(pd.DataFrame({"train_acc": hist.history["accuracy"], "val_acc": hist.history["val_accuracy"]}))
                    st.line_chart(pd.DataFrame({"train_loss": hist.history["loss"], "val_loss": hist.history["val_loss"]}))

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as fp:
                        model.save(fp.name)
                        with open(fp.name, "rb") as f:
                            st.download_button("Download Video Frame CNN (.h5)", f, file_name="video_frame_model.h5")

# ---------- 5) EVALUATION DASHBOARD ----------
elif opt.startswith("5️⃣"):
    if "split" not in st.session_state:
        st.warning("Please run Feature Engineering + Split first.")
    else:
        X_train, X_test, y_train, y_test, target_col = st.session_state["split"]
        st.subheader("Quick Evaluate a Saved/Trained Model")
        up = st.file_uploader("Upload a scikit-learn .pkl model", type=["pkl"])
        if up is not None:
            pipe = pickle.load(up)
            preds = pipe.predict(X_test)
            is_classification = (pd.Series(y_test).dtype == "O") or (pd.Series(y_test).nunique() <= max(20, int(0.05*len(y_test))))
            if is_classification:
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average="weighted")
                st.write({"Accuracy": acc, "F1_weighted": f1})
                labels = sorted(pd.Series(y_test).unique())
                cm = confusion_matrix(y_test, preds, labels=labels)
                fig = plt.figure()
                plt.imshow(cm, interpolation='nearest')
                plt.title("Confusion Matrix"); plt.colorbar()
                tick_marks = np.arange(len(labels))
                plt.xticks(tick_marks, labels, rotation=45); plt.yticks(tick_marks, labels)
                plt.tight_layout(); plt.ylabel('True'); plt.xlabel('Pred')
                st.pyplot(fig)
                st.text("Classification Report:")
                st.text(classification_report(y_test, preds))
            else:
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                st.write({"R2": r2, "MAE": mae})
                fig = plt.figure()
                plt.scatter(y_test, preds, s=10)
                plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Regression: True vs Pred")
                st.pyplot(fig)

# ---------------- Main UPLOAD UI ----------------
st.markdown("---")
st.subheader("📎 Upload File to Preprocess")
file = st.file_uploader(
    "Supported: CSV/XLS/XLSX/TXT/PNG/JPG/JPEG/MP4/AVI/WAV/MP3/FLAC/OGG/M4A",
    type=["csv", "xls", "xlsx", "txt", "png", "jpg", "jpeg", "mp4", "avi", "wav", "mp3", "flac", "ogg", "m4a"]
)

if file:
    ext = Path(file.name).suffix.lower()

    # CSV / Excel
    if ext in [".csv", ".xls", ".xlsx"]:
        df = load_file(file)
        basic_df_profile(df)
        # Missingness viz + Profiling report
        show_missing_value_viz(df)
        render_profiling_report(df)
        preprocess_tabular(df)

    # Text
    elif ext == ".txt":
        text = load_file(file)
        st.text_area("📄 Original Text", text, height=200)
        preprocess_text(text)

    # Image
    elif ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image")
        preprocess_image(img)
        st.markdown("—")
        if st.checkbox("🚀 Auto-train a quick image classifier now (use your ZIP dataset)"):
            st.info("Upload ZIP with structure: train/<class>/*.jpg and optional val/<class>/*.jpg")
            st.session_state["_jump_to_image_trainer"] = True

    # Video
    elif ext in [".mp4", ".avi"]:
        st.video(file)
        preprocess_video(file)
        st.markdown("—")
        if st.checkbox("🚀 Auto-train a quick video frame classifier now (use your ZIP dataset)"):
            st.info("Upload ZIP with videos in train/<class>/*.mp4 (optional val/..)")
            st.session_state["_jump_to_video_trainer"] = True

    # Audio
    elif ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        preprocess_audio(file)

st.markdown("---")
st.caption(
    "Included: Auto target detect • Missing-value viz • Profilers • Feature Engineering + Split • "
    "1) AutoML Tabular • 2) NLP TF-IDF • 3) Image CNN • 4) Video CNN • 5) Evaluation • FastAPI+Streamlit deploy"
)
