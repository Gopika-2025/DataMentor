import streamlit as st
import pandas as pd
import numpy as np
import io, os, re, pickle, tempfile, zipfile, warnings, shutil
from pathlib import Path
from PIL import Image
warnings.filterwarnings("ignore")

# ---------------- Optional deps (graceful fallbacks) ---------------- #
# ML extras
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except Exception:
    HAS_TF = False

# Audio
try:
    import librosa, soundfile as sf
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

# CV / aug
try:
    import cv2
    HAS_CV = True
except Exception:
    HAS_CV = False

try:
    import albumentations as A
    HAS_ALB = True
except Exception:
    HAS_ALB = False

# Missingness viz
try:
    import missingno as msno
    HAS_MSNO = True
except Exception:
    HAS_MSNO = False

# Profiling
try:
    from ydata_profiling import ProfileReport
    HAS_PROF = True
except Exception:
    HAS_PROF = False

try:
    import sweetviz as sv
    HAS_SWEET = True
except Exception:
    HAS_SWEET = False

# Embeddings (optional)
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

# ---------------- Sklearn ---------------- #
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, r2_score, mean_absolute_error,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------------- Streamlit Meta ---------------- #
st.set_page_config(page_title="DataMentor – Preprocess • AutoML • Deploy", layout="wide")
st.title("🧠 DataMentor: Universal Preprocessing • AutoML • FastAPI Export")

# ---------------- Utils ---------------- #
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

def guess_target_column(df: pd.DataFrame):
    """Heuristics for target column auto-detection."""
    candidates = ["target", "label", "class", "y", "outcome", "price"]
    lc = [c.lower() for c in df.columns]
    for name in candidates:
        if name in lc:
            return df.columns[lc.index(name)]
    # If last column is non-id looking and has few unique values, pick it
    last = df.columns[-1]
    if df[last].nunique() <= max(20, int(0.05 * len(df))):
        return last
    # Else fallback to last column
    return last

def allow_download_bytes(label: str, b: bytes, fname: str):
    st.download_button(label, data=b, file_name=fname)

def save_pickle(obj, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fp:
        pickle.dump(obj, fp)
        path = fp.name
    with open(path, "rb") as f:
        allow_download_bytes(f"Download {filename}", f.read(), filename)

# ---------------- Missing Value Visualizations ---------------- #
def show_missing_viz(df: pd.DataFrame):
    st.subheader("🧩 Missing-Value Visualizations")
    if HAS_MSNO:
        st.write("**missingno** matrix & bar")
        import matplotlib.pyplot as plt
        import matplotlib
        fig1 = plt.figure()
        msno.matrix(df)
        st.pyplot(fig1)

        fig2 = plt.figure()
        msno.bar(df)
        st.pyplot(fig2)
    else:
        st.info("`missingno` not installed, showing quick heatmap & counts.")
        import matplotlib.pyplot as plt
        null_counts = df.isna().sum().sort_values(ascending=False)
        st.write(null_counts[null_counts > 0])

        fig = plt.figure()
        plt.imshow(df.isna(), aspect='auto')
        plt.title("Missingness Heatmap")
        plt.xlabel("Columns"); plt.ylabel("Rows")
        st.pyplot(fig)

# ---------------- Profiling ---------------- #
def run_profilers(df: pd.DataFrame):
    st.subheader("📑 Data Profiling Reports")
    c1, c2 = st.columns(2)
    if c1.button("Generate Pandas-Profiling (ydata-profiling)"):
        if not HAS_PROF:
            st.error("Install: pip install ydata-profiling")
        else:
            prof = ProfileReport(df, title="DataMentor Profiling", explorative=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as fp:
                prof.to_file(fp.name)
                with open(fp.name, "rb") as f:
                    allow_download_bytes("Download Pandas-Profiling HTML", f.read(), "profiling_report.html")
            st.success("Generated Pandas-Profiling report.")
    if c2.button("Generate Sweetviz"):
        if not HAS_SWEET:
            st.error("Install: pip install sweetviz")
        else:
            r = sv.analyze(df)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as fp:
                r.show_html(filepath=fp.name, open_browser=False)
                with open(fp.name, "rb") as f:
                    allow_download_bytes("Download Sweetviz HTML", f.read(), "sweetviz_report.html")
            st.success("Generated Sweetviz report.")

# ---------------- Tabular Preprocessing ---------------- #
def preprocess_tabular(df: pd.DataFrame):
    st.header("🧹 Tabular Data Preprocessing")

    show_missing_viz(df)
    run_profilers(df)

    missing_method = st.selectbox("Missing Values", ["None", "Mean", "Median", "Mode"])
    scale_method = st.selectbox("Scaling", ["None", "Standard", "MinMax", "Robust"])
    encoding = st.selectbox("Categorical Encoding", ["None", "Label", "One-Hot"])

    if st.button("Apply Preprocessing"):
        df_proc = df.copy()

        # Missing values
        if missing_method != "None":
            numeric = df_proc.select_dtypes(include=['number'])
            if not numeric.empty:
                strategy = (
                    "mean" if missing_method == "Mean"
                    else "median" if missing_method == "Median"
                    else "most_frequent"
                )
                imputer = SimpleImputer(strategy=strategy)
                df_proc[numeric.columns] = imputer.fit_transform(numeric)

        # Encoding
        if encoding == "Label":
            le = LabelEncoder()
            for col in df_proc.select_dtypes(include="object"):
                try: df_proc[col] = le.fit_transform(df_proc[col].astype(str))
                except: pass
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

        allow_download_bytes(
            "Download Preprocessed CSV",
            df_proc.to_csv(index=False).encode(),
            "processed_data.csv"
        )

        st.session_state["df_processed"] = df_proc

# ---------------- Feature Engineering & Split (with auto target) ---------------- #
st.header("🧩 Feature Engineering & Train/Test Split")

df_processed = st.session_state.get("df_processed", None)

if df_processed is not None:
    data = df_processed

    default_target = guess_target_column(data)
    st.subheader("Target Column")
    target_col = st.selectbox("Select target (auto-detected shown first)",
                              options=list(data.columns),
                              index=list(data.columns).index(default_target))

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

        # ---- Polynomial Features ----
        if poly_feat:
            num_cols = X.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = poly.fit_transform(X[num_cols])
                X = pd.concat([
                    pd.DataFrame(poly_features, columns=poly.get_feature_names_out(num_cols), index=X.index),
                    X.drop(columns=num_cols)
                ], axis=1)

        # ---- PCA ----
        if pca_feat:
            numeric_data = X.select_dtypes(include=np.number)
            if numeric_data.shape[1] > 0:
                pca = PCA(n_components=min(5, numeric_data.shape[1]))
                pca_data = pca.fit_transform(numeric_data)
                pca_df = pd.DataFrame(pca_data, columns=[f"PCA{i+1}" for i in range(pca_data.shape[1])], index=X.index)
                X = pd.concat([pca_df, X.select_dtypes(exclude=np.number)], axis=1)

        # ---- TF-IDF per text col ----
        if text_embed:
            from sklearn.feature_extraction.text import TfidfVectorizer
            text_cols = X.select_dtypes(include='object').columns
            for col in text_cols:
                tfidf = TfidfVectorizer(max_features=2000)
                tfidf_matrix = tfidf.fit_transform(X[col].fillna(""))
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])], index=X.index)
                X = pd.concat([X.drop(columns=[col]), tfidf_df], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42,
                                                            stratify=y if (y.dtype=='O' or y.nunique()<=20) else None)

        Path("data/processed/").mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump((X_train, X_test, y_train, y_test), "data/processed/split_data.pkl")

        st.success("✅ Feature Engineering + Train/Test Split Completed")
        st.write("Training Data:", X_train.shape)
        st.write("Test Data:", X_test.shape)
        st.caption("Saved at: data/processed/split_data.pkl")

        # Save a fitted preprocessing-only pipeline (impute/scale/one-hot)
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
        # Fit preprocessor on train
        pre.fit(X_train)
        joblib.dump(pre, "data/processed/preprocessing.pkl")
        st.success("🧰 Saved preprocessing pipeline at data/processed/preprocessing.pkl")
        with open("data/processed/preprocessing.pkl","rb") as f:
            allow_download_bytes("Download preprocessing.pkl", f.read(), "preprocessing.pkl")

        st.session_state["split"] = (X_train, X_test, y_train, y_test, target_col)
else:
    st.info("ℹ️ Preprocess and set a dataframe first (Tabular section) to enable Feature Engineering.")

# ---------------- AutoML Trainer & Report ---------------- #
st.header("⚙️ AutoML (Tabular) – Leaderboard & Report")
if "split" not in st.session_state:
    st.info("Run Feature Engineering + Split above first.")
else:
    X_train, X_test, y_train, y_test, target_col = st.session_state["split"]
    is_classification = (y_train.dtype == "O") or (pd.Series(y_train).nunique() <= max(20, int(0.05*len(y_train))))
    st.write(f"Detected problem type: **{'Classification' if is_classification else 'Regression'}**")

    cv_folds = st.slider("CV folds", 3, 10, 5)
    run_automl = st.button("Run AutoML")

    if run_automl:
        # Build preprocessor
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

        models = {}
        results = []

        if is_classification:
            candidates = {
                "LogReg": LogisticRegression(max_iter=300),
                "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42),
            }
            if HAS_XGB:
                candidates["XGBoost"] = XGBClassifier(n_estimators=600, learning_rate=0.07,
                                                      subsample=0.9, colsample_bytree=0.9, random_state=42)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            metric_name = "F1_weighted"
            for name, mdl in candidates.items():
                pipe = Pipeline([("pre", pre), ("model", mdl)])
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted")
                results.append({"Model": name, "CV_mean_F1w": np.mean(cv_scores), "CV_std": np.std(cv_scores)})
                models[name] = pipe

            # Fit and evaluate on holdout
            report_rows = []
            for name, pipe in models.items():
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                acc = accuracy_score(y_test, preds)
                f1w = f1_score(y_test, preds, average="weighted")
                report_rows.append({"Model": name, "Holdout_Accuracy": acc, "Holdout_F1w": f1w})
            st.subheader("📈 CV Leaderboard")
            st.dataframe(pd.DataFrame(results).sort_values("CV_mean_F1w", ascending=False))
            st.subheader("🧾 Holdout Performance")
            hold_df = pd.DataFrame(report_rows).sort_values("Holdout_F1w", ascending=False)
            st.dataframe(hold_df)

            best_name = hold_df.iloc[0]["Model"]
            st.success(f"🏆 Best model: **{best_name}**")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fp:
                pickle.dump(models[best_name], fp)
                with open(fp.name, "rb") as f:
                    allow_download_bytes("Download Best AutoML Model (.pkl)", f.read(), f"automl_best_{best_name}.pkl")

            # Classification report
            preds = models[best_name].predict(X_test)
            st.text("Classification Report for best model:")
            st.text(classification_report(y_test, preds))

        else:
            candidates = {
                "Linear": LinearRegression(),
                "RandomForestReg": RandomForestRegressor(n_estimators=400, random_state=42),
            }
            if HAS_XGB:
                candidates["XGBReg"] = XGBRegressor(n_estimators=600, learning_rate=0.07,
                                                    subsample=0.9, colsample_bytree=0.9, random_state=42)
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            metric_name = "R2"
            for name, mdl in candidates.items():
                pipe = Pipeline([("pre", pre), ("model", mdl)])
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
                results.append({"Model": name, "CV_mean_R2": np.mean(cv_scores), "CV_std": np.std(cv_scores)})
                models[name] = pipe

            report_rows = []
            for name, pipe in models.items():
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                report_rows.append({"Model": name, "Holdout_R2": r2, "Holdout_MAE": mae})
            st.subheader("📈 CV Leaderboard")
            st.dataframe(pd.DataFrame(results).sort_values("CV_mean_R2", ascending=False))
            st.subheader("🧾 Holdout Performance")
            hold_df = pd.DataFrame(report_rows).sort_values("Holdout_R2", ascending=False)
            st.dataframe(hold_df)

            best_name = hold_df.iloc[0]["Model"]
            st.success(f"🏆 Best model: **{best_name}**")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as fp:
                pickle.dump(models[best_name], fp)
                with open(fp.name, "rb") as f:
                    allow_download_bytes("Download Best AutoML Model (.pkl)", f.read(), f"automl_best_{best_name}.pkl")

            # Quick scatter
            import matplotlib.pyplot as plt
            best = models[best_name]; preds = best.predict(X_test)
            fig = plt.figure()
            plt.scatter(y_test, preds, s=10)
            plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Regression: True vs Pred")
            st.pyplot(fig)

# ---------------- FastAPI Exporter ---------------- #
st.header("🚀 FastAPI Exporter")
st.caption("Writes a ready-to-run `fastapi_app.py` using your saved preprocessing pipeline and best model.")
preproc_path = st.text_input("Path to preprocessing.pkl", "data/processed/preprocessing.pkl")
model_path = st.text_input("Path to trained model .pkl", "automl_best_X.pkl (or your downloaded model)")

def write_fastapi_stub(preproc_path, model_path, outfile="fastapi_app.py"):
    template = f'''from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="DataMentor FastAPI")

with open(r"{preproc_path}", "rb") as f:
    pre = pickle.load(f)
with open(r"{model_path}", "rb") as f:
    model = pickle.load(f)

class Item(BaseModel):
    payload: dict  # single-row dict of features

@app.get("/")
def root():
    return {{"status": "ok", "msg": "DataMentor inference server"}}

@app.post("/predict")
def predict(item: Item):
    df = pd.DataFrame([item.payload])
    # If your model is a pipeline with preprocessor inside, you can just do model.predict(df)
    try:
        preds = model.predict(df)
    except Exception:
        X = pre.transform(df)
        preds = model.predict(X)
    return {{"prediction": preds.tolist()}}
'''
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(template)
    return outfile

if st.button("✍️ Write fastapi_app.py"):
    out = write_fastapi_stub(preproc_path, model_path if model_path.strip() else "best_model.pkl")
    with open(out, "rb") as f:
        allow_download_bytes("Download fastapi_app.py", f.read(), "fastapi_app.py")
    st.success("FastAPI stub created. Run:  uvicorn fastapi_app:app --reload")

# ---------------- Image/Video Auto Deep Train ---------------- #
st.header("🖼️/🎥 Auto Deep Training on Upload (optional)")
auto_train = st.checkbox("Auto-train deep model when image/video dataset ZIP is uploaded", value=False)

def train_image_ds(train_dir, val_dir=None, img_size=224, batch_size=32, epochs=5):
    if not HAS_TF: 
        st.error("TensorFlow required."); return None
    seed=123
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
    train_ds = train_ds.prefetch(AUTOTUNE); val_ds = val_ds.prefetch(AUTOTUNE)
    num_classes = len(train_ds.class_names)
    try:
        base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
        base.trainable = False
        inputs = keras.Input(shape=(img_size, img_size, 3))
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base(x, training=False); x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
    except Exception:
        model = keras.Sequential([
            layers.Input((img_size, img_size, 3)),
            layers.Conv2D(32,3,activation="relu"), layers.MaxPooling2D(),
            layers.Conv2D(64,3,activation="relu"), layers.MaxPooling2D(),
            layers.Conv2D(128,3,activation="relu"), layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax"),
        ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    st.success("✅ Deep training complete")
    st.write("Classes:", train_ds.class_names)
    st.line_chart(pd.DataFrame({"train_acc": hist.history["accuracy"], "val_acc": hist.history["val_accuracy"]}))
    st.line_chart(pd.DataFrame({"train_loss": hist.history["loss"], "val_loss": hist.history["val_loss"]}))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as fp:
        model.save(fp.name)
        with open(fp.name, "rb") as f:
            allow_download_bytes("Download Keras Model (.h5)", f.read(), "image_model.h5")

def extract_frames_dir(src_dir, out_dir, img_size=224, fps_target=1):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for cls in os.listdir(src_dir):
        src_cls = os.path.join(src_dir, cls)
        if not os.path.isdir(src_cls): continue
        dst_cls = os.path.join(out_dir, cls); Path(dst_cls).mkdir(exist_ok=True)
        for vid_name in os.listdir(src_cls):
            if not vid_name.lower().endswith((".mp4",".avi",".mov",".mkv")): continue
            cap = cv2.VideoCapture(os.path.join(src_cls, vid_name))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
            idx=0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if idx % max(1, fps // fps_target) == 0:
                    fr = cv2.resize(frame[:, :, ::-1], (img_size, img_size))  # BGR->RGB
                    out_name = f"{Path(vid_name).stem}_{idx}.jpg"
                    cv2.imwrite(os.path.join(dst_cls, out_name), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                idx += 1
            cap.release()
    return out_dir

st.subheader("Upload Image/Video Dataset ZIP")
zipf = st.file_uploader("ZIP with train/<class>/... and optional val/<class>/...", type=["zip"])
epochs = st.slider("Epochs", 1, 20, 5, key="img_epochs")
img_size = st.selectbox("Image size", [128,160,192,224], index=3, key="img_size")
batch_size = st.selectbox("Batch size", [8,16,32,64], index=2, key="img_bs")
fps_extract = st.selectbox("Video extract FPS", [1,2,4], index=0, key="vidfps")

if zipf is not None and auto_train:
    if not HAS_TF:
        st.error("TensorFlow not available.")
    else:
        with tempfile.TemporaryDirectory() as tmpd:
            zpath = os.path.join(tmpd, "data.zip")
            with open(zpath, "wb") as f: f.write(zipf.read())
            with zipfile.ZipFile(zpath) as z: z.extractall(tmpd)

            # Try image dataset first
            train_dir = os.path.join(tmpd, "train")
            val_dir = os.path.join(tmpd, "val")
            if os.path.isdir(train_dir) and any(os.listdir(train_dir)):
                # If videos instead of images, convert to frames
                possible_vid = False
                for cls in os.listdir(train_dir):
                    cdir = os.path.join(train_dir, cls)
                    if os.path.isdir(cdir):
                        for fn in os.listdir(cdir):
                            if fn.lower().endswith((".mp4",".avi",".mov",".mkv")):
                                possible_vid = True; break
                if possible_vid and HAS_CV:
                    frames_train = extract_frames_dir(train_dir, os.path.join(tmpd,"frames_train"),
                                                      img_size=img_size, fps_target=fps_extract)
                    frames_val = None
                    if os.path.isdir(val_dir):
                        frames_val = extract_frames_dir(val_dir, os.path.join(tmpd,"frames_val"),
                                                        img_size=img_size, fps_target=fps_extract)
                    train_image_ds(frames_train, frames_val, img_size=img_size, batch_size=batch_size, epochs=epochs)
                else:
                    train_image_ds(train_dir, val_dir if os.path.isdir(val_dir) else None,
                                   img_size=img_size, batch_size=batch_size, epochs=epochs)
            else:
                st.error("ZIP must contain train/<class>/... (images or videos).")

# ---------------- Upload & basic preprocess (top entry) ---------------- #
st.markdown("---")
st.subheader("📎 Upload File to Preprocess")
file = st.file_uploader(
    "Supported: CSV/XLS/XLSX/TXT/PNG/JPG/JPEG/MP4/AVI/WAV/MP3/FLAC/OGG/M4A",
    type=["csv","xls","xlsx","txt","png","jpg","jpeg","mp4","avi","wav","mp3","flac","ogg","m4a"]
)

# Basic previews and route to preprocessors
if file:
    ext = Path(file.name).suffix.lower()
    if ext in [".csv", ".xls", ".xlsx"]:
        df = load_file(file)
        basic_df_profile(df)
        preprocess_tabular(df)

    elif ext == ".txt":
        text = load_file(file)
        st.text_area("📄 Original Text", text, height=200)
        st.info("Text cleaning/embeddings available in your previous version; add back if needed.")

    elif ext in [".png",".jpg",".jpeg"]:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image")
        st.info("Use the Auto Deep Training section to train on datasets (ZIP).")

    elif ext in [".mp4",".avi"]:
        st.video(file)
        st.info("Use the Auto Deep Training section to train on video datasets (ZIP).")

    elif ext in [".wav",".mp3",".flac",".ogg",".m4a"]:
        if not HAS_AUDIO:
            st.error("Install: pip install librosa soundfile")
        else:
            st.success("Audio file loaded. Add your audio preprocessing block here if required.")
