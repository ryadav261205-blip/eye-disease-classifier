"""
Streamlit Web Application:
AI-Based Eye Disease Classification Using Convolutional Neural Network (CNN).

Educational classroom tool for classifying retinal fundus images into:
- Normal (Class 0)
- Diabetic Retinopathy (Class 1)
"""

import os
import json
from pathlib import Path
from PIL import Image
import streamlit as st

# Environment overrides: disable telemetry and set writable cache paths
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from config import (
    MODEL_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    TRAINING_HISTORY_PATH,
    ARCHITECTURE_PATH,
    SAMPLE_PREDICTIONS_PATH,
    SAMPLE_IMAGES_DIR,
    CLASS_NAMES,
)
from preprocessing import preprocess_image, PreprocessingError
from model import load_trained_model, predict_disease


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Eye Disease Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Load model once and cache it in memory."""
    if not MODEL_PATH.exists():
        return None
    try:
        return load_trained_model(MODEL_PATH)
    except Exception as err:
        st.error(f"Error loading model: {err}")
        return None


# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None


# -----------------------------------------------------------------------------
# Sidebar Navigation & Model Info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 👁️ Eye Disease AI")
    st.caption("Convolutional Neural Network Classifier")
    st.divider()

    st.markdown("### Model Information")
    st.markdown(
        """
        - **Architecture:** 2D CNN (Conv2D + MaxPool + Dense + Dropout)
        - **Benchmark:** MedMNIST RetinaMNIST
        - **Resolution:** 28 × 28 × 3 RGB
        - **Target:** Normal vs. Diabetic Retinopathy (DR)
        - **Execution:** CPU-Compatible Lightweight Inference
        """
    )
    st.divider()

    if MODEL_PATH.exists():
        st.success("✓ Model Ready (`eye_disease_model.keras`)")
    else:
        st.error("✗ Model Missing. Run `python train.py` first.")

    st.divider()
    st.caption("College Computer Science & AI Project")


# -----------------------------------------------------------------------------
# Header Block
# -----------------------------------------------------------------------------
st.title("AI-Based Eye Disease Classification Using CNN")
st.markdown(
    "Automated deep learning tool for binary screening of retinal fundus photographs "
    "into **Normal** and **Diabetic Retinopathy (DR)**."
)

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab_predict, tab_metrics, tab_arch, tab_about = st.tabs([
    "🔍 Live Image Classification",
    "📊 Model Evaluation & Metrics",
    "📐 System Architecture",
    "📖 Documentation & Instructions",
])


# =============================================================================
# TAB 1: Live Image Classification
# =============================================================================
with tab_predict:
    st.subheader("Retinal Fundus Image Analysis")

    model = get_model()
    if model is None:
        st.error(
            "⚠️ **Trained model file not found (`models/eye_disease_model.keras`).**\n\n"
            "Please train the model first by running:\n\n"
            "```bash\n"
            "python train.py\n"
            "```"
        )
    else:
        col_input, col_result = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("#### 1. Select or Upload Image")
            input_source = st.radio(
                "Choose input source:",
                ["Use Pre-loaded Benchmark Sample", "Upload Custom Image (JPG/PNG)"],
                horizontal=True,
            )

            selected_image = None
            image_name = ""

            if input_source == "Use Pre-loaded Benchmark Sample":
                normal_samples = sorted(list((SAMPLE_IMAGES_DIR / "normal").glob("*.png"))) if (SAMPLE_IMAGES_DIR / "normal").exists() else []
                dr_samples = sorted(list((SAMPLE_IMAGES_DIR / "diabetic_retinopathy").glob("*.png"))) if (SAMPLE_IMAGES_DIR / "diabetic_retinopathy").exists() else []

                sample_options = {}
                for p in normal_samples:
                    sample_options[f"Normal Retina (Sample {p.stem.split('_')[-1]})"] = p
                for p in dr_samples:
                    sample_options[f"Diabetic Retinopathy (Sample {p.stem.split('_')[-1]})"] = p

                if sample_options:
                    choice = st.selectbox("Select a benchmark image:", list(sample_options.keys()))
                    sample_path = sample_options[choice]
                    selected_image = Image.open(sample_path)
                    image_name = sample_path.name
                else:
                    st.info("Sample benchmark images not found.")

            else:
                uploaded_file = st.file_uploader(
                    "Upload a fundus image (JPG, JPEG, PNG):",
                    type=["jpg", "jpeg", "png"],
                )
                if uploaded_file is not None:
                    try:
                        selected_image = Image.open(uploaded_file)
                        image_name = uploaded_file.name
                    except Exception as err:
                        st.error(f"Failed to decode image file: {err}")

            if selected_image is not None:
                # Update session state image if changed
                if st.session_state.current_image_name != image_name:
                    st.session_state.current_image_name = image_name
                    st.session_state.current_image = selected_image
                    st.session_state.prediction_result = None

                st.write(f"**Loaded:** `{image_name}` (Dimensions: {selected_image.size[0]}×{selected_image.size[1]} px)")
                st.image(selected_image, caption="Loaded Fundus Photograph", width=260)

                classify_clicked = st.button("🚀 Classify Retinal Image", type="primary")
                if classify_clicked:
                    with st.spinner("Processing image and running CNN inference..."):
                        try:
                            res = predict_disease(model, selected_image)
                            st.session_state.prediction_result = res
                        except PreprocessingError as p_err:
                            st.error(f"Preprocessing error: {p_err}")
                        except Exception as err:
                            st.error(f"Inference error: {err}")

        with col_result:
            st.markdown("#### 2. Classification Results & Analysis")

            if st.session_state.prediction_result is not None:
                res = st.session_state.prediction_result
                is_dr = res["class_id"] == 1
                prob = res["probability"]
                conf = res["confidence"] * 100.0

                if is_dr:
                    st.error(f"### ⚠️ Diabetic Retinopathy Detected\n**Predicted Class:** {res['class_name']} (Class 1)")
                else:
                    st.success(f"### ✓ Normal Retinal Appearance\n**Predicted Class:** {res['class_name']} (Class 0)")

                st.metric("Model Confidence", f"{conf:.2f}%")

                st.markdown("##### Sigmoid Probability Output")
                clamped_prob = min(max(float(prob), 0.0), 1.0)
                st.progress(clamped_prob, text=f"Sigmoid Score: {prob:.4f}")

                st.info(f"**Explanation:** {res['explanation']}")

                st.markdown("---")
                st.markdown("##### Preprocessed CNN Input (28×28 Normalized)")
                preproc_arr = preprocess_image(st.session_state.current_image)
                st.image(preproc_arr, caption="28x28 RGB Tensor", width=90)

            elif selected_image is not None:
                st.info("Click the **'Classify Retinal Image'** button on the left to run prediction.")
            else:
                st.write("Please select or upload a retinal image to start.")

    # Educational Disclaimer
    st.warning(
        "**⚠️ Mandatory Educational Disclaimer:**\n\n"
        "This application is an educational computer vision project built for academic demonstration. "
        "It is **NOT** a certified diagnostic medical tool. The predictions and sigmoid confidence "
        "scores are generated by an experimental lightweight CNN model trained on 28×28 pixel images "
        "and must **never** be used for clinical decisions, medical diagnosis, or treatment recommendations."
    )


# =============================================================================
# TAB 2: Model Evaluation & Metrics
# =============================================================================
with tab_metrics:
    st.subheader("Model Evaluation on Untouched Test Split (400 Samples)")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

        m = metrics_data.get("metrics", {})
        cm_data = metrics_data.get("confusion_matrix", {})
        tp = cm_data.get("true_positive", 0)
        tn = cm_data.get("true_negative", 0)
        fp = cm_data.get("false_positive", 0)
        fn = cm_data.get("false_negative", 0)
        total_test = metrics_data.get("test_samples", 400)

        # Built-in Streamlit metric cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m.get('accuracy', 0.0)*100:.2f}%")
        col2.metric("Precision", f"{m.get('precision', 0.0)*100:.2f}%")
        col3.metric("Recall (Sensitivity)", f"{m.get('recall', 0.0)*100:.2f}%")
        col4.metric("F1 Score", f"{m.get('f1_score', 0.0)*100:.2f}%")

        st.divider()

        col_cm, col_math = st.columns([1, 1], gap="medium")

        with col_cm:
            st.markdown("#### Confusion Matrix Heatmap")
            if CONFUSION_MATRIX_PATH.exists():
                st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix on RetinaMNIST Test Split")
            else:
                st.info("Run `python evaluate.py` to generate the confusion matrix plot.")

        with col_math:
            st.markdown("#### Confusion Matrix Counts")
            st.markdown(
                f"""
                - **Total Untouched Test Images:** `{total_test}`
                - **True Positives (TP):** `{tp}` *(Diabetic Retinopathy correctly identified)*
                - **True Negatives (TN):** `{tn}` *(Normal retinas correctly identified)*
                - **False Positives (FP):** `{fp}` *(Normal incorrectly flagged as DR)*
                - **False Negatives (FN):** `{fn}` *(DR missed and labeled as Normal)*

                #### Metric Formulas
                - **Accuracy:** $(TP + TN) / Total = ({tp} + {tn}) / {total_test} = {m.get('accuracy', 0.0)*100:.2f}\\%$
                - **Precision:** $TP / (TP + FP) = {tp} / ({tp} + {fp}) = {m.get('precision', 0.0)*100:.2f}\\%$
                - **Recall:** $TP / (TP + FN) = {tp} / ({tp} + {fn}) = {m.get('recall', 0.0)*100:.2f}\\%$
                - **F1 Score:** $2 \\times \\frac{{\\text{{Precision}} \\times \\text{{Recall}}}}{{\\text{{Precision}} + \\text{{Recall}}}} = {m.get('f1_score', 0.0)*100:.2f}\\%$
                """
            )

        st.divider()

        col_curves, col_preds = st.columns([1, 1], gap="medium")
        with col_curves:
            st.markdown("#### Training & Validation History")
            if TRAINING_HISTORY_PATH.exists():
                st.image(str(TRAINING_HISTORY_PATH), caption="Loss and Accuracy Curves Across Epochs")
            else:
                st.info("Run `python train.py` to generate training history curves.")

        with col_preds:
            st.markdown("#### Sample Test Set Predictions")
            if SAMPLE_PREDICTIONS_PATH.exists():
                st.image(str(SAMPLE_PREDICTIONS_PATH), caption="Test Samples with Ground Truth vs. Predicted Class")
            else:
                st.info("Run `python evaluate.py` to generate sample prediction thumbnails.")

    else:
        st.warning("Metrics not found. Run `python evaluate.py` to generate them.")


# =============================================================================
# TAB 3: System Architecture
# =============================================================================
with tab_arch:
    st.subheader("System Architecture & Processing Pipeline")

    if ARCHITECTURE_PATH.exists():
        st.image(str(ARCHITECTURE_PATH), caption="End-to-End System Architecture Flowchart")
    else:
        st.info("Run `python generate_architecture_diagram.py` to generate the diagram.")

    st.divider()
    st.markdown(
        """
        ### CNN Layer Specifications
        1. **Input Layer:** Receives $28 \\times 28 \\times 3$ RGB retinal image tensors.
        2. **Conv2D (16 filters):** $3\\times 3$ kernel, ReLU activation, same padding. Detects low-level edge and vascular boundaries.
        3. **MaxPooling2D (Pool 1):** $2\\times 2$ downsampling to $14 \\times 14 \\times 16$.
        4. **Conv2D (32 filters):** $3\\times 3$ kernel, ReLU activation, same padding. Extracts lesion textures and exudate patterns.
        5. **MaxPooling2D (Pool 2):** $2\\times 2$ downsampling to $7 \\times 7 \\times 32$.
        6. **Flatten Layer:** Unrolls $7 \\times 7 \\times 32 = 1,568$ features into a 1D vector.
        7. **Dense Layer (64 units):** Non-linear feature combination with ReLU activation.
        8. **Dropout Layer (30%):** Regularization layer deactivating 30% of neurons during training to prevent overfitting.
        9. **Output Layer (1 unit):** Single neuron with Sigmoid activation producing a scalar probability in $[0.0, 1.0]$.
        """
    )


# =============================================================================
# TAB 4: Instructions & Documentation
# =============================================================================
with tab_about:
    st.subheader("How to Run & Reproduce")
    st.markdown(
        """
        ### Terminal Commands
        ```bash
        # 1. Activate virtual environment
        source .venv/bin/activate

        # 2. Run automated tests
        pytest -v

        # 3. Train the model
        python train.py

        # 4. Evaluate on untouched test split
        python evaluate.py

        # 5. Start this web application
        streamlit run app.py
        ```

        ### Dataset Summary
        - **Source:** MedMNIST v3.0 (RetinaMNIST)
        - **Training Set:** 1,080 images (486 Normal, 594 DR)
        - **Validation Set:** 120 images (54 Normal, 66 DR)
        - **Test Set:** 400 images (174 Normal, 226 DR)
        """
    )
