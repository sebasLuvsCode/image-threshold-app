import io
import numpy as np
import streamlit as st
from PIL import Image
import cv2

# Page setup
st.set_page_config(page_title="Image Lab – Thresholds & Watermark", layout="wide")

# ---------- Utility functions ----------
def read_image(upload):
    if upload is None:
        h, w = 320, 480
        grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        demo = cv2.merge([grad, grad, grad])
        return demo
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def apply_threshold(img, method, invert, t_value):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if method == "Global (manual T)":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, result = cv2.threshold(gray, t_value, 255, thresh_type)
    elif method == "Otsu (auto)":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, result = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
    else:
        result = gray
    return result

def add_watermark(img, text):
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        overlay, text, (50, img.shape[0] - 50),
        font, 2, (255, 255, 255), 4, cv2.LINE_AA
    )
    return overlay

# ---------- Sidebar Navigation ----------
st.sidebar.title("Image Lab")
page = st.sidebar.radio("Choose a module", ["Step 1 — Thresholds", "Step 2 — Watermark"])

# ---------- Step 1: Thresholding ----------
if page == "Step 1 — Thresholds":
    st.header("Step 1 — Thresholding")
    upload = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    img = read_image(upload)

    st.subheader("Original")
    st.image(img, caption="Original Image", use_container_width=True)

    st.subheader("Controls")
    method = st.selectbox("Method", ["Global (manual T)", "Otsu (auto)"])
    invert = st.checkbox("Invert", value=False)
    t_value = st.slider("Global T", 0, 255, 127)

    result = apply_threshold(img, method, invert, t_value)

    st.subheader("Thresholded Result")
    st.image(result, caption=f"Thresholded ({method})", use_container_width=True)

# ---------- Step 2: Watermark ----------
elif page == "Step 2 — Watermark":
    st.header("Step 2 — Watermark")
    upload = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    img = read_image(upload)

    watermark_text = st.text_input("Enter watermark text:", "My Watermark")

    if st.button("Apply Watermark"):
        result = add_watermark(img, watermark_text)
        st.image(result, caption="Image with Watermark", use_container_width=True)

# ---------- Step 3: e-Signature ----------
elif page == "Step 3 — e-Signature":
    st.header("Step 3 — e-Signature")
    st.write("Draw your signature below:")

    # Create drawing canvas
    from streamlit_drawable_canvas import st_canvas

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Transparent background
        stroke_width=3,
        stroke_color="black",
        background_color="white",
        update_streamlit=True,
        height=200,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data, caption="Your e-Signature")
