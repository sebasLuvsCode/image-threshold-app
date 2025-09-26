import io
import numpy as np
import streamlit as st
from PIL import Image
import cv2

st.set_page_config(page_title="Image Lab — Thresholds (Step 1)", layout="wide")

def read_image(upload):
    if upload is None:
        h, w = 320, 480
        grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        demo = cv2.merge([grad, grad, grad])
        return demo
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb

def threshold_global(gray, t=127, invert=False):
    typ = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, t, 255, typ)
    return out

def threshold_otsu(gray, invert=False):
    typ = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, 0, 255, typ + cv2.THRESH_OTSU)
    return out

def threshold_adaptive(gray, method="Gaussian", block_size=31, C=2, invert=False):
    if block_size % 2 == 0: block_size += 1
    meth = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    out = cv2.adaptiveThreshold(gray, 255, meth,
                                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                                block_size, C)
    return out

def pil_download(arr):
    im = Image.fromarray(arr if arr.ndim==2 else arr)
    buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0); return buf

st.sidebar.title("Image Lab")
page = st.sidebar.radio("Choose a module",
    ["Step 1 — Thresholds", "Step 2 — Watermark (next)", "Step 3 — e-Signature (next)"])

if page == "Step 1 — Thresholds":
    st.title("Step 1 — Thresholding")
    upload = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
    rgb = read_image(upload); gray = to_gray(rgb)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original"); st.image(rgb, use_column_width=True)
        st.subheader("Grayscale"); st.image(gray, use_column_width=True)

    with c2:
        st.subheader("Controls")
        method = st.selectbox("Method", ["Global (manual T)","Otsu (auto T)","Adaptive"])
        invert = st.checkbox("Invert", value=False)
        if method == "Global (manual T)":
            t = st.slider("Global T", 0, 255, 127, 1)
            out = threshold_global(gray, t, invert)
        elif method == "Otsu (auto T)":
            out = threshold_otsu(gray, invert)
        else:
            m = st.radio("Adaptive Method", ["Gaussian","Mean"], horizontal=True)
            block = st.slider("Block Size (odd)", 3, 99, 31, 2)
            C = st.slider("C", -20, 20, 2, 1)
            out = threshold_adaptive(gray, m, block, C, invert)
        st.image(out, caption="Thresholded Output", use_column_width=True)
        st.download_button("Download PNG", data=pil_download(out),
                           file_name="threshold_result.png", mime="image/png")

else:
    st.title(page); st.write("Coming soon.")