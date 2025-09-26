import io
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Image Lab — Thresholds • Watermark • e-Signature",
                   layout="wide")

# ---------- helpers ----------
def read_rgb(upload):
    if upload is None:
        h, w = 320, 480
        grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        return cv2.merge([grad, grad, grad])
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb

def to_rgba(rgb):
    a = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, a], axis=-1)

def paste_rgba_on_rgb(base_rgb, overlay_rgba, x, y, opacity=1.0):
    base = Image.fromarray(base_rgb).convert("RGBA")
    ov = Image.fromarray(overlay_rgba).copy()
    if opacity < 1.0:
        a = ov.split()[-1].point(lambda v: int(v * float(opacity)))
        ov.putalpha(a)
    base.alpha_composite(ov, dest=(int(x), int(y)))
    return np.array(base.convert("RGB"))

def pil_download(arr, name="result.png"):
    im = Image.fromarray(arr if arr.ndim==3 else arr)
    buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0); return buf

# ---------- sidebar ----------
st.sidebar.title("Image Lab")
page = st.sidebar.radio("Choose a module", [
    "Step 1 — Thresholds",
    "Step 2 — Watermark",
    "Step 3 — e-Signature",
])

# ---------- Step 1: Thresholds ----------
if page == "Step 1 — Thresholds":
    st.title("Step 1 — Thresholding")
    up = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
    rgb = read_rgb(up); gray = to_gray(rgb)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original"); st.image(rgb, use_container_width=True)
        st.subheader("Grayscale"); st.image(gray, use_container_width=True)
    with c2:
        st.subheader("Controls")
        method = st.selectbox("Method", ["Global (manual T)", "Otsu (auto)", "Adaptive"], index=1)
        invert = st.checkbox("Invert", value=False)
        if method == "Global (manual T)":
            T = st.slider("Global T", 0, 255, 127)
            typ = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, out = cv2.threshold(gray, T, 255, typ)
        elif method == "Otsu (auto)":
            typ = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, out = cv2.threshold(gray, 0, 255, typ + cv2.THRESH_OTSU)
        else:
            m = st.radio("Adaptive method", ["Gaussian","Mean"], horizontal=True)
            block = st.slider("Block Size (odd)", 3, 99, 31, 2)
            C = st.slider("C", -20, 20, 2)
            meth = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if m=="Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
            out = cv2.adaptiveThreshold(gray, 255, meth,
                                        cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                                        block, C)
        st.image(out, caption="Thresholded Output", use_container_width=True)
        st.download_button("Download PNG", data=pil_download(out, "threshold_result.png"),
                           file_name="threshold_result.png", mime="image/png")

# ---------- Step 2: Watermark ----------
elif page == "Step 2 — Watermark":
    st.title("Step 2 — Watermark")
    base_file = st.file_uploader("Upload base image (JPG/PNG)", type=["jpg","jpeg","png"])
    base_rgb = read_rgb(base_file); H, W = base_rgb.shape[:2]
    st.subheader("Original"); st.image(base_rgb, use_container_width=True)

    wm_type = st.radio("Watermark type", ["Text", "Image"], horizontal=True)
    pos = st.selectbox("Position", ["Center","Top-Left","Top-Right","Bottom-Left","Bottom-Right"])
    pad = st.slider("Padding (px)", 0, max(20, min(W,H)//5), 20)
    opacity = st.slider("Opacity", 0.0, 1.0, 0.35, 0.01)

    comp = base_rgb.copy()
    if wm_type == "Text":
        text = st.text_input("Watermark text", "© Your Name")
        scale = st.slider("Text scale", 0.02, 0.25, 0.08, 0.01)
        canvas = Image.new("RGBA", (W, H), (0,0,0,0))
        draw = ImageDraw.Draw(canvas)
        size = max(12, int(min(W,H)*scale))
        try: font = ImageFont.truetype("DejaVuSans.ttf", size)
        except: font = ImageFont.load_default()
        bbox = draw.textbbox((0,0), text, font=font)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        x0, y0 = (W-tw)//2, (H-th)//2
        draw.text((x0,y0), text, fill=(255,255,255,255), font=font)
        overlay = np.array(canvas)
        ys, xs = np.where(overlay[...,3]>0)
        if len(xs): overlay = overlay[ys.min():ys.max()+1, xs.min():xs.max()+1]
        ow, oh = overlay.shape[1], overlay.shape[0]
        def posxy():
            if   pos=="Top-Left": return pad, pad
            elif pos=="Top-Right": return W-ow-pad, pad
            elif pos=="Bottom-Left": return pad, H-oh-pad
            elif pos=="Bottom-Right": return W-ow-pad, H-oh-pad
            else: return (W-ow)//2, (H-oh)//2
        x,y = posxy()
        comp = paste_rgba_on_rgb(base_rgb, overlay, x, y, opacity=opacity)
    else:
        wm_file = st.file_uploader("Upload watermark image (PNG recommended)", type=["png","jpg","jpeg"])
        scale_pct = st.slider("Watermark width (% of base width)", 5, 50, 22, 1)
        if wm_file:
            wm_rgb = read_rgb(wm_file)
            wm_rgba = to_rgba(wm_rgb)
            target_w = int(W*(scale_pct/100.0))
            ratio = target_w / wm_rgba.shape[1]
            target_h = max(1, int(wm_rgba.shape[0]*ratio))
            wm_rgba = cv2.resize(wm_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
            ow, oh = wm_rgba.shape[1], wm_rgba.shape[0]
            def posxy():
                if   pos=="Top-Left": return pad, pad
                elif pos=="Top-Right": return W-ow-pad, pad
                elif pos=="Bottom-Left": return pad, H-oh-pad
                elif pos=="Bottom-Right": return W-ow-pad, H-oh-pad
                else: return (W-ow)//2, (H-oh)//2
            x,y = posxy()
            comp = paste_rgba_on_rgb(base_rgb, wm_rgba, x, y, opacity=opacity)

    st.subheader("Preview"); st.image(comp, use_container_width=True)
    st.download_button("Download watermarked PNG", data=pil_download(comp, "watermarked.png"),
                       file_name="watermarked.png", mime="image/png")

# ---------- Step 3: e-Signature ----------
elif page == "Step 3 — e-Signature":
    st.title("Step 3 — e-Signature")

    st.markdown("**1) Draw your signature (transparent background)**")
    pen_color = st.color_picker("Pen color", "#000000")
    pen_w = st.slider("Pen width", 1, 20, 4)
    canvas_w = st.slider("Canvas width", 300, 1200, 600, 50)
    canvas_h = st.slider("Canvas height", 150, 600, 240, 10)

    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=pen_w,
        stroke_color=pen_color,
        background_color="rgba(0,0,0,0)",
        height=canvas_h, width=canvas_w,
        drawing_mode="freedraw",
        key="sig_canvas",
    )

    sig_rgba = None
    if canvas.image_data is not None:
        sig_rgba = canvas.image_data.astype("uint8")
        st.caption("Signature preview"); st.image(sig_rgba, use_container_width=True)
        buf = io.BytesIO(); Image.fromarray(sig_rgba, "RGBA").save(buf, format="PNG"); buf.seek(0)
        st.download_button("Download signature (PNG)", buf, "signature.png", "image/png")

    st.markdown("---")
    st.markdown("**2) Place signature on an image/document**")
    base_file = st.file_uploader("Upload base image (JPG/PNG)", type=["jpg","jpeg","png"])
    if base_file:
        base_rgb = read_rgb(base_file); H, W = base_rgb.shape[:2]
        st.image(base_rgb, caption="Base image", use_container_width=True)
        up_sig = st.file_uploader("Or upload an existing signature PNG", type=["png"], key="sig_upload")
        if up_sig is not None:
            sig_rgba = np.array(Image.open(up_sig).convert("RGBA"))

        if sig_rgba is not None:
            scale = st.slider("Signature width (% of base width)", 5, 60, 25, 1)
            opacity = st.slider("Opacity", 0.1, 1.0, 0.9, 0.05)
            pos = st.selectbox("Position", ["Bottom-Right","Bottom-Left","Top-Right","Top-Left","Center"])
            pad = st.slider("Padding (px)", 0, max(20, min(W,H)//5), 20)

            target_w = int(W*(scale/100.0))
            ratio = target_w / sig_rgba.shape[1]
            target_h = max(1, int(sig_rgba.shape[0]*ratio))
            sig_resized = cv2.resize(sig_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

            sh, sw = sig_resized.shape[0], sig_resized.shape[1]
            if   pos=="Bottom-Right": x,y = W-sw-pad, H-sh-pad
            elif pos=="Bottom-Left":  x,y = pad, H-sh-pad
            elif pos=="Top-Right":    x,y = W-sw-pad, pad
            elif pos=="Top-Left":     x,y = pad, pad
            else:                     x,y = (W-sw)//2, (H-sh)//2

            composed = paste_rgba_on_rgb(base_rgb, sig_resized, x, y, opacity=opacity)
            st.image(composed, caption="Signed image", use_container_width=True)
            st.download_button("Download signed image (PNG)",
                               data=pil_download(composed, "signed_output.png"),
                               file_name="signed_output.png", mime="image/png")
        else:
            st.info("Draw a signature above or upload a signature PNG to place it.")
    else:
        st.info("Upload a base image/document to place your signature on.")
