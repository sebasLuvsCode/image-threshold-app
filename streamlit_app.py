# streamlit_app.py
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
from streamlit_drawable_canvas import st_canvas

# -------------------- Page config --------------------
st.set_page_config(
    page_title="Image Lab — Thresholds • Watermark • e-Signature • Logical Ops • Alpha",
    layout="wide",
)

# -------------------- Helpers --------------------
def read_rgb(upload):
    """Read an uploaded image as RGB numpy array. If None, return a gradient demo."""
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
    """Alpha-composite overlay_rgba onto base_rgb at (x,y) with optional global opacity."""
    base = Image.fromarray(base_rgb).convert("RGBA")
    ov = Image.fromarray(overlay_rgba).copy()
    if opacity < 1.0:
        a = ov.split()[-1].point(lambda v: int(v * float(opacity)))
        ov.putalpha(a)
    base.alpha_composite(ov, dest=(int(x), int(y)))
    return np.array(base.convert("RGB"))

def pil_download(arr, name="result.png"):
    """Return a BytesIO PNG for st.download_button."""
    im = Image.fromarray(arr if arr.ndim == 3 else arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf

def checkerboard(h, w, tile=16):
    """Create a gray checkerboard (for alpha previews)."""
    c1, c2 = 230, 200
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((yy // tile) + (xx // tile)) % 2) * (c1 - c2) + c2
    return np.stack([board, board, board], axis=-1).astype(np.uint8)

def preview_rgba(rgba_img):
    """Composite RGBA over a checkerboard for visualizing transparency."""
    h, w = rgba_img.shape[:2]
    bg = checkerboard(h, w, tile=20).astype(np.float32)
    fg = rgba_img[..., :3].astype(np.float32)
    a  = (rgba_img[..., 3:4].astype(np.float32)) / 255.0
    comp = (fg * a + bg * (1 - a)).astype(np.uint8)
    return comp

# -------------------- Sidebar (robust mapping) --------------------
st.sidebar.title("Image Lab")
pages = {
    "Step 1 — Thresholds": "thresh",
    "Step 2 — Watermark": "wm",
    "Step 3 — e-Signature": "sign",
    "Step 4 — Logical Operations": "logic",
    "Step 2-4 — Alpha / Transparency": "alpha",
}
label = st.sidebar.radio("Choose a module", list(pages.keys()))
page = pages[label]

# ==================== Step 1: Thresholds ====================
if page == "thresh":
    st.title("Step 1 — Thresholding")

    up = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    rgb = read_rgb(up)
    gray = to_gray(rgb)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(rgb, use_container_width=True)
        st.subheader("Grayscale")
        st.image(gray, use_container_width=True)

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
            choose = st.radio("Adaptive method", ["Gaussian", "Mean"], horizontal=True)
            block = st.slider("Block size (odd)", 3, 99, 31, step=2)
            C = st.slider("C", -20, 20, 2)
            meth = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if choose == "Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
            out = cv2.adaptiveThreshold(
                gray, 255, meth,
                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                block, C
            )

        st.subheader("Thresholded Output")
        st.image(out, use_container_width=True)
        st.download_button(
            "Download PNG",
            data=pil_download(out, "threshold_result.png"),
            file_name="threshold_result.png",
            mime="image/png",
        )

# ==================== Step 2: Watermark ====================
elif page == "wm":
    st.title("Step 2 — Watermark")

    base_file = st.file_uploader("Upload base image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    base_rgb = read_rgb(base_file)
    H, W = base_rgb.shape[:2]

    st.subheader("Original")
    st.image(base_rgb, use_container_width=True)

    wm_type = st.radio("Watermark type", ["Text", "Image"], horizontal=True)
    pos = st.selectbox("Position", ["Center", "Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"])
    pad = st.slider("Padding (px)", 0, max(20, min(W, H) // 5), 20)
    opacity = st.slider("Opacity", 0.0, 1.0, 0.35, 0.01)

    comp = base_rgb.copy()

    if wm_type == "Text":
        text = st.text_input("Watermark text", "© Your Name")
        scale = st.slider("Text scale", 0.02, 0.25, 0.08, 0.01)

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        size = max(12, int(min(W, H) * scale))
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x0, y0 = (W - tw) // 2, (H - th) // 2
        draw.text((x0, y0), text, fill=(255, 255, 255, 255), font=font)

        overlay = np.array(canvas)
        ys, xs = np.where(overlay[..., 3] > 0)
        if len(xs):
            overlay = overlay[ys.min():ys.max()+1, xs.min():xs.max()+1]
        ow, oh = overlay.shape[1], overlay.shape[0]

        def posxy():
            if pos == "Top-Left": return pad, pad
            if pos == "Top-Right": return W - ow - pad, pad
            if pos == "Bottom-Left": return pad, H - oh - pad
            if pos == "Bottom-Right": return W - ow - pad, H - oh - pad
            return (W - ow) // 2, (H - oh) // 2

        x, y = posxy()
        comp = paste_rgba_on_rgb(base_rgb, overlay, x, y, opacity=opacity)

    else:
        wm_file = st.file_uploader("Upload watermark image (PNG recommended)", type=["png", "jpg", "jpeg"])
        scale_pct = st.slider("Watermark width (% of base width)", 5, 50, 22, 1)
        if wm_file:
            wm_rgb = read_rgb(wm_file)
            wm_rgba = to_rgba(wm_rgb)

            target_w = int(W * (scale_pct / 100.0))
            ratio = target_w / wm_rgba.shape[1]
            target_h = max(1, int(wm_rgba.shape[0] * ratio))
            wm_rgba = cv2.resize(wm_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
            ow, oh = wm_rgba.shape[1], wm_rgba.shape[0]

            def posxy():
                if pos == "Top-Left": return pad, pad
                if pos == "Top-Right": return W - ow - pad, pad
                if pos == "Bottom-Left": return pad, H - oh - pad
                if pos == "Bottom-Right": return W - ow - pad, H - oh - pad
                return (W - ow) // 2, (H - oh) // 2

            x, y = posxy()
            comp = paste_rgba_on_rgb(base_rgb, wm_rgba, x, y, opacity=opacity)

    st.subheader("Preview")
    st.image(comp, use_container_width=True)
    st.download_button(
        "Download watermarked PNG",
        data=pil_download(comp, "watermarked.png"),
        file_name="watermarked.png",
        mime="image/png",
    )

# ==================== Step 3: e-Signature ====================
elif page == "sign":
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
        height=canvas_h,
        width=canvas_w,
        drawing_mode="freedraw",
        key="sig_canvas",
    )

    sig_rgba = None
    if canvas.image_data is not None:
        sig_rgba = canvas.image_data.astype("uint8")
        st.caption("Signature preview")
        st.image(sig_rgba, use_container_width=True)
        st.download_button(
            "Download signature (PNG)",
            data=pil_download(sig_rgba, "signature.png"),
            file_name="signature.png",
            mime="image/png",
        )

    st.markdown("---")
    st.markdown("**2) Place signature on a base image**")

    base_file = st.file_uploader("Upload base image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if base_file:
        base_rgb = read_rgb(base_file)
        H, W = base_rgb.shape[:2]
        st.image(base_rgb, caption="Base image", use_container_width=True)

        up_sig = st.file_uploader("Or upload an existing signature PNG", type=["png"], key="sig_upload")
        if up_sig is not None:
            sig_rgba = np.array(Image.open(up_sig).convert("RGBA"))

        if sig_rgba is not None:
            scale = st.slider("Signature width (% of base width)", 5, 60, 25, 1)
            opacity = st.slider("Opacity", 0.1, 1.0, 0.9, 0.05)
            pos = st.selectbox("Position", ["Bottom-Right", "Bottom-Left", "Top-Right", "Top-Left", "Center"])
            pad = st.slider("Padding (px)", 0, max(20, min(W, H) // 5), 20)

            target_w = int(W * (scale / 100.0))
            ratio = target_w / sig_rgba.shape[1]
            target_h = max(1, int(sig_rgba.shape[0] * ratio))
            sig_resized = cv2.resize(sig_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

            sh, sw = sig_resized.shape[0], sig_resized.shape[1]
            if pos == "Bottom-Right":
                x, y = W - sw - pad, H - sh - pad
            elif pos == "Bottom-Left":
                x, y = pad, H - sh - pad
            elif pos == "Top-Right":
                x, y = W - sw - pad, pad
            elif pos == "Top-Left":
                x, y = pad, pad
            else:
                x, y = (W - sw) // 2, (H - sh) // 2

            composed = paste_rgba_on_rgb(base_rgb, sig_resized, x, y, opacity=opacity)
            st.image(composed, caption="Signed image", use_container_width=True)
            st.download_button(
                "Download signed image (PNG)",
                data=pil_download(composed, "signed_output.png"),
                file_name="signed_output.png",
                mime="image/png",
            )
        else:
            st.info("Draw a signature above or upload a signature PNG to place it.")
    else:
        st.info("Upload a base image to place your signature on.")

# ==================== Step 4: Logical Operations ====================
elif page == "logic":
    st.title("Step 4 — Logical Operations")

    st.write("Upload **two images**. The second will be resized to match the first. "
             "Optionally threshold them to see binary logical results.")

    cA, cB = st.columns(2)
    with cA:
        up1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"], key="logic1")
    with cB:
        up2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"], key="logic2")

    do_thresh = st.checkbox("Convert to grayscale + threshold (recommended)", value=True)
    T = st.slider("Threshold (used if above is checked)", 0, 255, 127)

    if up1 is not None and up2 is not None:
        img1 = read_rgb(up1)
        img2 = read_rgb(up2)
        # match sizes
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        if do_thresh:
            g1 = to_gray(img1)
            g2 = to_gray(img2)
            _, b1 = cv2.threshold(g1, T, 255, cv2.THRESH_BINARY)
            _, b2 = cv2.threshold(g2, T, 255, cv2.THRESH_BINARY)
            src1, src2 = b1, b2
        else:
            # Use color directly (bitwise works channel-wise)
            src1, src2 = img1, img2

        and_img = cv2.bitwise_and(src1, src2)
        or_img  = cv2.bitwise_or(src1, src2)
        xor_img = cv2.bitwise_xor(src1, src2)
        not_img = cv2.bitwise_not(src1)

        st.subheader("Inputs")
        c1, c2 = st.columns(2)
        with c1: st.image(src1, caption="Image 1 (processed)", use_container_width=True)
        with c2: st.image(src2, caption="Image 2 (processed)", use_container_width=True)

        st.subheader("Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.image(and_img, caption="AND", use_container_width=True)
        r2.image(or_img,  caption="OR",  use_container_width=True)
        r3.image(xor_img, caption="XOR", use_container_width=True)
        r4.image(not_img, caption="NOT (of Image 1)", use_container_width=True)

        st.download_button(
            "Download AND (PNG)",
            data=pil_download(and_img, "logical_and.png"),
            file_name="logical_and.png",
            mime="image/png",
        )
    else:
        st.info("Upload both images above to run logical operations.")

# ==================== Step 2-4: Alpha / Transparency ====================
elif page == "alpha":
    st.title("Step 2-4 — Alpha Channel / Transparency")

    st.markdown(
        "Upload an image and create transparency by (a) removing a color, or (b) "
        "using a threshold as the alpha channel. Download result as PNG."
    )

    up = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="alpha_img")
    if up is None:
        st.info("Upload an image to begin.")
        st.stop()

    rgb = read_rgb(up)       # (H,W,3)
    rgba = to_rgba(rgb)      # (H,W,4)

    st.subheader("Original")
    st.image(rgb, use_container_width=True)

    mode = st.radio("Mode", ["Remove a color", "Threshold as alpha"], horizontal=True)

    if mode == "Remove a color":
        st.markdown("**Pick a color to remove** (that color becomes transparent).")
        pick = st.color_picker("Target color", "#FFFFFF")
        tol  = st.slider("Tolerance (0–255)", 0, 255, 25)

        # Parse color picker (#RRGGBB) to (R,G,B)
        R = int(pick[1:3], 16)
        G = int(pick[3:5], 16)
        B = int(pick[5:7], 16)
        target = np.array([R, G, B], dtype=np.int16)

        diff = np.abs(rgb.astype(np.int16) - target[None, None, :]).sum(axis=2)
        mask = (diff <= tol).astype(np.uint8) * 255  # 255 where we remove

        out = rgba.copy()
        out[..., 3] = np.where(mask == 255, 0, 255).astype(np.uint8)

        st.subheader("Preview (checkerboard shows transparency)")
        st.image(preview_rgba(out), use_container_width=True)

        st.download_button(
            "Download PNG (color removed)",
            data=pil_download(out, "alpha_color_removed.png"),
            file_name="alpha_color_removed.png",
            mime="image/png",
        )

    else:  # "Threshold as alpha"
        st.markdown("**Make alpha from a grayscale threshold** (white=opaque by default).")
        invert = st.checkbox("Invert alpha (white = transparent)", value=False)
        method = st.selectbox("Threshold method", ["Global (manual T)", "Otsu (auto)"], index=1)

        gray = to_gray(rgb)
        if method == "Global (manual T)":
            T = st.slider("T", 0, 255, 160)
            _, alpha = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
        else:
            _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if invert:
            alpha = cv2.bitwise_not(alpha)

        out = rgba.copy()
        out[..., 3] = alpha

        st.subheader("Alpha channel")
        st.image(alpha, clamp=True, use_container_width=True)

        st.subheader("Preview (checkerboard shows transparency)")
        st.image(preview_rgba(out), use_container_width=True)

        st.download_button(
            "Download PNG (threshold alpha)",
            data=pil_download(out, "alpha_threshold.png"),
            file_name="alpha_threshold.png",
            mime="image/png",
        )
