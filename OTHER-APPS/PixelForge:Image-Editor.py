import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io

# Try to import the cropper, handle if missing
try:
    from streamlit_cropper import st_cropper
    HAS_CROPPER = True
except ImportError:
    HAS_CROPPER = False

# =============================================================================
# Configuration & Setup
# =============================================================================
st.set_page_config(
    page_title="PixelForge Pro",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button { width: 100%; }
    .main .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================
def convert_image(img, fmt):
    """Converts a PIL Image to bytes for download."""
    buf = io.BytesIO()
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    byte_im = buf.getvalue()
    return byte_im

def apply_resize(img, width, height):
    return img.resize((width, height), Image.LANCZOS)

# =============================================================================
# Main App Logic
# =============================================================================
st.sidebar.title("🛠️ Editor Tools")

# 1. Upload
st.sidebar.subheader("1. File Input")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file:
    try:
        # Load Original
        original_image = Image.open(uploaded_file)
        
        # Working image starts as original
        working_image = original_image.copy()

        # =====================================================================
        # STAGE 1: CROPPING (Interactive)
        # =====================================================================
        # We handle cropping in the main area (sidebar is too small for precision)
        
        st.title("✂️ PixelForge Editor")
        
        with st.expander("Step 1: Crop Image", expanded=True):
            if not HAS_CROPPER:
                st.warning("⚠️ `streamlit-cropper` not found. Run `pip install streamlit-cropper` to enable cropping.")
                cropped_image = original_image
            else:
                col_c1, col_c2 = st.columns([3, 1])
                with col_c2:
                    st.info("Adjust the box on the left, then check the result below.")
                    realtime_update = st.checkbox("Realtime Update", value=True)
                    box_color = st.color_picker("Box Color", "#0000FF")
                    aspect_choice = st.selectbox("Aspect Ratio", ["Free", "1:1", "16:9", "4:3"])
                    
                    aspect_dict = {
                        "Free": None,
                        "1:1": (1, 1),
                        "16:9": (16, 9),
                        "4:3": (4, 3)
                    }
                
                with col_c1:
                    # The st_cropper returns the cropped image directly
                    cropped_image = st_cropper(
                        original_image,
                        realtime_update=realtime_update,
                        box_color=box_color,
                        aspect_ratio=aspect_dict[aspect_choice],
                        should_resize_image=True # Resize for display, crop on original high-res
                    )
        
        # Update our pipeline to use the result of the crop
        working_image = cropped_image
        
        # Get current dimensions after crop
        curr_w, curr_h = working_image.size
        st.sidebar.info(f"Current Size: {curr_w} x {curr_h} px")

        # =====================================================================
        # STAGE 2: RESIZING
        # =====================================================================
        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Resize")
        resize_mode = st.sidebar.radio("Resize Mode", ["No Resize", "Percentage", "Custom"], index=0)
        
        new_w, new_h = curr_w, curr_h
        
        if resize_mode == "Percentage":
            scale_percent = st.sidebar.slider("Scale %", 10, 200, 100, 10)
            new_w = int(curr_w * scale_percent / 100)
            new_h = int(curr_h * scale_percent / 100)
            working_image = apply_resize(working_image, new_w, new_h)
            
        elif resize_mode == "Custom":
            lock_aspect = st.sidebar.checkbox("Lock Aspect Ratio", value=True)
            target_w = st.sidebar.number_input("Width (px)", min_value=1, value=curr_w)
            
            if lock_aspect:
                aspect_ratio = curr_h / curr_w
                target_h = int(target_w * aspect_ratio)
                st.sidebar.text(f"Height (Auto): {target_h} px")
            else:
                target_h = st.sidebar.number_input("Height (px)", min_value=1, value=curr_h)
            
            working_image = apply_resize(working_image, target_w, target_h)

        # =====================================================================
        # STAGE 3: ORIENTATION & FILTERS
        # =====================================================================
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. Edit & Enhance")
        
        # Orientation
        rotate_angle = st.sidebar.slider("Rotate", -180, 180, 0, 90)
        if rotate_angle != 0:
            working_image = working_image.rotate(-rotate_angle, expand=True)
            
        col_flip1, col_flip2 = st.sidebar.columns(2)
        if col_flip1.checkbox("Flip Horizontal"):
            working_image = working_image.transpose(Image.FLIP_LEFT_RIGHT)
        if col_flip2.checkbox("Flip Vertical"):
            working_image = working_image.transpose(Image.FLIP_TOP_BOTTOM)

        # Filters
        filter_type = st.sidebar.selectbox("Filter", ["None", "Grayscale", "Blur", "Sharpen", "Contour", "Emboss"])
        if filter_type == "Grayscale":
            working_image = working_image.convert("L").convert("RGB")
        elif filter_type == "Blur":
            working_image = working_image.filter(ImageFilter.BLUR)
        elif filter_type == "Sharpen":
            working_image = working_image.filter(ImageFilter.SHARPEN)
        elif filter_type == "Contour":
            working_image = working_image.filter(ImageFilter.CONTOUR)
        elif filter_type == "Emboss":
            working_image = working_image.filter(ImageFilter.EMBOSS)

        # Color Adjustment
        brightness = st.sidebar.slider("Brightness", 0.5, 3.5, 1.0, 0.1)
        if brightness != 1.0:
            working_image = ImageEnhance.Brightness(working_image).enhance(brightness)
            
        contrast = st.sidebar.slider("Contrast", 0.5, 3.5, 1.0, 0.1)
        if contrast != 1.0:
            working_image = ImageEnhance.Contrast(working_image).enhance(contrast)

        # =====================================================================
        # FINAL DISPLAY & EXPORT
        # =====================================================================
        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.subheader("Current Preview")
            # Show the image after all edits
            st.image(working_image, use_container_width=True)
            
        with col_res2:
            st.subheader("Export")
            format_opt = st.selectbox("Format", ["PNG", "JPEG", "WEBP", "BMP"], index=0)
            
            original_name = uploaded_file.name.rsplit(".", 1)[0]
            new_filename = f"{original_name}_edited.{format_opt.lower()}"
            
            file_bytes = convert_image(working_image, format_opt)
            
            st.download_button(
                label=f"⬇️ Download Result ({working_image.width}x{working_image.height})",
                data=file_bytes,
                file_name=new_filename,
                mime=f"image/{format_opt.lower()}",
                type="primary",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    # Landing Page
    st.title("✂️ PixelForge Pro")
    st.info("Upload an image in the sidebar to begin cropping and editing.")
