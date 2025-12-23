import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import io

# =============================================================================
# Configuration & Setup
# =============================================================================
st.set_page_config(
    page_title="PixelForge: Image Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================
def convert_image(img, fmt):
    """Converts a PIL Image to bytes for download."""
    buf = io.BytesIO()
    # JPEG does not support transparency (RGBA), convert to RGB first
    if fmt.upper() == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    byte_im = buf.getvalue()
    return byte_im

def apply_resize(img, width, height):
    """Resizes the image."""
    return img.resize((width, height), Image.LANCZOS)

# =============================================================================
# Sidebar: Controls
# =============================================================================
st.sidebar.title("🛠️ Editor Tools")

# 1. Upload
st.sidebar.subheader("1. Upload")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded_file:
    # Load Image
    try:
        original_image = Image.open(uploaded_file)
        # Processed image variable to hold state through transformations
        processed_image = original_image.copy()
        
        # Get metadata
        orig_w, orig_h = original_image.size
        st.sidebar.info(f"Original Size: {orig_w} x {orig_h} px")

        # ---------------------------------------------------------
        # 2. Resize Logic
        # ---------------------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Resize")
        
        resize_mode = st.sidebar.radio("Resize Mode", ["Percentage", "Custom Dimensions"], index=0)
        
        if resize_mode == "Percentage":
            scale_percent = st.sidebar.slider("Scale %", 10, 200, 100, 10)
            new_w = int(orig_w * scale_percent / 100)
            new_h = int(orig_h * scale_percent / 100)
        else:
            # Aspect Ratio Logic
            lock_aspect = st.sidebar.checkbox("Lock Aspect Ratio", value=True)
            new_w = st.sidebar.number_input("Width (px)", min_value=1, value=orig_w)
            
            if lock_aspect:
                # Calculate height based on aspect ratio
                aspect_ratio = orig_h / orig_w
                new_h = int(new_w * aspect_ratio)
                st.sidebar.text(f"Height (Auto): {new_h} px")
            else:
                new_h = st.sidebar.number_input("Height (px)", min_value=1, value=orig_h)

        # Apply Resize immediately to the processed_image pipeline
        processed_image = apply_resize(processed_image, new_w, new_h)

        # ---------------------------------------------------------
        # 3. Orientation (Rotate/Flip)
        # ---------------------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.subheader("3. Orientation")
        
        rotate_angle = st.sidebar.slider("Rotate (Degrees)", -180, 180, 0, 90)
        flip_h = st.sidebar.checkbox("Flip Horizontal")
        flip_v = st.sidebar.checkbox("Flip Vertical")
        
        if rotate_angle != 0:
            processed_image = processed_image.rotate(-rotate_angle, expand=True)
        if flip_h:
            processed_image = processed_image.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_v:
            processed_image = processed_image.transpose(Image.FLIP_TOP_BOTTOM)

        # ---------------------------------------------------------
        # 4. Enhancements (Color/Filter)
        # ---------------------------------------------------------
        st.sidebar.markdown("---")
        st.sidebar.subheader("4. Enhancements")
        
        # Filters
        filter_type = st.sidebar.selectbox("Apply Filter", ["None", "Grayscale", "Blur", "Sharpen", "Contour", "Emboss"])
        if filter_type == "Grayscale":
            processed_image = processed_image.convert("L").convert("RGB") # Keep RGB mode for consistency
        elif filter_type == "Blur":
            processed_image = processed_image.filter(ImageFilter.BLUR)
        elif filter_type == "Sharpen":
            processed_image = processed_image.filter(ImageFilter.SHARPEN)
        elif filter_type == "Contour":
            processed_image = processed_image.filter(ImageFilter.CONTOUR)
        elif filter_type == "Emboss":
            processed_image = processed_image.filter(ImageFilter.EMBOSS)

        # Adjustments
        brightness = st.sidebar.slider("Brightness", 0.5, 3.5, 1.0, 0.1)
        contrast = st.sidebar.slider("Contrast", 0.5, 3.5, 1.0, 0.1)
        sharpness = st.sidebar.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)

        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(processed_image)
            processed_image = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(contrast)
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(processed_image)
            processed_image = enhancer.enhance(sharpness)

        # =============================================================================
        # Main Display Area
        # =============================================================================
        st.title("🎨 PixelForge Editor")
        
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.subheader("Original")
            st.image(original_image, use_container_width=True)
            
        with col2:
            st.subheader("Edited Result")
            st.image(processed_image, use_container_width=True)
            
            st.markdown("### Export")
            
            # Export Settings
            format_opt = st.selectbox("Format", ["PNG", "JPEG", "WEBP", "BMP"], index=0)
            
            # File name generation
            original_name = uploaded_file.name.rsplit(".", 1)[0]
            new_filename = f"{original_name}_edited.{format_opt.lower()}"
            
            # Prepare download
            file_bytes = convert_image(processed_image, format_opt)
            
            st.download_button(
                label=f"⬇️ Download {format_opt}",
                data=file_bytes,
                file_name=new_filename,
                mime=f"image/{format_opt.lower()}",
                type="primary",
                use_container_width=True
            )
            
            st.success(f"New Dimensions: {processed_image.width} x {processed_image.height}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("Please upload a valid image file.")

else:
    # Welcome Screen
    st.title("🎨 PixelForge Image Editor")
    st.markdown("""
    ### Welcome!
    Upload an image from the sidebar to get started.
    
    **Features:**
    * **Resize:** Scale by percentage or custom pixel dimensions.
    * **Edit:** Rotate, flip, and apply filters.
    * **Enhance:** Adjust brightness, contrast, and sharpness.
    * **Convert:** Save as PNG, JPEG, WEBP, or BMP.
    """)
    
    # Decorative placeholder
    st.info("👈 Waiting for upload...")
