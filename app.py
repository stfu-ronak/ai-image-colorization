import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import os
import io
import numpy as np

# ==== Page Setup ====
st.set_page_config(page_title="AI Image Colorizer", layout="wide")
st.title("🎨 AI-Powered Image Colorization for Real Estate")

# ==== Load Pretrained Model (Example Custom CNN) ====
class SimpleColorizationCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleColorizationCNN, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 2, 4, stride=2, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

@st.cache_resource
def load_model():
    model = SimpleColorizationCNN()
    model.load_state_dict(torch.load("colorization_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ==== Utility Functions ====
def preprocess(image: Image.Image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

def postprocess(ab_output, original_l):
    ab_output = ab_output.squeeze(0).detach().numpy()
    ab_output = np.clip(ab_output, -1, 1)
    ab_output = ab_output.transpose((1, 2, 0)) * 128

    original_l = original_l.squeeze(0).numpy()[0] * 50 + 50
    original_l = np.clip(original_l, 0, 100)

    lab = np.zeros((256, 256, 3))
    lab[..., 0] = original_l
    lab[..., 1:] = ab_output

    from skimage.color import lab2rgb
    rgb_image = lab2rgb(lab)
    rgb_image = (rgb_image * 255).astype(np.uint8)
    return Image.fromarray(rgb_image)

# ==== Sidebar Main Sections ====
main_section = st.sidebar.radio("Navigate", ["🏠 Home", "🛠️ Tools", "📊 Business", "⚙️ Settings"])

# ==== Tools Submodule Buttons ====
if main_section == "🛠️ Tools":
    st.subheader("Toolbox")
    selected_tool = st.radio("Choose a tool:", [
        "Upload & Preview",
        "AI Colorization",
        "Comparison Slider",
        "Batch Processing",
        "Image Enhancements",
        "Blueprint Colorization",
        "Interior Photo Colorization",
        "Property Catalog Generator",
        "Live Demo (Webcam)",
        "Model Selection",
        "Image Format Converter",
        "Colorization History",
        "Interactive Gallery"
    ], horizontal=True)

    if selected_tool == "Upload & Preview":
        st.subheader("Upload and Preview Images")

        uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.session_state["original_image"] = image
            st.success("Image uploaded and previewed successfully ✅")
        else:
            st.info("Please upload an image to preview it.")

    elif selected_tool == "AI Colorization":
        st.subheader("AI-Powered Colorization")

        if "original_image" not in st.session_state:
            st.warning("Please upload an image first in 'Upload & Preview'.")
        else:
            original_image = st.session_state["original_image"]

            with st.spinner("Colorizing image using AI model..."):
                input_l = preprocess(original_image)
                with torch.no_grad():
                    ab_output = model(input_l)
                colorized = postprocess(ab_output, input_l)

            st.image(colorized, caption="Colorized Image", use_column_width=True)
            st.success("Colorization complete ✅")

            st.download_button(
                label="📥 Download Colorized Image",
                data=io.BytesIO(colorized.convert("RGB").tobytes()),
                file_name="colorized.png",
                mime="image/png"
            )

            st.session_state["colorized_image"] = colorized

    elif selected_tool == "Comparison Slider":
        st.subheader("Before/After Comparison")
        
        def custom_image_comparison(img1, img2, width=700):
            # Convert images to base64 to embed directly in HTML (optional)
            import base64
            import io
            
            def img_to_base64(img):
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
        
            img1_b64 = img_to_base64(img1)
            img2_b64 = img_to_base64(img2)
            
            slider_html = f"""
            <style>
            .container {{
                position: relative;
                width: {width}px;
                max-width: 100%;
            }}
            .image-wrapper {{
                position: relative;
                width: 100%;
                overflow: hidden;
            }}
            .image-wrapper img {{
                display: block;
                width: 100%;
                height: auto;
            }}
            .img-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                width: 50%;
                overflow: hidden;
            }}
            .slider {{
                position: absolute;
                top: 0;
                left: 50%;
                width: 5px;
                height: 100%;
                background: #fff;
                cursor: ew-resize;
                z-index: 10;
                transition: left 0.3s ease;
                border-radius: 2px;
                box-shadow: 0 0 10px rgba(0,0,0,0.3);
            }}
            </style>
            <div class="container" id="slider-container">
              <div class="image-wrapper">
                <img src="data:image/png;base64,{img1_b64}" alt="Image 1" />
                <div class="img-overlay" id="img-overlay">
                  <img src="data:image/png;base64,{img2_b64}" alt="Image 2" />
                </div>
                <div class="slider" id="slider"></div>
              </div>
            </div>
            <script>
            const slider = document.getElementById('slider');
            const imgOverlay = document.getElementById('img-overlay');
            const container = document.getElementById('slider-container');
            
            function moveSlider(e) {{
              let rect = container.getBoundingClientRect();
              let posX = e.clientX - rect.left;
              if(posX < 0) posX = 0;
              if(posX > rect.width) posX = rect.width;
              slider.style.left = posX + 'px';
              imgOverlay.style.width = posX + 'px';
            }}
            
            slider.onmousedown = function(e) {{
              window.onmousemove = moveSlider;
              window.onmouseup = function() {{
                window.onmousemove = null;
                window.onmouseup = null;
              }}
            }};
            
            // Touch support for mobiles
            slider.ontouchstart = function(e) {{
              window.ontouchmove = function(evt) {{
                moveSlider(evt.touches[0]);
              }};
              window.ontouchend = function() {{
                window.ontouchmove = null;
                window.ontouchend = null;
              }};
            }};
            </script>
            """
            st.components.v1.html(slider_html, height=400)

# Usage inside your streamlit app:
if "original_image" in st.session_state and "colorized_image" in st.session_state:
    grayscale_preview = st.session_state["original_image"].convert("L").resize((256, 256))
    colorized_image = st.session_state["colorized_image"].resize((256, 256))
    custom_image_comparison(grayscale_preview, colorized_image, width=700)
else:
    st.warning("Upload and colorize an image first.")
    

    elif selected_tool == "Batch Processing":
        st.subheader("Batch Colorization")
        import zipfile
import tempfile
import os

if selected_tool == "Batch Processing":
    st.subheader("Batch Colorization - Upload Multiple Images")

    uploaded_files = st.file_uploader("Upload multiple images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} images. Processing...")

        colorized_images = []
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, width=100, caption=f"Original: {uploaded_file.name}")

            # Preprocess and colorize
            input_l = preprocess(img)
            with torch.no_grad():
                ab_output = model(input_l)
            colorized_img = postprocess(ab_output, input_l)
            colorized_images.append((uploaded_file.name, colorized_img))

        st.write("---")
        st.subheader("Colorized Images")

        cols = st.columns(min(4, len(colorized_images)))
        for idx, (name, cimg) in enumerate(colorized_images):
            with cols[idx % 4]:
                st.image(cimg, width=150, caption=f"Colorized: {name}")

        # Create ZIP for download
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "colorized_batch.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for name, cimg in colorized_images:
                    img_byte_arr = io.BytesIO()
                    cimg.save(img_byte_arr, format='PNG')
                    zipf.writestr(f"colorized_{name}", img_byte_arr.getvalue())

            with open(zip_path, "rb") as f:
                st.download_button(
                    label="📥 Download All Colorized Images (ZIP)",
                    data=f,
                    file_name="colorized_batch.zip",
                    mime="application/zip"
                )
    else:
        st.info("Please upload at least one image to start batch colorization.")


    elif selected_tool == "Image Enhancements":
        st.subheader("Enhance Your Colorized Images")
        # To Do: Brightness, contrast, saturation controls

    elif selected_tool == "Blueprint Colorization":
        st.subheader("Blueprint Colorization")
        # To Do: Color-coded zones (bedroom, kitchen etc.)

    elif selected_tool == "Interior Photo Colorization":
        st.subheader("Interior Image Colorization")
        # To Do: Specialized colorization for real estate photos

    elif selected_tool == "Property Catalog Generator":
        st.subheader("Generate Property Catalog (PDF)")
        # To Do: Combine colorized images into a PDF

    elif selected_tool == "Live Demo (Webcam)":
        st.subheader("Live Camera Colorization Demo")
        # To Do: Webcam feed and live grayscale->color pipeline

    elif selected_tool == "Model Selection":
        st.subheader("Model Selection & Settings")
        # To Do: Select between pretrained models or tweak settings

    elif selected_tool == "Image Format Converter":
        st.subheader("Convert Image Formats")
        # To Do: PNG <-> JPEG etc.

    elif selected_tool == "Colorization History":
        st.subheader("History & Session")
        # To Do: Store previous colorizations

    elif selected_tool == "Interactive Gallery":
        st.subheader("Gallery View")
        # To Do: Preview all uploaded/colorized images

elif main_section == "🏠 Home":
    st.subheader("Welcome to the AI Real Estate Colorizer")
    st.markdown("Enhance your property visuals with cutting-edge AI-powered image colorization. Ideal for real estate showcases, catalogs, and virtual staging.")

elif main_section == "📊 Business":
    st.subheader("Why This App Works for Real Estate")
    # To Do: Add business use case pitch and visuals

elif main_section == "⚙️ Settings":
    st.subheader("System Info & Settings")
    # To Do: Show CPU/GPU usage, processing time etc.
