import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import io
import numpy as np
import zipfile

# ==== Page Setup ====
st.set_page_config(page_title="AI Image Colorizer", layout="wide")
st.title("🎨 AI-Powered Image Colorization for Real Estate")

# ==== Load Pretrained Model ====
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

def custom_image_comparison(img1, img2, width=700):
    import base64
    def img_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    img1_b64 = img_to_base64(img1)
    img2_b64 = img_to_base64(img2)

    slider_html = f"""
    <style>
    .container {{ position: relative; width: {width}px; max-width: 100%; }}
    .image-wrapper {{ position: relative; width: 100%; overflow: hidden; }}
    .image-wrapper img {{ display: block; width: 100%; height: auto; }}
    .img-overlay {{ position: absolute; top: 0; left: 0; width: 50%; overflow: hidden; }}
    .slider {{ position: absolute; top: 0; left: 50%; width: 5px; height: 100%; background: #fff; cursor: ew-resize; z-index: 10; transition: left 0.3s ease; border-radius: 2px; box-shadow: 0 0 10px rgba(0,0,0,0.3); }}
    </style>
    <div class="container" id="slider-container">
      <div class="image-wrapper">
        <img src="data:image/png;base64,{img1_b64}" />
        <div class="img-overlay" id="img-overlay">
          <img src="data:image/png;base64,{img2_b64}" />
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
      posX = Math.max(0, Math.min(rect.width, posX));
      slider.style.left = posX + 'px';
      imgOverlay.style.width = posX + 'px';
    }}
    slider.onmousedown = function() {{
      window.onmousemove = moveSlider;
      window.onmouseup = () => window.onmousemove = null;
    }};
    slider.ontouchstart = function(e) {{
      window.ontouchmove = evt => moveSlider(evt.touches[0]);
      window.ontouchend = () => window.ontouchmove = null;
    }};
    </script>
    """

# ==== Sidebar Navigation ====
main_section = st.sidebar.radio("Navigate", ["🏠 Home", "🛠️ Tools", "📊 Business", "⚙️ Settings"])

if main_section == "🏠 Home":
    st.subheader("Welcome to the AI Real Estate Colorizer")
    st.markdown("Enhance your property visuals with cutting-edge AI-powered image colorization. Ideal for real estate showcases, catalogs, and virtual staging.")

elif main_section == "📊 Business":
    st.subheader("Why This App Works for Real Estate")
    st.markdown("""
    - Save time and cost on professional photography and staging.
    - Bring old blueprints and grayscale photos to life with vivid colors.
    - Impress potential buyers with colorized interior and exterior images.
    - Generate ready-to-use property catalogs with enhanced visuals.
    """)

elif main_section == "⚙️ Settings":
    st.subheader("System Info & Settings")
    st.info("Coming Soon!")

elif main_section == "🛠️ Tools":
    st.subheader("Toolbox")
    selected_tool = st.radio("Choose a tool:", [
        "Colorization Tool",
        "Comparison Slider",
        "Batch Processing",
        "Blueprint Colorization",
        "Property Catalog Generator",
    ], horizontal=True)

    if selected_tool == "Colorization Tool":
        st.subheader("Upload and Colorize an Image")
        uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)
            input_l = preprocess(image)
            with torch.no_grad():
                ab_output = model(input_l)
            colorized = postprocess(ab_output, input_l)
            st.image(colorized, caption="Colorized Image", use_column_width=True)

            img_byte_arr = io.BytesIO()
            colorized.save(img_byte_arr, format='PNG')
            st.download_button("📥 Download Colorized Image", data=img_byte_arr.getvalue(), file_name="colorized.png", mime="image/png")

            st.session_state["original_image"] = image
            st.session_state["colorized_image"] = colorized

    elif selected_tool == "Comparison Slider":
        st.subheader("Before/After Comparison")
        if "original_image" in st.session_state and "colorized_image" in st.session_state:
            grayscale_preview = st.session_state["original_image"].convert("L").resize((256, 256))
            colorized_image = st.session_state["colorized_image"].resize((256, 256))
            custom_image_comparison(grayscale_preview, colorized_image)
        else:
            st.warning("Upload and colorize an image first.")

    elif selected_tool == "Batch Processing":
        st.subheader("Batch Colorization")
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
                for file in uploaded_files:
                    img = Image.open(file).convert("RGB")
                    input_l = preprocess(img)
                    with torch.no_grad():
                        ab_output = model(input_l)
                    colorized = postprocess(ab_output, input_l)
                    img_io = io.BytesIO()
                    colorized.save(img_io, format="PNG")
                    zip_file.writestr(f"colorized_{file.name}.png", img_io.getvalue())
            st.download_button("📥 Download All (ZIP)", data=zip_buffer.getvalue(), file_name="colorized_images.zip", mime="application/zip")
        else:
            st.info("Upload images to colorize.")
            
    elif selected_tool == "Blueprint Colorization":
        st.subheader("Blueprint Image Colorization")
        uploaded_blueprint = st.file_uploader("Upload a blueprint (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
        if uploaded_blueprint:
            image = Image.open(uploaded_blueprint).convert("RGB")
    
            # Show original
            st.image(image, caption="Original Blueprint", use_column_width=True)
    
            # Preprocess and enhance contrast
            blueprint_gray = image.convert("L").resize((256, 256))
            blueprint_np = np.array(blueprint_gray)
            enhanced = Image.fromarray(np.uint8(np.clip(1.5 * blueprint_np, 0, 255)))  # Contrast boost
    
            # Model expects normalized tensor input
            input_tensor = preprocess(enhanced)
    
            with torch.no_grad():
                ab_output = model(input_tensor)
            colorized = postprocess(ab_output, input_tensor)
    
            st.image(colorized, caption="Colorized Blueprint", use_column_width=True)
    
            # Download button
            img_byte_arr = io.BytesIO()
            colorized.save(img_byte_arr, format='PNG')
            st.download_button("📥 Download Colorized Blueprint", data=img_byte_arr.getvalue(), file_name="colorized_blueprint.png", mime="image/png")

    elif selected_tool == "Property Catalog Generator":
        st.subheader("🗂️ Generate a Property Catalog (PDF)")
    
        uploaded_catalog_images = st.file_uploader(
            "Upload property images (colorized, JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
    
        title = st.text_input("Catalog Title", value="AI-Powered Property Catalog")
        agent_name = st.text_input("Agent Name", value="John Doe")
        contact_info = st.text_input("Contact Info", value="📞 123-456-7890 | ✉️ johndoe@example.com")
    
        if uploaded_catalog_images:
            from fpdf import FPDF
    
            class CatalogPDF(FPDF):
                def header(self):
                    self.set_font("Arial", "B", 16)
                    self.cell(0, 10, title, ln=1, align="C")
    
                def footer(self):
                    self.set_y(-15)
                    self.set_font("Arial", "I", 10)
                    self.cell(0, 10, f"{agent_name} | {contact_info}", 0, 0, "C")
    
            pdf = CatalogPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
    
            for img in uploaded_catalog_images:
                image = Image.open(img).convert("RGB")
                img_path = f"/tmp/{img.name}"
                image.save(img_path)
    
                pdf.add_page()
                pdf.image(img_path, x=10, y=30, w=190)
    
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            st.success("📄 Property catalog generated!")
    
            st.download_button(
                label="📥 Download Catalog (PDF)",
                data=pdf_output.getvalue(),
                file_name="property_catalog.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Upload images to create a catalog.")

    else:
        st.subheader(f"{selected_tool}")
        st.info("Coming Soon!")
