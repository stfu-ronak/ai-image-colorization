import os
os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = "torch"
import streamlit as st
import torch

import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import numpy as np
import io

st.title("🎨 Colorize Your Image")

# Model
@st.cache_resource
def load_model():
    class ColorizationNet(nn.Module):
        def __init__(self):
            super(ColorizationNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
            self.conv4 = nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.relu(self.conv3(x))
            x = torch.sigmoid(self.conv4(x))
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorizationNet().to(device)
    model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Color Exaggeration
def exaggerate_colors(images, saturation_factor=1.5, value_factor=1.2):
    import torchvision.transforms.functional as F
    images = images.clamp(0, 1)
    np_img = images.permute(1, 2, 0).numpy()
    pil_img = Image.fromarray((np_img * 255).astype(np.uint8))
    hsv = pil_img.convert("HSV")
    hsv_np = np.array(hsv)
    hsv_np[..., 1] = np.clip(hsv_np[..., 1] * saturation_factor, 0, 255)
    hsv_np[..., 2] = np.clip(hsv_np[..., 2] * value_factor, 0, 255)
    enhanced_img = Image.fromarray(hsv_np, mode="HSV").convert("RGB")
    return enhanced_img

# Upload UI
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Colorize"):
        transform = transforms.ToTensor()
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).squeeze(0).cpu()

        result = exaggerate_colors(output_tensor)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Grayscale", use_column_width=True)
        col2.image(result, caption="Colorized", use_column_width=True)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("📥 Download Colorized Image", buf.getvalue(), "colorized.png", "image/png")
