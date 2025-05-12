import os
os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = "torch"
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = "torch"

st.title("📊 Visualization of Color Enhancement")

st.markdown("Upload an image and we'll show **color channel distribution** before and after enhancement.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


def plot_color_histogram(image, title):
    np_img = np.array(image)
    plt.figure(figsize=(5, 2.5))
    for i, color in enumerate(["r", "g", "b"]):
        plt.hist(np_img[..., i].ravel(), bins=256, color=color, alpha=0.5)
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    st.pyplot(plt.gcf())
    plt.close()


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    plot_color_histogram(image, "Original RGB Distribution")

    transform = transforms.ToTensor()
    tensor_img = transform(image)


    def exaggerate_colors(tensor, saturation=1.5, value=1.2):
        np_img = tensor.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray((np_img * 255).astype(np.uint8))
        hsv = pil_img.convert("HSV")
        hsv_np = np.array(hsv)
        hsv_np[..., 1] = np.clip(hsv_np[..., 1] * saturation, 0, 255)
        hsv_np[..., 2] = np.clip(hsv_np[..., 2] * value, 0, 255)
        enhanced_img = Image.fromarray(hsv_np, mode="HSV").convert("RGB")
        return enhanced_img


    enhanced_image = exaggerate_colors(tensor_img)
    st.image(enhanced_image, caption="Enhanced Image")

    plot_color_histogram(enhanced_image, "Enhanced RGB Distribution")
