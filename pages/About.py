import streamlit as st

st.title("📘 About This Project")

st.markdown("""
This app is developed as a final college project to demonstrate **AI-based grayscale image colorization**.

### 🔧 Tech Stack
- **PyTorch** for deep learning model
- **Streamlit** for web app interface
- **PIL** for image manipulation
- **Matplotlib** for visualizations

### 🧠 Model Overview
- 4 convolutional layers
- Uses **dilated convolutions** for better spatial context
- Trained on **CIFAR-10 grayscale to RGB** conversion
- Uses **MSE loss**

### 🎯 Purpose
To demonstrate how deep learning can be used for creative tasks like colorization and how enhancement techniques can make results more visually appealing.

---

© 2025 – Built by [Your Name]. For academic purposes only.
""")
