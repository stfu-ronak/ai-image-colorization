import asyncio
import sys


if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import os
os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = "torch"

  
import streamlit as st

st.set_page_config(page_title="AI Colorizer", layout="centered")

st.title("🎨 AI Image Colorizer")
st.write("Welcome! Use the sidebar to navigate through different sections of the app.")

st.markdown("""
### 🔍 Navigation Guide
- **Colorizer**: Upload a grayscale image and colorize it.
- **Visualization**: See how color channels are enhanced.
- **About**: Learn how this model works and who built it.
""")

st.success("Use the sidebar ➡️ to explore.")
