# Wrapper entrypoint for Streamlit app
#s
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit demo arayüzü.
Bu arayüz FastAPI endpoint'ini kullanır.

Örnek:
streamlit run scripts/serve/launch_streamlit.py -- \
    --api-url http://127.0.0.1:8000/predict-file
"""

import argparse
import base64
import io

import requests
import streamlit as st
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Streamlit frontend for pneumonia demo.")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000/predict-file")
    args, _ = parser.parse_known_args()
    return args


def b64_to_pil(b64_string: str) -> Image.Image:
    data = base64.b64decode(b64_string.encode("utf-8"))
    return Image.open(io.BytesIO(data))


def main():
    args = parse_args()

    st.set_page_config(page_title="Pneumonia AI Demo", layout="wide")
    st.title("Pneumonia Risk + GAN Anomaly + Grad-CAM Demo")
    st.write("Chest X-ray yükleyin. Sistem pnömoni riski, Grad-CAM ve opsiyonel GAN anomaly haritası üretsin.")

    with st.sidebar:
        st.header("API Ayarları")
        api_url = st.text_input("Predict endpoint", value=args.api_url)

    uploaded_file = st.file_uploader("CXR görüntüsü yükle", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.subheader("Yüklenen Görüntü")
        st.image(image, use_container_width=True)

        if st.button("Tahmin Yap"):
            with st.spinner("API çağrılıyor..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(api_url, files=files, timeout=300)

            if response.status_code != 200:
                st.error(f"API hatası: {response.status_code}")
                st.text(response.text)
                return

            result = response.json()

            st.subheader("Tahmin Özeti")
            c1, c2, c3 = st.columns(3)
            c1.metric("Probability", f"{result['probability']:.4f}")
            c2.metric("Threshold", f"{result['threshold']:.4f}")
            c3.metric("Prediction", result["prediction_label"])

            if "gan_score" in result:
                c4, c5 = st.columns(2)
                c4.metric("GAN Score", f"{result['gan_score']:.4f}")
                c5.metric("Grad-CAM Score", f"{result.get('gradcam_score', 0.0):.4f}")
            else:
                st.metric("Grad-CAM Score", f"{result.get('gradcam_score', 0.0):.4f}")

            st.subheader("Görselleştirmeler")
            cols = st.columns(3)

            original_img = b64_to_pil(result["original_image_b64"])
            cols[0].image(original_img, caption="Original", use_container_width=True)

            gradcam_overlay = b64_to_pil(result["gradcam_overlay_b64"])
            cols[1].image(gradcam_overlay, caption="Grad-CAM Overlay", use_container_width=True)

            if "anomaly_overlay_b64" in result:
                anomaly_overlay = b64_to_pil(result["anomaly_overlay_b64"])
                cols[2].image(anomaly_overlay, caption="GAN Anomaly Overlay", use_container_width=True)

            st.subheader("Ham Haritalar")
            cols2 = st.columns(2)
            gradcam_map = b64_to_pil(result["gradcam_map_b64"])
            cols2[0].image(gradcam_map, caption="Grad-CAM Map", use_container_width=True)

            if "anomaly_map_b64" in result:
                anomaly_map = b64_to_pil(result["anomaly_map_b64"])
                cols2[1].image(anomaly_map, caption="GAN Anomaly Map", use_container_width=True)

            st.subheader("JSON Sonuç")
            st.json(result)


if __name__ == "__main__":
    main()