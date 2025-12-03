import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =================== KONFIGURASI ===================

IMG_HEIGHT = 160
IMG_WIDTH = 160

# Label kelas sesuai dataset kamu
class_labels = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@st.cache_resource
def load_tomat_model():
    model = load_model("final_model_tomat.h5")
    return model

model = load_tomat_model()

# =================== FUNGSI PREDIKSI ===================

def predict_tomato(img: Image.Image):
    # resize & normalisasi sama seperti di Colab
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx])

    label = class_labels[idx]
    return label, confidence, preds[0]

# =================== STREAMLIT UI ===================

st.title("Deteksi Penyakit Daun Tomat")
st.subheader("Lightweight MobileNetV2 + Hyperparameter Optimization")

st.write("Upload gambar daun tomat, lalu klik tombol **Prediksi**.")

uploaded_file = st.file_uploader(
    "Pilih gambar daun tomat",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    if st.button("Prediksi"):
        label, conf, all_probs = predict_tomato(img)

        st.success(f"Prediksi: **{label}**")
        st.write(f"Confidence: **{conf:.2%}**")

        # Tampilkan probabilitas semua kelas
        probs_percent = (all_probs * 100).round(2)
        prob_dict = {cls: float(p) for cls, p in zip(class_labels, probs_percent)}
        st.write("Probabilitas semua kelas (%):")
        st.json(prob_dict)
