import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="AI vs Human Image Detector", page_icon="🖼️")

# غيّري "YOUR-USERNAME/ai-human-resnet50" باسم المستخدم بتاعك الحقيقي على Hugging Face
# وغيّري "best_resnet50.keras" لو سميتي الملف بشكل مختلف
HF_REPO_ID = "YOUR-USERNAME/ai-human-resnet50"
MODEL_FILENAME = "best_resnet50.keras"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
    return keras.models.load_model(model_path)

model = load_model()

st.title("🖼️ AI vs Human Image Detector")
st.write("ارفع صورة وهيقولّك احتمالية إنها AI-generated ولا Human-generated.")

uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="الصورة اللي رفعتيها", use_container_width=True)

    img_resized = img.convert("RGB").resize((224, 224))
    arr = np.array(img_resized).astype("float32")
    arr = keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("بحلل الصورة..."):
        prob_ai = float(model.predict(arr)[0][0])
        prob_human = 1 - prob_ai

    st.subheader("النتيجة:")
    st.progress(prob_ai)
    st.write(f"**AI-Generated:** {prob_ai*100:.2f}%")
    st.write(f"**Human-Generated:** {prob_human*100:.2f}%")

    if prob_ai > 0.5:
        st.error(f"الصورة على الأرجح **AI-Generated** ({prob_ai*100:.1f}%)")
    else:
        st.success(f"الصورة على الأرجح **Human-Generated** ({prob_human*100:.1f}%)")
