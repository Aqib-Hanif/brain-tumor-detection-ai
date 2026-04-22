import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# 🔷 PAGE CONFIG
st.set_page_config(page_title="Brain Tumor AI Dashboard", layout="wide")

# 🔷 LOAD MODEL
model = tf.keras.models.load_model("brain_tumor_model.h5")
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 🔷 CUSTOM STYLE (CLEAN UI)
st.markdown("""
    <style>
    .main-title {text-align:center; font-size:36px; font-weight:bold;}
    .sub-title {text-align:center; font-size:18px; color:gray;}
    </style>
""", unsafe_allow_html=True)

# 🔷 HEADER
st.markdown("<div class='main-title'>🧠 Brain Tumor Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI + IoT Enabled Smart Healthcare System</div>", unsafe_allow_html=True)

# 🔷 SIDEBAR
with st.sidebar:
    st.header("👨‍🎓 Research Info")
    st.write("**Aqib Hanif**")
    st.write("**Assigned By: Dr.Adnan Amin**")
    st.write("MS Data Science")
    st.write("IMSciences Peshawar")
    st.markdown("---")
    st.info("Upload MRI to analyze tumor")

# 🔷 GRAD-CAM FUNCTION
def get_gradcam_heatmap(model, img_array, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 🔷 FILE UPLOAD
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 🔷 PREPROCESS
    img_resized = cv2.resize(img, (224,224))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🔷 PREDICTION
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # 🔷 METRIC CARDS
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", class_labels[class_index])

    with col2:
        st.metric("Confidence", f"{confidence:.2f}")

    with col3:
        st.metric("Status", "Tumor" if class_index != 2 else "Healthy")

    # 🔷 IMAGE DISPLAY (SIDE BY SIDE)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original MRI")
        st.image(img, width='stretch')

    # 🔷 GRAD-CAM
    heatmap = get_gradcam_heatmap(model, img_array)
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    gradcam_img = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)

    with col2:
        st.subheader("🔥 Model Attention (Grad-CAM)")
        st.image(gradcam_img, width='stretch')

    # 🔷 PROBABILITY BAR CHART
    st.markdown("---")
    st.subheader("📊 Class Probability Distribution")

    fig, ax = plt.subplots()
    ax.bar(class_labels, prediction[0])
    ax.set_ylabel("Probability")
    ax.set_ylim([0,1])
    st.pyplot(fig)

    # 🔷 ALERT MESSAGE
    if class_index == 2:
        st.success("✅ No Tumor Detected — Patient appears healthy")
    else:
        st.error(f"⚠ Tumor Detected: {class_labels[class_index]} — Further diagnosis recommended")

# 🔷 FOOTER
st.markdown("---")
st.markdown("<p style='text-align:center;'>Developed by Aqib Hanif | AI-based Brain Tumor Detection System</p>", unsafe_allow_html=True)