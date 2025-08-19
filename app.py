import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd

st.title("黑灰色层状物层界面 & 异常杂质识别系统（带厚度统计）")

# --- 图像处理函数 ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def detect_layer_edges(enhanced_gray):
    edges = cv2.Canny(enhanced_gray, 50, 150)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    edges_dilated = cv2.dilate(edges, horizontal_kernel, iterations=1)
    return edges_dilated

def detect_anomalies(enhanced_gray):
    thresh = cv2.adaptiveThreshold(enhanced_gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 5)
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return clean

def overlay_results(original_img, edges, anomalies):
    overlay = original_img.copy()
    overlay[edges > 0] = [0,0,255]      # 层界面红色
    overlay[anomalies > 0] = [0,255,0]  # 异常绿色
    return overlay

# --- 层厚度计算 ---
def calculate_layer_thickness(edges):
    h, w = edges.shape
    thickness_data = []

    for col in range(w):
        ys = np.where(edges[:, col] > 0)[0]
        if len(ys) >= 2:
            col_thickness = ys[1:] - ys[:-1]
            thickness_data.append(col_thickness)

    if not thickness_data:
        return None, None

    thickness_array = np.array(thickness_data)
    df = pd.DataFrame(thickness_array).fillna(0).astype(int)
    stats = {
        "平均厚度": df.values[df.values>0].mean(),
        "最小厚度": df.values[df.values>0].min(),
        "最大厚度": df.values[df.values>0].max()
    }
    return stats, df

# --- 图像输入 ---
option = st.radio("选择图像输入方式", ["上传图片", "使用摄像头"])

if option == "上传图片":
    uploaded_file = st.file_uploader("选择图片文件", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
elif option == "使用摄像头":
    captured_file = st.camera_input("拍摄照片")
    if captured_file is not None:
        image = Image.open(captured_file).convert("RGB")
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# --- 处理与显示 ---
if 'img' in locals():
    enhanced = preprocess_image(img)
    edges = detect_layer_edges(enhanced)
    anomalies = detect_anomalies(enhanced)
    result = overlay_results(img, edges, anomalies)

    st.subheader("识别结果")
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 厚度统计
    thickness_stats, thickness_df = calculate_layer_thickness(edges)
    if thickness_stats:
        st.subheader("层厚度统计 (像素)")
        st.write(thickness_stats)

        # 下载 CSV
        csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        thickness_df.to_csv(csv_file.name, index=False)
        st.download_button("下载层厚度 CSV", csv_file.name, file_name="layer_thickness.csv")
    else:
        st.write("未检测到有效层厚度")

    # 下载识别图片
    st.subheader("下载识别结果图片")
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        result_pil.save(tmp_file.name)
        st.download_button("下载图片", tmp_file.name, file_name="layer_detection_result.png")

