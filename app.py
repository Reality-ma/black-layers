import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from skimage import color, filters, morphology, feature, exposure

st.title("黑灰色层状物层界面 & 异常杂质识别系统（无 OpenCV）")

# --- 图像处理函数 ---
def preprocess_image(img):
    gray = color.rgb2gray(np.array(img))
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.03)
    return enhanced

def detect_layer_edges(enhanced_gray):
    edges = feature.canny(enhanced_gray, sigma=1.0)
    edges_dilated = morphology.dilation(edges, morphology.rectangle(1,25))
    return edges_dilated

def detect_anomalies(enhanced_gray):
    thresh_val = filters.threshold_local(enhanced_gray, block_size=35)
    binary = enhanced_gray < thresh_val
    clean = morphology.opening(binary, morphology.square(3))
    return clean

def overlay_results(original_img, edges, anomalies):
    overlay = np.array(original_img).copy()
    overlay[edges] = [255,0,0]
    overlay[anomalies] = [0,255,0]
    return overlay

# --- 层厚度计算（修复空列报错） ---
def calculate_layer_thickness(edges):
    h, w = edges.shape
    thickness_data = []

    for col in range(w):
        ys = np.where(edges[:, col])[0]
        if len(ys) >= 2:
            col_thickness = ys[1:] - ys[:-1]
            if len(col_thickness) > 0:
                thickness_data.append(col_thickness)

    if not thickness_data:
        return None, None

    # 找到最长厚度列
    max_len = max(len(x) for x in thickness_data)

    # 构建 DataFrame，每列不足长度用 NaN 填充
    df_data = [np.pad(x, (0, max_len - len(x)), constant_values=np.nan) for x in thickness_data]
    df = pd.DataFrame(df_data).astype(float)

    # 统计非零厚度
    valid_values = df.values[~np.isnan(df.values) & (df.values>0)]
    stats = {
        "平均厚度": np.mean(valid_values),
        "最小厚度": np.min(valid_values),
        "最大厚度": np.max(valid_values)
    }

    return stats, df

# --- 图像输入 ---
option = st.radio("选择图像输入方式", ["上传图片", "使用摄像头"])

if option == "上传图片":
    uploaded_file = st.file_uploader("选择图片文件", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
elif option == "使用摄像头":
    captured_file = st.camera_input("拍摄照片")
    if captured_file is not None:
        img = Image.open(captured_file).convert("RGB")

# --- 处理与显示 ---
if 'img' in locals():
    enhanced = preprocess_image(img)
    edges = detect_layer_edges(enhanced)
    anomalies = detect_anomalies(enhanced)
    result = overlay_results(img, edges, anomalies)

    st.subheader("识别结果")
    st.image(result, use_container_width=True)

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
    result_pil = Image.fromarray(result.astype(np.uint8))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        result_pil.save(tmp_file.name)
        st.download_button("下载图片", tmp_file.name, file_name="layer_detection_result.png")
