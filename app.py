import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from skimage import color, filters, morphology, feature, exposure
from sklearn.cluster import KMeans

st.title("黑灰色层状物层界面 & 异常杂质识别系统（增强版）")

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
    overlay[edges] = [255,0,0]   # 红色表示层界面
    overlay[anomalies] = [0,255,0]  # 绿色表示异常杂质
    return overlay

# --- 层厚度计算 ---
def calculate_layer_thickness(edges):
    h, w = edges.shape
    thickness_data = []

    for col in range(w):
        ys = np.where(edges[:, col])[0]
        if len(ys) >= 2:
            col_thickness = ys[1:] - ys[:-1]
            if len(col_thickness) > 0:
                thickness_data.append(col_thickness.astype(float))  # ✅ 转 float

    if not thickness_data:
        return None, None

    max_len = max(len(x) for x in thickness_data)
    df_data = [
        np.pad(x, (0, max_len - len(x)), constant_values=np.nan).astype(float)
        for x in thickness_data
    ]
    df = pd.DataFrame(df_data)

    valid_values = df.values[~np.isnan(df.values) & (df.values > 0)]
    stats = {
        "平均厚度": float(np.mean(valid_values)),
        "最小厚度": float(np.min(valid_values)),
        "最大厚度": float(np.max(valid_values))
    }

    return stats, df

# --- 颜色聚类分割 ---
def color_clustering(img, n_clusters=3):
    """对图像进行 KMeans 聚类，区分不同颜色层/杂质"""
    img_np = np.array(img)
    h, w, c = img_np.shape
    reshaped = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(reshaped)

    clustered_img = labels.reshape((h, w))
    clustered_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, size=(n_clusters, 3))
    for i in range(n_clusters):
        clustered_rgb[clustered_img == i] = colors[i]

    return clustered_rgb

# --- 图像输入 ---
option = st.radio("选择图像输入方式", ["上传图片", "使用摄像头"])

if option == "上传图片":
    uploaded_file = st.file_uploader(
        "选择图片文件", 
        type=["jpg","png","jpeg","tif","tiff"]
    )
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

    # ✅ 新增颜色聚类
    clustered_result = color_clustering(img, n_clusters=4)

    st.subheader("识别结果对比")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="原图", use_container_width=True)
    with col2:
        st.image(result, caption="边界+杂质识别", use_container_width=True)
    with col3:
        st.image(clustered_result, caption="颜色聚类分割", use_container_width=True)

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


