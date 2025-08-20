import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from skimage import color, filters, morphology, feature, exposure
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2

st.title("层状物识别与颜色标注系统（增强版）")

# --- 图像预处理 ---
def preprocess_image(img):
    gray = color.rgb2gray(np.array(img))
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.03)
    return enhanced

def detect_layer_edges(enhanced_gray):
    edges = feature.canny(enhanced_gray, sigma=1.0)
    return morphology.dilation(edges, morphology.rectangle(1, 25))

def detect_anomalies(enhanced_gray):
    thresh_val = filters.threshold_local(enhanced_gray, block_size=35)
    binary = enhanced_gray < thresh_val
    return morphology.opening(binary, morphology.square(3))

def overlay_edges(original_img, edges):
    overlay = np.array(original_img).copy()
    overlay[edges] = [255, 0, 0]  # 红色
    return overlay

def overlay_anomalies(original_img, anomalies):
    overlay = np.array(original_img).copy()
    overlay[anomalies] = [0, 255, 0]  # 绿色
    return overlay

# --- 自动颜色命名（偏黑敏感） ---
def rgb_to_name(rgb):
    r, g, b = rgb
    brightness = (r + g + b)/3
    if brightness < 15:
        return "纯黑"
    elif brightness < 35:
        return "黑色"
    elif brightness < 55:
        return "深灰"
    elif brightness < 85:
        return "灰色"
    elif brightness < 120:
        return "浅灰"
    elif brightness < 170:
        return "亮灰"
    else:
        return "白色"

# --- 颜色聚类 ---
def color_clustering(img, n_clusters=4):
    img_np = np.array(img)
    h, w, _ = img_np.shape
    reshaped = img_np.reshape((-1, 3))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(reshaped)

    clustered_img = labels.reshape((h, w))
    clustered_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    avg_colors, proportions = [], []
    for i in range(n_clusters):
        cluster_pixels = reshaped[labels == i]
        avg_rgb = np.mean(cluster_pixels, axis=0)
        avg_colors.append(avg_rgb)
        proportions.append(len(cluster_pixels) / len(reshaped) * 100)
        clustered_rgb[clustered_img == i] = avg_rgb.astype(np.uint8)

    # 排序：按面积比例从大到小
    color_info = sorted(
        [(rgb_to_name(avg_colors[i]), proportions[i], avg_colors[i], i) for i in range(n_clusters)],
        key=lambda x: x[1], reverse=True
    )
    return clustered_rgb, color_info, clustered_img

# --- 加标签 ---
def add_labels(clustered_rgb, clustered_img, color_info, custom_names):
    labeled_img = clustered_rgb.copy()
    for idx, (default_name, _, _, cluster_id) in enumerate(color_info):
        mask = clustered_img == cluster_id
        if np.sum(mask) == 0:
            continue
        y, x = np.argwhere(mask).mean(axis=0).astype(int)
        name = custom_names.get(default_name, default_name)
        cv2.putText(labeled_img, name, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return labeled_img

# --- 界面输入 ---
option = st.radio("选择图像输入方式", ["上传图片", "使用摄像头"])
img = None
if option == "上传图片":
    uploaded_file = st.file_uploader("选择图片文件", type=["jpg","png","jpeg","tif","tiff"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
elif option == "使用摄像头":
    captured_file = st.camera_input("拍摄照片")
    if captured_file:
        img = Image.open(captured_file).convert("RGB")

# --- 处理 ---
if img is not None:
    enhanced = preprocess_image(img)
    edges = detect_layer_edges(enhanced)
    anomalies = detect_anomalies(enhanced)

    edge_img = overlay_edges(img, edges)
    anomaly_img = overlay_anomalies(img, anomalies)

    n_clusters = st.slider("选择颜色类别数量", 3, 6, 4)
    clustered_result, color_info, clustered_img = color_clustering(img, n_clusters)

    # 人工修改颜色命名
    st.subheader("人工修改颜色命名")
    custom_names = {}
    for name, _, _, _ in color_info:
        new_name = st.text_input(f"修改 [{name}] 的标签：", value=name)
        custom_names[name] = new_name

    clustered_with_labels = add_labels(clustered_result, clustered_img, color_info, custom_names)

    # 显示结果
    st.subheader("识别结果对比")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.image(img, caption="原图", use_container_width=True)
    with col2: st.image(edge_img, caption="层界面识别", use_container_width=True)
    with col3: st.image(anomaly_img, caption="杂质识别", use_container_width=True)
    with col4: st.image(clustered_with_labels, caption="颜色聚类分割+标签", use_container_width=True)

    # 统计比例
    st.subheader("颜色比例统计")
    df_props = pd.DataFrame(
        [(custom_names[name], prop) for name, prop, _, _ in color_info],
        columns=["颜色", "比例 (%)"]
    )
    st.table(df_props)

    fig, ax = plt.subplots()
    ax.pie(df_props["比例 (%)"], labels=df_props["颜色"], autopct="%.2f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # 单独显示各类
    st.subheader("单独颜色类别展示")
    cols = st.columns(len(color_info))
    for i, (name, _, avg_rgb, cluster_id) in enumerate(color_info):
        mask = clustered_img == cluster_id
        single_img = np.ones_like(clustered_result, dtype=np.uint8) * 255
        single_img[mask] = avg_rgb.astype(np.uint8)
        with cols[i % len(cols)]:
            st.image(single_img, caption=f"{custom_names[name]}", use_container_width=True)

    # CSV 下载
    csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_props.to_csv(csv_file.name, index=False)
    st.download_button("下载颜色比例 CSV", csv_file.name, file_name="color_proportions_named.csv")





