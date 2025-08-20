import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from skimage import color, filters, morphology, feature, exposure
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("层状物识别与颜色人工标注系统")

# --- 图像预处理 ---
def preprocess_image(img):
    gray = color.rgb2gray(np.array(img))
    enhanced = exposure.equalize_adapthist(gray, clip_limit=0.03)
    return enhanced

def detect_layer_edges(enhanced_gray):
    edges = feature.canny(enhanced_gray, sigma=1.0)
    edges_dilated = morphology.dilation(edges, morphology.rectangle(1, 25))
    return edges_dilated

def detect_anomalies(enhanced_gray):
    thresh_val = filters.threshold_local(enhanced_gray, block_size=35)
    binary = enhanced_gray < thresh_val
    clean = morphology.opening(binary, morphology.square(3))
    return clean

def overlay_results(original_img, edges, anomalies):
    overlay = np.array(original_img).copy()
    overlay[edges] = [255, 0, 0]     # 红色表示层界面
    overlay[anomalies] = [0, 255, 0] # 绿色表示异常杂质
    return overlay

# --- 层厚度计算（隐藏输出） ---
def calculate_layer_thickness(edges):
    h, w = edges.shape
    thickness_data = []
    for col in range(w):
        ys = np.where(edges[:, col])[0]
        if len(ys) >= 2:
            col_thickness = ys[1:] - ys[:-1]
            if len(col_thickness) > 0:
                thickness_data.append(col_thickness.astype(float))
    if not thickness_data:
        return None
    max_len = max(len(x) for x in thickness_data)
    df_data = [np.pad(x, (0, max_len - len(x)), constant_values=np.nan).astype(float) for x in thickness_data]
    df = pd.DataFrame(df_data)
    return df

# --- 丰富灰度颜色命名 ---
def rgb_to_name(rgb):
    r, g, b = rgb
    brightness = (r + g + b)/3
    if brightness < 30:
        return "黑色"
    elif brightness < 60:
        return "深灰"
    elif brightness < 100:
        return "灰色"
    elif brightness < 140:
        return "浅灰"
    elif brightness < 190:
        return "亮灰"
    else:
        return "白色"

# --- 颜色聚类 + 自动颜色命名 + 比例统计 ---
def color_clustering(img, n_clusters=4):
    img_np = np.array(img)
    h, w, c = img_np.shape
    reshaped = img_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(reshaped)
    clustered_img = labels.reshape((h, w))
    clustered_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    avg_colors = []
    for i in range(n_clusters):
        cluster_pixels = reshaped[labels == i]
        avg_rgb = np.mean(cluster_pixels, axis=0)
        avg_colors.append(avg_rgb)
        clustered_img_mask = labels.reshape((h, w)) == i
        clustered_rgb[clustered_img_mask] = avg_rgb.astype(np.uint8)
    color_names = [rgb_to_name(avg) for avg in avg_colors]
    proportions_named = {}
    for i, name in enumerate(color_names):
        proportions_named[name] = np.sum(labels == i) / len(labels) * 100
    return clustered_rgb, proportions_named, avg_colors, labels.reshape((h, w))

# --- 在聚类结果上叠加原图边界 ---
def overlay_on_cluster(clustered_img, edges, anomalies):
    overlay = clustered_img.copy()
    overlay[edges] = [255, 0, 0]      # 红色表示层界面
    overlay[anomalies] = [0, 255, 0]  # 绿色表示异常杂质
    return overlay

# --- 人工标注匹配函数 ---
def match_label(avg_rgb, mapping_df):
    diffs = np.sqrt(((mapping_df[['R','G','B']].values - avg_rgb) ** 2).sum(axis=1))
    return mapping_df.iloc[np.argmin(diffs)]['label']

# --- 图像输入 ---
option = st.radio("选择图像输入方式", ["上传图片", "使用摄像头"])
img = None
if option == "上传图片":
    uploaded_file = st.file_uploader("选择图片文件", type=["jpg","png","jpeg","tif","tiff"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
elif option == "使用摄像头":
    captured_file = st.camera_input("拍摄照片")
    if captured_file is not None:
        img = Image.open(captured_file).convert("RGB")

# --- 处理与显示 ---
if img is not None:
    enhanced = preprocess_image(img)
    edges = detect_layer_edges(enhanced)
    anomalies = detect_anomalies(enhanced)
    result = overlay_results(img, edges, anomalies)

    n_clusters = st.slider("选择颜色类别数量", min_value=3, max_value=6, value=4, step=1)
    clustered_result, proportions, avg_colors, labels_img = color_clustering(img, n_clusters=n_clusters)
    cluster_overlay_result = overlay_on_cluster(clustered_result, edges, anomalies)

    st.subheader("识别结果对比")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.image(img, caption="原图", use_container_width=True)
    with col2: st.image(result, caption="边界+杂质识别", use_container_width=True)
    with col3: st.image(clustered_result, caption="颜色聚类分割", use_container_width=True)
    with col4: st.image(cluster_overlay_result, caption="聚类+叠加边界", use_container_width=True)

    # 层厚度计算（隐藏）
    _ = calculate_layer_thickness(edges)

    # 颜色比例统计
    st.subheader("颜色比例统计")
    df_props = pd.DataFrame(list(proportions.items()), columns=["颜色", "比例 (%)"])
    st.table(df_props)
    fig, ax = plt.subplots()
    ax.pie(df_props["比例 (%)"], labels=df_props["颜色"], autopct="%.2f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # --- 人工标注 ---
    st.subheader("人工标注颜色类别")
    color_labels = {}
    for i, avg_rgb in enumerate(avg_colors):
        hex_color = '#%02x%02x%02x' % tuple(avg_rgb.astype(int))
        label = st.text_input(f"类别{i+1} ({hex_color}) 对应标签", value=f"标签{i+1}")
        color_labels[i] = label

    # 保存标注映射
    if st.button("保存标注映射 CSV"):
        mapping_df = pd.DataFrame({
            'cluster_index': list(color_labels.keys()),
            'label': list(color_labels.values()),
            'R': [avg[0] for avg in avg_colors],
            'G': [avg[1] for avg in avg_colors],
            'B': [avg[2] for avg in avg_colors]
        })
        mapping_df.to_csv("color_label_mapping.csv", index=False)
        st.success("标注映射已保存 color_label_mapping.csv")

    # --- 自动标签匹配展示 ---
    st.subheader("自动标签匹配结果（当前图片）")
    if st.button("匹配标签"):
        mapping_df = pd.DataFrame({
            'cluster_index': list(color_labels.keys()),
            'label': list(color_labels.values()),
            'R': [avg[0] for avg in avg_colors],
            'G': [avg[1] for avg in avg_colors],
            'B': [avg[2] for avg in avg_colors]
        })
        label_img = np.zeros(labels_img.shape, dtype=object)
        for i, avg_rgb in enumerate(avg_colors):
            label = match_label(avg_rgb, mapping_df)
            label_img[labels_img==i] = label
        st.write("标签示例（前100像素）:", label_img.flatten()[:100])

    # CSV 下载
    color_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_props.to_csv(color_csv_file.name, index=False)
    st.download_button("下载颜色比例 CSV", color_csv_file.name, file_name="color_proportions_named.csv")

    # 下载识别结果图片
    st.subheader("下载识别结果图片")
    result_pil = Image.fromarray(result.astype(np.uint8))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        result_pil.save(tmp_file.name)
        st.download_button("下载图片", tmp_file.name, file_name="layer_detection_result.png")



