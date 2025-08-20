import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from skimage import color, filters, morphology, feature, exposure
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance

st.title("层状物识别与颜色人工标注系统（偏黑敏感+近似颜色合并+次要颜色边界可调）")

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

# --- RGB -> 名称（偏黑敏感） ---
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

# --- Lab 空间合并近似颜色 ---
def merge_similar_colors(avg_colors, threshold=10):
    lab_colors = color.rgb2lab(np.array(avg_colors).reshape(-1,1,3)).reshape(-1,3)
    n = len(lab_colors)
    merged_idx = [-1]*n
    new_labels = []
    label_count = 0
    for i in range(n):
        if merged_idx[i] != -1:
            continue
        merged_idx[i] = label_count
        for j in range(i+1, n):
            if distance.euclidean(lab_colors[i], lab_colors[j]) < threshold:
                merged_idx[j] = label_count
        new_labels.append(label_count)
        label_count += 1
    return merged_idx

# --- 颜色聚类 + 合并近似颜色 + 比例统计 ---
def color_clustering(img, n_clusters=4):
    img_np = np.array(img)
    h, w, c = img_np.shape
    reshaped = img_np.reshape((-1,3))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(reshaped)
    clustered_img = labels.reshape((h, w))
    clustered_rgb = np.zeros((h,w,3), dtype=np.uint8)
    avg_colors = []
    for i in range(n_clusters):
        cluster_pixels = reshaped[labels == i]
        avg_rgb = np.mean(cluster_pixels, axis=0)
        avg_colors.append(avg_rgb)
        mask = labels.reshape((h,w)) == i
        clustered_rgb[mask] = avg_rgb.astype(np.uint8)

    # 合并近似颜色
    merged_idx = merge_similar_colors(avg_colors)
    proportions_dict = {}
    color_info = []
    for i, idx in enumerate(merged_idx):
        mask = clustered_img == i
        prop = np.sum(mask)/mask.size*100
        name = rgb_to_name(avg_colors[i])
        if name in proportions_dict:
            proportions_dict[name] += prop
        else:
            proportions_dict[name] = prop
        color_info.append((name, prop, avg_colors[i], i))
    # 按比例排序
    color_info_sorted = sorted(color_info, key=lambda x: proportions_dict[x[0]], reverse=True)
    return clustered_rgb, proportions_dict, color_info_sorted, clustered_img

# --- 添加标签与次要颜色边界 ---
def add_labels(clustered_rgb, clustered_img, color_labels, highlight_minor=False, minor_threshold=5, contour_color=(255,0,255), contour_thickness=2):
    labeled_img = clustered_rgb.copy()
    for label, idx in color_labels.items():
        mask = clustered_img == idx
        if np.sum(mask)==0:
            continue
        y, x = np.argwhere(mask).mean(axis=0).astype(int)
        cv2.putText(labeled_img, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        if highlight_minor:
            area_ratio = np.sum(mask)/mask.size*100
            if area_ratio<minor_threshold:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled_img, contours, -1, contour_color, contour_thickness)
    return labeled_img

# --- 侧边栏 ---
st.sidebar.header("次要颜色边界设置")
minor_threshold = st.sidebar.slider("次要颜色面积阈值 (%)",1,15,5,1)
contour_thickness = st.sidebar.slider("边界线宽 (px)",1,5,2,1)
contour_color_choice = st.sidebar.selectbox("边界颜色",["紫色","红色","黄色","蓝色"])
contour_color_dict = {"紫色": (255,0,255), "红色": (255,0,0), "黄色": (255,255,0), "蓝色": (0,0,255)}
contour_color = contour_color_dict[contour_color_choice]

# --- 图像输入 ---
option = st.radio("选择图像输入方式", ["上传图片", "使用摄像头"])
img = None
if option=="上传图片":
    uploaded_file = st.file_uploader("选择图片文件", type=["jpg","png","jpeg","tif","tiff"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
elif option=="使用摄像头":
    captured_file = st.camera_input("拍摄照片")
    if captured_file is not None:
        img = Image.open(captured_file).convert("RGB")

# --- 处理与显示 ---
if img is not None:
    enhanced = preprocess_image(img)
    edges = detect_layer_edges(enhanced)
    anomalies = detect_anomalies(enhanced)
    result = overlay_results(img, edges, anomalies)

    n_clusters = st.slider("选择颜色类别数量",3,6,4,1)
    clustered_result, proportions, color_info, clustered_img = color_clustering(img, n_clusters=n_clusters)

    st.subheader("人工修改颜色名称")
    color_labels = {}
    for name, _, avg_rgb, idx in color_info:
        hex_color = '#%02x%02x%02x' % tuple(avg_rgb.astype(int))
        user_label = st.text_input(f"{name} ({hex_color}) 的名称", value=name)
        color_labels[user_label] = idx

    clustered_with_labels = add_labels(clustered_result, clustered_img, color_labels,
                                       highlight_minor=True, minor_threshold=minor_threshold,
                                       contour_color=contour_color, contour_thickness=contour_thickness)

    st.subheader("识别结果对比")
    col1,col2,col3 = st.columns(3)
    with col1: st.image(img, caption="原图", use_container_width=True)
    with col2: st.image(result, caption="边界+杂质识别", use_container_width=True)
    with col3: st.image(clustered_with_labels, caption="颜色聚类分割+标签+次要颜色边界", use_container_width=True)

    st.subheader("颜色比例统计")
    df_props = pd.DataFrame([(label, proportions[name] if name in proportions else 0) for label,name in zip(color_labels.keys(), color_labels.keys())],
                            columns=["颜色","比例 (%)"])
    st.table(df_props)

    fig, ax = plt.subplots()
    ax.pie(df_props["比例 (%)"], labels=df_props["颜色"], autopct="%.2f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    st.subheader("单独颜色类别展示")
    cols = st.columns(len(color_labels))
    for i,(label, idx) in enumerate(color_labels.items()):
        mask = clustered_img==idx
        single_img = np.ones_like(clustered_result,dtype=np.uint8)*255
        single_img[mask] = np.array(img)[mask]
        with cols[i%len(cols)]:
            st.image(single_img, caption=label, use_container_width=True)

    # CSV 下载
    color_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df_props.to_csv(color_csv_file.name, index=False)
    st.download_button("下载颜色比例 CSV", color_csv_file.name, file_name="color_proportions_named.csv")


