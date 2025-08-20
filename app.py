import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from skimage import filters, measure, morphology, color
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="层状物识别系统", layout="wide")

# 读取图像
def load_image(file):
    return Image.open(file)

# 颜色聚类
def color_clustering(image, n_colors=5):
    img_np = np.array(image)
    h, w, c = img_np.shape
    img_2d = img_np.reshape(-1, c)

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(img_2d)
    centers = kmeans.cluster_centers_.astype("uint8")

    clustered = centers[labels].reshape(h, w, c)

    proportions = {}
    masks = []
    for i in range(n_colors):
        mask = (labels == i).astype(np.uint8).reshape(h, w)
        color_name = f"颜色_{i+1}"
        proportions[color_name] = np.sum(mask) / mask.size
        mask_img = np.zeros_like(img_np)
        mask_img[mask == 1] = centers[i]
        masks.append((color_name, proportions[color_name], centers[i], mask_img))

    return clustered, masks

# 层界面检测
def detect_layer_boundaries(image):
    gray = color.rgb2gray(image)
    edges = filters.sobel(gray)
    binary = edges > filters.threshold_otsu(edges)
    return morphology.dilation(binary, morphology.disk(1))

# 杂质识别
def detect_impurities(image):
    gray = color.rgb2gray(image)
    thresh = filters.threshold_otsu(gray)
    binary = gray < thresh * 0.7
    labeled = measure.label(binary)
    impurities = morphology.remove_small_objects(labeled, 50)
    return impurities > 0

# CSV 导出
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


# ----------------- Streamlit 界面 -----------------
st.title("🪨 层状物与杂质识别系统")
uploaded_file = st.file_uploader("上传一张层状物图像 (jpg/png/tif)", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    # 原图
    image = load_image(uploaded_file).convert("RGB")
    st.image(image, caption="原始图像", use_container_width=True)

    # 聚类
    clustered, masks = color_clustering(image, n_colors=5)

    st.subheader("🎨 颜色聚类结果")
    st.image(clustered, caption="聚类图像", use_container_width=True)

    # 颜色比例统计
    proportions = {name: prop for name, prop, _, _ in masks}
    sorted_masks = sorted(masks, key=lambda x: x[1], reverse=True)

    # 显示饼图
    fig, ax = plt.subplots()
    labels = [name for name, _, _, _ in sorted_masks]
    sizes = [prop for _, prop, _, _ in sorted_masks]
    colors = [np.array(color) / 255 for _, _, color, _ in sorted_masks]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
    st.pyplot(fig)

    # 表格 + 可改名
    st.subheader("📝 聚类结果与人工命名")
    renamed = {}
    for i, (name, proportion, color_val, mask_img) in enumerate(sorted_masks):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(mask_img, caption=f"{name} ({proportion:.2%})", use_container_width=True)
        with col2:
            new_name = st.text_input(
                f"修改 [{name}] 的标签：",
                value=name,
                key=f"rename_{i}_{name}"   # 确保唯一 key
            )
            renamed[new_name] = proportion

    # 转 DataFrame
    df = pd.DataFrame({
        "颜色类别": list(renamed.keys()),
        "比例": list(renamed.values())
    })
    st.dataframe(df)

    # CSV 下载
    csv = convert_df(df)
    st.download_button("📥 下载颜色比例数据 (CSV)", data=csv, file_name="color_proportions.csv", mime="text/csv")

    # 层界面检测
    st.subheader("📏 层界面检测")
    boundaries = detect_layer_boundaries(np.array(image))
    st.image(boundaries, caption="层界面检测结果", use_container_width=True)

    # 杂质识别
    st.subheader("🧩 杂质识别")
    impurities = detect_impurities(np.array(image))
    st.image(impurities, caption="杂质识别结果", use_container_width=True)
