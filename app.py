import streamlit as st
import os
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib  # 【新增】用于加载模型文件

# --- 1. 页面基础设置 ---
st.set_page_config(page_title="入窖温度预测系统", layout="centered")

# 定义全局标准字体样式 (16px)
standard_font_style = "font-size: 16px; font-weight: bold; color: #333;"

# --- 2. 顶部区域 (极致紧凑布局) ---

# Logo 区域
logo_path = "company_logo.png"
if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_path, width=140, use_column_width=False)
        # 使用 CSS 强制减少图片下方的留白
        st.markdown("<div style='margin-top: -20px;'></div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ 未找到 company_logo.png")

# 标题区域 (统一为 16px)
st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 5px;">
        <div style="{standard_font_style}">浓香型白酒大粮糟入窖温度预测系统</div>
    </div>
    """,
    unsafe_allow_html=True
)

# 副标题 (保持小字)
st.caption("基于逐步回归与随机森林算法的双重验证模型")

# 分割线
st.divider()

# --- 3. 数据录入区域 (标题字体统一为 16px) ---
st.markdown(
    f"""
    <div style="{standard_font_style}">数据录入</div>
    """,
    unsafe_allow_html=True
)

col_a, col_b, col_c = st.columns(3)

with col_a:
    temp = st.number_input("气温 (℃)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
with col_b:
    humidity = st.number_input("湿度 (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
with col_c:
    moisture = st.number_input("水分 (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.1)

# --- 4. 核心计算逻辑 ---
if st.button("🚀 开始双重预测", type="primary"):
    # A. 逐步回归计算
    formula_result = 47.249 + (0.618 * temp) + (0.129 * humidity) - (0.762 * moisture)

    # B. 随机森林计算 (真实模型加载)
    rf_result = None
    try:
        # 1. 准备输入数据
        # 注意：这里的顺序必须和你训练模型时的特征顺序完全一致！
        # 假设你训练时是按 [气温, 湿度, 水分] 的顺序输入的
        input_data = np.array([[temp, humidity, moisture]])

        # 2. 加载模型
        model_path = 'rf_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件 {model_path}，请确保它和 app.py 在同一目录下。")

        loaded_rf_model = joblib.load(model_path)

        # 3. 进行预测
        prediction = loaded_rf_model.predict(input_data)
        rf_result = prediction[0]

    except Exception as e:
        rf_result = None
        st.error(f"❌ 模型加载失败: {e}")

    # --- 5. 结果展示区 ---

    # 结果标题字体统一为 16px
    st.markdown(
        f"""
        <div style="{standard_font_style}">预测结果对比</div>
        """,
        unsafe_allow_html=True
    )

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="逐步回归预测值", value=f"{formula_result:.2f} ℃", delta="基准模型")
        st.info("公式：47.249 + 0.618T + 0.129H - 0.762W")

    with res_col2:
        if rf_result is not None:
            st.metric(label="随机森林预测值", value=f"{rf_result:.2f} ℃", delta=f"差异: {rf_result - formula_result:+.2f}")
            # 机器学习模型验证通过字体统一为 16px
            st.markdown(
                f"""
                <div style="color: green; {standard_font_style}">✅ 机器学习模型验证通过</div>
                """,
                unsafe_allow_html=True
            )

    # 偏差分析与文案输出 (字体统一为 16px)
    diff = abs(formula_result - rf_result)

    if diff < 1.0:
        st.markdown(
            f"""
            <div style="color: green; {standard_font_style}">✅ 两种模型结果高度一致 (偏差 {diff:.2f})，数据可信度高!</div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="color: orange; {standard_font_style}">⚠️ 两种模型结果存在一定偏差 ({diff:.2f})，建议人工复核。</div>
            """,
            unsafe_allow_html=True
        )

# --- 6. 底部信息栏 (已移至最下方，字体缩小) ---
st.divider()
st.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #888; line-height: 1.5;">
        <p style="margin: 0;"><b>开发制作：</b>陶金山 | 工号：016287</p>
        <p style="margin: 0;"><b>所属部门：</b>生产中心/酿酒一车间</p>
        <p style="margin: 0;"><b>版本：</b>v2.0</p>
    </div>
    """,
    unsafe_allow_html=True
)