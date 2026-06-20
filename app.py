import streamlit as st
import os
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib

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
    # A. 逐步回归计算 (作为基准锚点)
    formula_result = 47.249 + (0.618 * temp) + (0.129 * humidity) - (0.762 * moisture)

    # B. 随机森林计算 (带工程化修正)
    rf_raw = None
    rf_corrected = None

    try:
        # 1. 准备输入数据 (注意顺序必须与训练时一致: 气温, 湿度, 水分)
        input_data = np.array([[temp, humidity, moisture]])

        # 2. 加载模型
        model_path = 'rf_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件 {model_path}")

        loaded_rf_model = joblib.load(model_path)

        # 3. 获取原始预测值
        prediction = loaded_rf_model.predict(input_data)
        rf_raw = prediction[0]

        # ============================================================
        # 【关键步骤】工程化修正：解决 RF 预测值被“压扁”的问题
        # ============================================================
        # 原理：新预测值 = 原始值 * 斜率(slope) + 截距(intercept)
        # 作用：把 RF 的波动幅度拉大，使其能覆盖到极端高温或低温的情况。
        #
        # ⚠️ 参数说明：
        # slope (斜率): 建议设置在 1.3 ~ 1.8 之间。
        #   - 数值越大，RF 对高低值的反应越剧烈。
        #   - 如果 RF 还是偏低，请调大这个数。
        # intercept (截距): 用于微调整体基准线，通常设为 0 或很小的数。
        # ============================================================
        SLOPE_FACTOR = 1.5      # <--- 这里控制“拉伸”程度，先试 1.5
        INTERCEPT_OFFSET = 0.0  # <--- 这里控制整体上下平移

        rf_corrected = (rf_raw * SLOPE_FACTOR) + INTERCEPT_OFFSET

    except Exception as e:
        rf_raw = None
        rf_corrected = None
        st.error(f"❌ 模型计算失败: {e}")

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
        st.metric(label="逐步回归预测值 (基准)", value=f"{formula_result:.2f} ℃")
        st.info("公式：47.249 + 0.618T + 0.129H - 0.762W")

    with res_col2:
        if rf_corrected is not None:
            # 显示修正后的结果
            diff = rf_corrected - formula_result
            st.metric(label="随机森林预测值 (修正后)", value=f"{rf_corrected:.2f} ℃", delta=f"差异: {diff:+.2f}")

            # 简单的通过判定
            st.markdown(
                f"""
                <div style="color: green; {standard_font_style}">✅ 机器学习模型已应用偏差校正</div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("随机森林计算异常")

    # 偏差分析与文案输出 (字体统一为 16px)
    # 这里我们比较的是【修正后的RF】和【逐步回归】
    if rf_corrected is not None:
        final_diff = abs(formula_result - rf_corrected)

        if final_diff < 1.5:
            st.markdown(
                f"""
                <div style="color: green; {standard_font_style}">✅ 两种模型结果吻合度高 (偏差 {final_diff:.2f})，建议采纳！</div>
                """,
                unsafe_allow_html=True
            )
        elif final_diff < 3.0:
            st.markdown(
                f"""
                <div style="color: orange; {standard_font_style}">⚠️ 两种模型存在一定分歧 (偏差 {final_diff:.2f})。</div>
                <div style="{standard_font_style}">建议：参考两者平均值，或结合现场经验判断。</div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="color: red; {standard_font_style}">🛑 模型分歧较大 (偏差 {final_diff:.2f})！</div>
                <div style="{standard_font_style}">建议：此时环境可能处于极端状态，请务必人工复核母糟感官情况。</div>
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
        <p style="margin: 0;"><b>版本：</b>v2.1 (含RF偏差修正)</p>
    </div>
    """,
    unsafe_allow_html=True
)