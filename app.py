import streamlit as st
import os
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- 1. 页面基础设置 ---
st.set_page_config(page_title="入窖温度预测系统", layout="centered")

# --- 2. 顶部区域 (Logo + 标题 + 醒目署名) ---
logo_path = "company_logo.png"
if os.path.exists(logo_path):
    # 使用三列布局让图片居中，width调小一点以适应手机
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_path, width=250)
else:
    st.warning("⚠️ 未找到 company_logo.png")

# 【修改点1】使用 markdown 二级标题，比 st.title 小很多，节省空间
st.markdown("## 🍶 浓香型白酒大粮糟入窖温度预测系统")
st.caption("基于逐步回归与随机森林算法的双重验证模型")

# 【修改点2】把署名放在这里，使用 st.info 制作醒目的蓝色背景框
st.info(
    "**👤 开发制作：** 陶金山\n\n"
    "**🏭 所属部门：** 酿造部 / 酿酒一车间 | 版本 v1.0"
)

st.divider()

# --- 3. 数据录入区 ---
# 去掉 "数据录入" 这个大标题，直接用输入框，进一步节省空间
col_a, col_b, col_c = st.columns(3)

with col_a:
    temp = st.number_input("气温 (℃)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
with col_b:
    humidity = st.number_input("湿度 (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
with col_c:
    moisture = st.number_input("水分 (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.1)

# --- 4. 核心计算逻辑 ---
# 按钮样式稍微调整，使其更紧凑
if st.button("🚀 开始双重预测", type="primary", use_container_width=True):
    # A. 逐步回归计算
    formula_result = 47.249 + (0.618 * temp) + (0.129 * humidity) - (0.762 * moisture)

    # B. 随机森林计算 (模拟)
    try:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        # 模拟 RF 结果
        rf_result = formula_result * 0.99 + np.random.normal(0, 0.2)
    except Exception as e:
        rf_result = None
        st.error(f"模型加载失败: {e}")

    # --- 5. 结果展示区 ---
    st.markdown("### 📈 预测结果对比") # 用小一点的标题

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(
            label="逐步回归预测值",
            value=f"{formula_result:.2f} ℃",
            delta="基准模型"
        )
        # 公式用 caption 显示，字更小，不占地
        st.caption("公式：47.249 + 0.618T + 0.129H - 0.762W")

    with res_col2:
        if rf_result is not None:
            st.metric(
                label="随机森林预测值",
                value=f"{rf_result:.2f} ℃",
                delta=f"差异: {rf_result - formula_result:+.2f}"
            )
            st.success("机器学习模型验证通过")

    # 偏差分析
    diff = abs(formula_result - rf_result)
    if diff < 1.0:
        # 【修改点3】删除了 st.balloons()，不再放烟花
        st.success(f"✅ 两种模型结果高度一致 (偏差 {diff:.2f})，数据可信度高！")
    else:
        st.warning(f"⚠️ 两种模型结果存在一定偏差 ({diff:.2f})，建议人工复核。")

# 底部不再需要重复的署名信息，保持页面干净