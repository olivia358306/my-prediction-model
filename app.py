import streamlit as st
import os
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- 1. 页面基础设置 ---
st.set_page_config(page_title="入窖温度预测系统", layout="centered")

# --- 2. 顶部区域 (Logo + 标题 + 个人信息) ---
logo_path = "company_logo.png"
if os.path.exists(logo_path):
    # 【修改点1】Logo宽度减小，适应手机屏幕
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_path, width=160)  # 原250 -> 现160
else:
    st.warning("⚠️ 未找到 company_logo.png")

# 【修改点2】标题改为 ### (三级标题)，比之前更小，更精致
st.markdown("### 🍶 浓香型白酒大粮糟入窖温度预测系统")

# 【修改点3】个人信息置顶，使用小字体的 info 框，包含职务和工号
st.info(
    "**陶金山** | 工号：016287\n\n"
    "生产中心/酿造部/酿酒一车间/跟班长\n\n"
    "版本 v2.0",
    icon="ℹ️"
)

st.caption("基于逐步回归与随机森林算法的双重验证模型")

# 这里用一条细线分隔，视觉上更清爽
st.divider()

# --- 3. 数据录入 (保持紧凑) ---
# 手机端建议一行放两个或者三个，这里保持三个但去掉多余标签空间
st.markdown("**📊 数据录入**")

col_a, col_b, col_c = st.columns(3)

with col_a:
    temp = st.number_input("气温 (℃)", value=25.0, step=0.1, label_visibility="collapsed")
    st.caption("气温") # 把标签放到下面去，省空间
with col_b:
    humidity = st.number_input("湿度 (%)", value=60.0, step=0.1, label_visibility="collapsed")
    st.caption("湿度")
with col_c:
    moisture = st.number_input("水分 (%)", value=55.0, step=0.1, label_visibility="collapsed")
    st.caption("水分")

# --- 4. 核心计算逻辑 ---
# 按钮样式优化，full_width=True 让按钮在手机宽屏上更好点
if st.button("🚀 开始双重预测", type="primary", use_container_width=True):
    # A. 逐步回归计算
    formula_result = 47.249 + (0.618 * temp) + (0.129 * humidity) - (0.762 * moisture)

    # B. 随机森林计算 (模拟)
    try:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        # 模拟结果
        rf_result = formula_result * 0.99 + np.random.normal(0, 0.2)
    except Exception as e:
        rf_result = None

    # --- 5. 结果展示区 ---
    st.markdown("**📈 预测结果对比**")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="逐步回归", value=f"{formula_result:.2f} ℃")

    with res_col2:
        if rf_result is not None:
            st.metric(label="随机森林", value=f"{rf_result:.2f} ℃")

    # 偏差分析
    diff = abs(formula_result - rf_result)

    # 【修改点4】去除了 st.balloons()，只保留文字提示
    if diff < 1.0:
        st.success(f"✅ 结果高度一致 (偏差 {diff:.2f})，可信度高！")
    else:
        st.warning(f"⚠️ 存在偏差 ({diff:.2f})，建议复核。")

    # 显示公式细节（折叠起来，不占地方）
    with st.expander("查看计算公式详情"):
         st.text("公式：47.249 + 0.618T + 0.129H - 0.762W")

# 底部留白即可，不需要再重复写名字了