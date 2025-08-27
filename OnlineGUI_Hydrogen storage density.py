import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hydrogen Storage Density Predictor", layout="centered")

# ----------------------------
# 加载模型与数据
# ----------------------------
model = joblib.load('NGboost.pkl')

df = pd.read_excel('储氢密度.xlsx', engine='openpyxl')

# ----------------------------
# 目标列与候选特征列
# ----------------------------
target_col = 'Hydrogen storage density(%)'
if target_col not in df.columns:
    st.error(f"未在数据集中发现目标列：{target_col}。请检查表头。")
    st.stop()

candidate_features = [c for c in df.columns if c != target_col]

# 若模型保存了训练时的特征名，优先使用以确保顺序一致
model_feature_names = getattr(model, 'feature_names_', None)
features = list(model_feature_names) if model_feature_names else candidate_features

# 仅保留数据中存在的特征（避免模型特征与当前数据集不一致导致报错）
features = [c for c in features if c in df.columns]
if not features:
    st.error("当前数据集中没有与模型匹配的特征列。请检查模型训练时的特征名与数据表头是否一致。")
    st.stop()

# ----------------------------
# 根据数据自动判断离散/数值特征
# - 非数值或唯一值<=10 视为离散
# - 其余为数值
# 并对 Number of cycles 强制为仅 1/2 选项
# ----------------------------
auto_discrete = []
auto_numeric = []
for c in features:
    if pd.api.types.is_numeric_dtype(df[c]):
        # 数值列若唯一值很少，也当作离散
        if df[c].nunique(dropna=True) <= 10:
            auto_discrete.append(c)
        else:
            auto_numeric.append(c)
    else:
        auto_discrete.append(c)

# 强制将 Number of cycles 视为离散，并且仅允许 1/2
cycles_col = 'Number of cycles'
if cycles_col in features:
    if cycles_col in auto_numeric:
        auto_numeric.remove(cycles_col)
    if cycles_col not in auto_discrete:
        auto_discrete.append(cycles_col)

discrete_cols = auto_discrete
numeric_cols = [c for c in features if c not in discrete_cols]

# ----------------------------
# UI
# ----------------------------
st.title("Magnesium Hydride Hydrogen Storage Density Prediction")

st.caption("Goal: Predict **Hydrogen storage density(%)**")

inputs = {}

# 数值特征：number_input
for col in numeric_cols:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    default = float(df[col].mean())
    rng = col_max - col_min
    step = 0.001 if rng < 1 else (0.01 if rng < 10 else 0.1)

    inputs[col] = st.number_input(
        label=col,
        min_value=col_min,
        max_value=col_max,
        value=default,
        step=step,
        format="%.6f"
    )

# 离散特征：selectbox
for col in discrete_cols:
    if col == cycles_col:
        # 只允许 1 或 2
        options = [1, 2]
        # 默认选数据中更常见的一个；若算不出就选 1
        default_val = 1
        if cycles_col in df.columns:
            mode_series = df[cycles_col].dropna()
            if not mode_series.empty:
                # 众数里若包含 1/2 则用之
                try:
                    m = mode_series.mode().iloc[0]
                    default_val = int(m) if m in [1, 2] else 1
                except Exception:
                    pass
        default_idx = options.index(default_val)
        inputs[col] = st.selectbox(col, options=options, index=default_idx)
        st.caption("Number of cycles can only be selected as 1 or 2.")
    else:
        # 其余离散列：从数据集中提取并排序
        options = pd.Series(df[col].dropna().unique()).tolist()
        try:
            options = sorted(options)
        except Exception:
            pass
        if len(options) == 0:
            options = [None]
        default_val = df[col].mode().iloc[0] if not df[col].mode().empty else options[0]
        default_idx = options.index(default_val) if default_val in options else 0
        inputs[col] = st.selectbox(col, options=options, index=default_idx)

# ----------------------------
# 组装输入并类型保护
# ----------------------------
X_input_df = pd.DataFrame([[inputs[c] for c in features]], columns=features)

# 基本类型保护：数值列转数值
for c in numeric_cols:
    X_input_df[c] = pd.to_numeric(X_input_df[c], errors='coerce')

# cycles 列若存在，转为整数
if cycles_col in X_input_df.columns:
    X_input_df[cycles_col] = pd.to_numeric(X_input_df[cycles_col], errors='coerce').astype('Int64')

# ----------------------------
# 预测 & 可解释性
# ----------------------------
if st.button("Predict"):
    # 一致性校验（可选）
    get_cnt = getattr(model, 'get_feature_count', None)
    expected = int(get_cnt()) if callable(get_cnt) else len(features)
    provided = X_input_df.shape[1]
    if provided != expected:
        st.error(
            f"❌ 特征数量不一致：模型期望 {expected} 个，当前提供 {provided} 个。\n"
            f"请核对特征顺序/名称：{features}"
        )
    else:
        # 预测（保留四位小数）
        y_pred = float(model.predict(X_input_df)[0])
        st.success(f"**Predicted Hydrogen storage density (%): {y_pred:.4f}**")

        # SHAP 可视化（可选）
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input_df)
            if isinstance(shap_values, list):  # 兼容分类模型返回 list 的情况
                shap_values = shap_values[0]
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                X_input_df.iloc[0],
                matplotlib=True
            )
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=600)
            st.image("shap_force_plot.png", caption="SHAP 力图（单样本）")
        except Exception as e:
            st.warning(f"SHAP 可视化未成功：{e}")
