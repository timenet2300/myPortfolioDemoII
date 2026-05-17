# 必要套件（請確認已安裝）：
# pip install streamlit numpy pandas scipy plotly
# 或將上述套件寫入 requirements.txt 後部署至 Streamlit Cloud

import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ----- 安全匯入 plotly -----
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("❌ 缺少 plotly 套件，請在終端機執行：`pip install plotly` 或將 'plotly' 加入 requirements.txt 後重新部署。")
    st.stop()   # 中止後續執行，避免因 go 未定義而引發更多錯誤

st.set_page_config(layout="wide", page_title="投資組合最佳化工具")
st.title("📊 互動式投資組合最佳化")
st.markdown("設定 **2~5 檔股票** 的預期報酬、風險與相關係數，自動繪製效率前緣並計算四種最佳組合。")

# ----------------------------- 左側控制區 -----------------------------
with st.sidebar:
    st.header("⚙️ 基本設定")
    n_assets = st.selectbox("📈 資產數量", [2, 3, 4, 5], index=0)
    risk_free_rate = st.number_input("💰 無風險利率 (%)", value=0.0, step=0.1) / 100.0
    allow_short = st.checkbox("✅ 允許放空 (權重可為負)", value=True)

    with st.expander("📊 個股報酬與風險", expanded=True):
        returns = []
        vols = []
        cols = st.columns(2)
        for i in range(n_assets):
            with cols[0]:
                ret = st.number_input(f"股票 {i+1} 預期報酬 (%)", value=(i+1)*3.0, step=0.5, key=f"ret_{i}") / 100.0
            with cols[1]:
                vol = st.number_input(f"股票 {i+1} 風險 (%)", value=(i+1)*5.0, step=1.0, key=f"vol_{i}") / 100.0
            returns.append(ret)
            vols.append(vol)

    with st.expander("🔗 相關係數設定", expanded=True):
        corr_mode = st.radio("輸入方式", ["固定相關係數 ρ", "自訂相關係數矩陣"], horizontal=True)
        if corr_mode == "固定相關係數 ρ":
            rho_const = st.slider("ρ 值 (-1 ~ 1)", -1.0, 1.0, 0.5, 0.05)
            corr_matrix = np.full((n_assets, n_assets), rho_const)
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            st.markdown("編輯下方對稱矩陣 (對角線固定為 1)")
            default_corr = np.eye(n_assets)
            for i in range(n_assets):
                for j in range(i):
                    default_corr[i, j] = default_corr[j, i] = 0.3
            edited_corr = st.data_editor(
                pd.DataFrame(default_corr, columns=[f"資產{i+1}" for i in range(n_assets)], index=[f"資產{i+1}" for i in range(n_assets)]),
                key="corr_editor",
                use_container_width=True
            )
            corr_matrix = edited_corr.values
            np.fill_diagonal(corr_matrix, 1.0)

    with st.expander("🎯 目標報酬與風險", expanded=False):
        target_return = st.number_input("目標報酬率 r0 (%)", value=8.0, step=1.0) / 100.0
        target_risk = st.number_input("目標風險 R0 (%)", value=15.0, step=1.0) / 100.0

# ----------------------------- 計算共變異數矩陣 -----------------------------
vols_arr = np.array(vols)
cov_matrix = np.outer(vols_arr, vols_arr) * corr_matrix

# ----------------------------- 最佳化輔助函數 -----------------------------
def portfolio_stats(weights):
    ret = np.dot(weights, returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - risk_free_rate) / risk if risk > 1e-8 else -np.inf
    return ret, risk, sharpe

def neg_sharpe(weights):
    return -portfolio_stats(weights)[2]

def portfolio_variance(weights):
    return portfolio_stats(weights)[1]**2

def neg_return(weights):
    return -portfolio_stats(weights)[0]

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n_assets)]
init_weights = np.ones(n_assets) / n_assets

# 1. 最大夏普比率
try:
    res_sharpe = minimize(neg_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    tangency_weights = res_sharpe.x if res_sharpe.success else init_weights
except:
    tangency_weights = init_weights

# 2. 最小風險組合
res_minvar = minimize(portfolio_variance, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
minvar_weights = res_minvar.x if res_minvar.success else init_weights

# 3. 給定目標報酬 r0 的最小風險
def target_return_constraint(w, target):
    return np.dot(w, returns) - target
cons_r0 = [constraints, {'type': 'eq', 'fun': target_return_constraint, 'args': (target_return,)}]
res_r0 = minimize(portfolio_variance, init_weights, method='SLSQP', bounds=bounds, constraints=cons_r0)
target_ret_weights = res_r0.x if res_r0.success else init_weights

# 4. 給定目標風險 R0 的最大報酬
def target_risk_constraint(w, target):
    risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    return risk - target
cons_rR = [constraints, {'type': 'eq', 'fun': target_risk_constraint, 'args': (target_risk,)}]
res_rR = minimize(neg_return, minvar_weights, method='SLSQP', bounds=bounds, constraints=cons_rR)
target_risk_weights = res_rR.x if res_rR.success else minvar_weights

# ----------------------------- 效率前緣計算 -----------------------------
def efficient_frontier_points(n_points=100):
    min_ret = np.min(returns)
    max_ret = np.max(returns)
    if allow_short:
        min_ret = min_ret - 0.5 * (max_ret - min_ret)
        max_ret = max_ret + 0.5 * (max_ret - min_ret)
    target_rets = np.linspace(min_ret, max_ret, n_points)
    ef_rets = []
    ef_risks = []
    for r in target_rets:
        cons = [constraints, {'type': 'eq', 'fun': target_return_constraint, 'args': (r,)}]
        res = minimize(portfolio_variance, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success and res.fun > 0:
            ef_rets.append(r)
            ef_risks.append(np.sqrt(res.fun))
    return ef_rets, ef_risks

with st.spinner("📐 正在計算效率前緣..."):
    ef_rets, ef_risks = efficient_frontier_points(150)

# ----------------------------- 顯示結果表格 -----------------------------
results = []
for name, w in [("🏆 最大夏普比率", tangency_weights),
                ("🛡️ 最小風險組合", minvar_weights),
                (f"🎯 目標報酬 = {target_return*100:.1f}% (最小風險)", target_ret_weights),
                (f"⚡ 目標風險 = {target_risk*100:.1f}% (最大報酬)", target_risk_weights)]:
    ret, risk, sharpe = portfolio_stats(w)
    results.append({
        "組合名稱": name,
        "預期報酬 (%)": f"{ret*100:.2f}",
        "風險 (標準差 %)": f"{risk*100:.2f}",
        "夏普比率": f"{sharpe:.3f}",
        "權重": "、".join([f"{w_i:.2%}" for w_i in w])
    })

st.subheader("📋 最佳投資組合")
st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ----------------------------- 繪製效率前緣圖 -----------------------------
fig = go.Figure()

# 效率前緣曲線
fig.add_trace(go.Scatter(x=[r*100 for r in ef_risks], y=[r*100 for r in ef_rets],
                         mode='lines', name='效率前緣',
                         line=dict(color='#1f77b4', width=3),
                         hovertemplate='風險: %{x:.2f}%<br>報酬: %{y:.2f}%<extra></extra>'))

# 個別資產 (加上右側文字標籤)
for i in range(n_assets):
    fig.add_trace(go.Scatter(x=[vols[i]*100], y=[returns[i]*100],
                             mode='markers+text',
                             name=f'資產 {i+1}',
                             text=[f'資產 {i+1}'],
                             textposition='middle right',
                             textfont=dict(size=11, color='black'),
                             marker=dict(size=12, symbol='circle', line=dict(width=1, color='DarkSlateGrey')),
                             hovertemplate=f'資產 {i+1}<br>報酬: {returns[i]*100:.2f}%<br>風險: {vols[i]*100:.2f}%<extra></extra>'))

# 四個最佳組合 (加上右側文字標籤)
portfolio_markers = [
    (tangency_weights, "最大夏普比率", "red", "star"),
    (minvar_weights, "最小風險組合", "green", "diamond"),
    (target_ret_weights, f"目標報酬 {target_return*100:.1f}%", "orange", "cross"),
    (target_risk_weights, f"目標風險 {target_risk*100:.1f}%", "purple", "x")
]

for w, label, color, symbol in portfolio_markers:
    ret, risk, _ = portfolio_stats(w)
    fig.add_trace(go.Scatter(x=[risk*100], y=[ret*100],
                             mode='markers+text',
                             name=label,
                             text=[label],
                             textposition='middle right',
                             textfont=dict(size=11, color=color, weight='bold'),
                             marker=dict(size=16, color=color, symbol=symbol, line=dict(width=1, color='black')),
                             hovertemplate=f'{label}<br>報酬: {ret*100:.2f}%<br>風險: {risk*100:.2f}%<extra></extra>'))

# ---------- 自動調整座標軸範圍 ----------
all_x = []
all_y = []
if ef_risks:
    all_x.extend([r*100 for r in ef_risks])
    all_y.extend([r*100 for r in ef_rets])
for i in range(n_assets):
    all_x.append(vols[i]*100)
    all_y.append(returns[i]*100)
for w, _, _, _ in portfolio_markers:
    ret, risk, _ = portfolio_stats(w)
    all_x.append(risk*100)
    all_y.append(ret*100)

if all_x and all_y:
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = (x_max - x_min) * 0.1 if x_max != x_min else 5
    y_pad = (y_max - y_min) * 0.1 if y_max != y_min else 2
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]
else:
    x_range, y_range = None, None

# ---------- 圖表佈局 ----------
fig.update_layout(
    title="📈 效率前緣與最佳投資組合",
    xaxis_title="風險 (標準差 %)",
    yaxis_title="預期報酬 (%)",
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(240,240,240,0.9)',
        bordercolor='black',
        borderwidth=1.5,
        font=dict(size=13, color='black'),
        title=dict(text='📌 圖例說明', font=dict(size=14, color='black'))
    ),
    width=1000,
    height=650,
    hovermode='closest',
    plot_bgcolor='white',
    xaxis=dict(range=x_range, gridcolor='lightgray', showgrid=True),
    yaxis=dict(range=y_range, gridcolor='lightgray', showgrid=True)
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------- 超出範圍警告 -----------------------------
max_feasible_return = max(returns)
min_feasible_return = min(returns)
if target_return > max_feasible_return or target_return < min_feasible_return:
    st.warning(f"⚠️ 目標報酬率 {target_return*100:.1f}% 超出單一資產的報酬範圍 ({min_feasible_return*100:.1f}% ~ {max_feasible_return*100:.1f}%)，最佳化可能無法達成。")
min_vol = min(vols)
max_vol = max(vols)
if target_risk < min_vol or target_risk > 2 * max_vol:
    st.warning(f"⚠️ 目標風險 {target_risk*100:.1f}% 可能不切實際，建議範圍約 {min_vol*100:.1f}% ~ {2*max_vol*100:.1f}%。")

# ----------------------------- 使用提示 -----------------------------
with st.expander("ℹ️ 使用提示", expanded=False):
    st.markdown("""
    - **固定相關係數**：輸入單一 ρ 值（例如 1 或 -1），所有資產間的相關程度相同。
    - **自訂相關係數矩陣**：可直接修改表格中的數字，矩陣必須對稱且對角線為 1。
    - **允許放空**：權重可為負值，否則權重限制在 0~1 之間（不得放空）。
    - **效率前緣**：若無法收斂（例如 ρ = -1 且放空限制導致無解），曲線可能出現缺口，請調整參數。
    - **圖表互動**：可縮放、懸停查看數據點，點擊圖例可隱藏特定線條。
    - **目標報酬無法達成時**：系統會盡可能接近目標，但效率前緣上可能沒有對應點。請勾選「允許放空」或提高個股報酬率。
    """)
