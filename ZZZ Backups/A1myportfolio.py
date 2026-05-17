"""
投資組合最佳化器 - 完整版（新增個股統計 & 最佳化公式說明）
支援：收益率、相關性、協方差、權重最佳化、風險偏好、效用函式、有效前沿、資本市場線
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 設定頁面配置
st.set_page_config(page_title="智慧投資組合最佳化器", layout="wide", page_icon="📊")

# 設定中文字型支援（可選）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 資料載入模組 ====================
@st.cache_data
def load_data(tickers, start_date, end_date):
    """下載股票資料並計算收益率、協方差等"""
    try:
        # 下載資料
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        
        # 處理缺失值
        data = data.dropna(axis=1, how='any')
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # 計算收益率
        returns = data.pct_change().dropna()
        
        # 年化統計量 (252個交易日)
        annual_returns = returns.mean() * 252
        annual_cov = returns.cov() * 252
        annual_vol = np.sqrt(np.diag(annual_cov))
        
        # 計算相關性矩陣
        corr_matrix = returns.corr()
        
        return {
            'prices': data,
            'returns': returns,
            'annual_returns': annual_returns,
            'annual_cov': annual_cov,
            'annual_vol': annual_vol,
            'corr_matrix': corr_matrix
        }
    except Exception as e:
        st.error(f"資料載入失敗: {str(e)}")
        return None

# ==================== 最佳化模組 ====================
def calculate_portfolio_stats(weights, returns, annual_returns, annual_cov):
    """計算投資組合的收益率、波動率、夏普比率"""
    weights = np.array(weights)
    port_return = np.sum(annual_returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
    sharpe = (port_return - 0.02) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe

def optimize_max_sharpe(returns, annual_returns, annual_cov, risk_free_rate=0.02):
    """最大化夏普比率"""
    n_assets = len(annual_returns)
    
    def neg_sharpe(weights):
        port_return = np.sum(annual_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
        if port_vol == 0:
            return 0
        return -(port_return - risk_free_rate) / port_vol
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        ret, vol, sharpe = calculate_portfolio_stats(weights, returns, annual_returns, annual_cov)
        return weights, ret, vol, sharpe
    return None, 0, 0, 0

def optimize_min_volatility(returns, annual_returns, annual_cov):
    """最小化波動率"""
    n_assets = len(annual_returns)
    
    def portfolio_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(portfolio_vol, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        ret, vol, sharpe = calculate_portfolio_stats(weights, returns, annual_returns, annual_cov)
        return weights, ret, vol, sharpe
    return None, 0, 0, 0

def maximize_utility(returns, annual_returns, annual_cov, risk_aversion):
    """
    最大化效用函式: U = E[R] - 0.5 * λ * σ²
    λ: 風險厭惡係數 (越大越厭惡風險)
    """
    n_assets = len(annual_returns)
    
    def neg_utility(weights):
        port_return = np.sum(annual_returns * weights)
        port_var = np.dot(weights.T, np.dot(annual_cov, weights))
        utility = port_return - 0.5 * risk_aversion * port_var
        return -utility
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(neg_utility, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        ret = np.sum(annual_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
        utility = ret - 0.5 * risk_aversion * (vol ** 2)
        return weights, ret, vol, utility
    return None, 0, 0, 0

def generate_efficient_frontier(returns, annual_returns, annual_cov, n_points=50):
    """生成有效前沿上的點"""
    target_returns = np.linspace(annual_returns.min(), annual_returns.max() * 1.2, n_points)
    efficient_portfolios = []
    
    for target_ret in target_returns:
        n_assets = len(annual_returns)
        
        def portfolio_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(annual_returns * x) - target_ret}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_vol, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            vol = portfolio_vol(result.x)
            efficient_portfolios.append({
                'return': target_ret,
                'volatility': vol,
                'sharpe': (target_ret - 0.02) / vol if vol > 0 else 0
            })
    
    return pd.DataFrame(efficient_portfolios)

# ==================== 新增：個股統計模組 ====================
def calculate_individual_stats(returns, annual_returns, annual_vol, tickers):
    """計算個股的詳細統計量"""
    stats_df = pd.DataFrame(index=tickers)
    
    # 收益率統計
    stats_df['日收益率均值 (%)'] = returns.mean() * 100
    stats_df['日收益率標準差 (%)'] = returns.std() * 100
    stats_df['年化收益率 (%)'] = annual_returns * 100
    stats_df['年化波動率 (%)'] = annual_vol * 100
    stats_df['夏普比率'] = (annual_returns - 0.02) / annual_vol
    
    # 風險指標
    stats_df['最大回撤 (%)'] = [(returns[ticker].cumsum().cummax() - returns[ticker].cumsum()).min() * 100 
                                for ticker in tickers]
    stats_df['偏度'] = returns.skew()
    stats_df['峰度'] = returns.kurtosis()
    
    # 正收益機率
    stats_df['正收益機率 (%)'] = (returns > 0).mean() * 100
    
    return stats_df

def plot_individual_returns_distribution(returns, tickers):
    """繪製個股收益率分佈圖"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, ticker in enumerate(tickers[:4]):  # 最多顯示4個
        ax = axes[idx]
        ax.hist(returns[ticker], bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title(f'{ticker} 日收益率分佈')
        ax.set_xlabel('收益率')
        ax.set_ylabel('頻率')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ==================== 新增：最佳化公式說明模組 ====================
def display_formulas():
    """顯示最佳化公式的數學說明"""
    st.markdown("""
    ## 📐 投資組合最佳化數學原理
    
    ### 1️⃣ 基礎定義
    
    #### 投資組合收益率
    $$E[R_p] = \\sum_{i=1}^{n} w_i \\cdot E[R_i]$$
    
    其中：
    - $w_i$：資產 $i$ 的權重，滿足 $\\sum w_i = 1$ 且 $w_i \\geq 0$
    - $E[R_i]$：資產 $i$ 的預期收益率
    
    #### 投資組合風險（方差）
    $$\\sigma_p^2 = \\sum_{i=1}^{n} \\sum_{j=1}^{n} w_i w_j \\sigma_{ij} = \\mathbf{w}^T \\Sigma \\mathbf{w}$$
    
    其中 $\\sigma_{ij}$ 是資產 $i$ 和 $j$ 的協方差
    
    ---
    
    ### 2️⃣ 最佳化目標函式
    
    #### 目標1：最大化夏普比率（Max Sharpe Ratio）
    
    **夏普比率公式：**
    $$\\text{Sharpe} = \\frac{E[R_p] - R_f}{\\sigma_p}$$
    
    **最佳化問題：**
    $$\\max_{\\mathbf{w}} \\frac{\\mathbf{w}^T \\boldsymbol{\\mu} - R_f}{\\sqrt{\\mathbf{w}^T \\Sigma \\mathbf{w}}}$$
    $$\\text{s.t.} \\sum w_i = 1, \\quad w_i \\geq 0$$
    
    其中 $R_f$ 是無風險利率
    
    #### 目標2：最小化風險（Minimize Risk）
    
    **最佳化問題：**
    $$\\min_{\\mathbf{w}} \\sqrt{\\mathbf{w}^T \\Sigma \\mathbf{w}}$$
    $$\\text{s.t.} \\sum w_i = 1, \\quad w_i \\geq 0$$
    
    這是全域性最小方差組合（Global Minimum Variance Portfolio）
    
    #### 目標3：最大化效用函式（Utility Maximization）
    
    **效用函式（Quadratic Utility）：**
    $$U(\\mathbf{w}) = E[R_p] - \\frac{1}{2} \\lambda \\sigma_p^2$$
    
    **最佳化問題：**
    $$\\max_{\\mathbf{w}} \\left( \\mathbf{w}^T \\boldsymbol{\\mu} - \\frac{1}{2} \\lambda \\mathbf{w}^T \\Sigma \\mathbf{w} \\right)$$
    $$\\text{s.t.} \\sum w_i = 1, \\quad w_i \\geq 0$$
    
    其中 $\\lambda$ 是**風險厭惡係數**：
    - $\\lambda \\to 0$：風險中性，只關心收益
    - $\\lambda \\to \\infty$：極度風險厭惡，只關心風險
    
    ---
    
    ### 3️⃣ 有效前沿（Efficient Frontier）
    
    有效前沿是所有**帕累托最優**投資組合的集合，即在給定風險水平下收益最大，或給定收益水平下風險最小。
    
    **數學表達：**
    $$\\min_{\\mathbf{w}} \\sigma_p^2 \\quad \\text{s.t.} \\quad \\mathbf{w}^T \\boldsymbol{\\mu} = R_{target}, \\sum w_i = 1, w_i \\geq 0$$
    
    對於不同的目標收益率 $R_{target}$，求解上述問題得到一系列點，連線這些點即得有效前沿。
    
    ---
    
    ### 4️⃣ 資本市場線（Capital Market Line）
    
    當引入無風險資產後，最優投資組合位於**切點組合**（Tangency Portfolio）與無風險資產的連線上。
    
    **CAL方程：**
    $$E[R_c] = R_f + \\frac{E[R_t] - R_f}{\\sigma_t} \\times \\sigma_c$$
    
    其中 $(\\sigma_t, E[R_t])$ 是切點組合（最大夏普比率組合）
    
    ---
    
    ### 5️⃣ 數值求解方法
    
    本應用使用 **Sequential Least Squares Programming (SLSQP)** 演算法求解上述約束最佳化問題：
    
    1. **初始權重**：等權重分配 $w_i = 1/n$
    2. **約束條件**：
       - 等式約束：$\\sum w_i = 1$
       - 不等式約束：$w_i \\geq 0$（非負約束）
    3. **迭代最佳化**：透過梯度下降找到最優解
    
    ---
    
    ### 6️⃣ 關鍵統計量解釋
    
    | 統計量 | 公式 | 含義 |
    |--------|------|------|
    | **收益率** | $R_t = \\frac{P_t}{P_{t-1}} - 1$ | 資產價格變化率 |
    | **年化收益率** | $R_{ann} = (1 + R_{daily})^{252} - 1$ | 假設252個交易日 |
    | **波動率** | $\\sigma = \\sqrt{\\frac{1}{n-1} \\sum (R_i - \\bar{R})^2}$ | 收益率的離散程度 |
    | **協方差** | $\\sigma_{ij} = \\text{Cov}(R_i, R_j)$ | 兩個資產的協同變動 |
    | **相關係數** | $\\rho_{ij} = \\frac{\\sigma_{ij}}{\\sigma_i \\sigma_j}$ | 標準化後的協方差 |
    | **夏普比率** | $\\frac{R_p - R_f}{\\sigma_p}$ | 風險調整後收益 |
    
    ---
    
    ### 💡 使用建議
    
    1. **保守型投資者**：選擇最小化風險或高風險厭惡係數（λ > 10）
    2. **均衡型投資者**：選擇最大化夏普比率
    3. **進取型投資者**：選擇低風險厭惡係數（λ < 2）或自定義目標
    4. **分散化原則**：關注相關性矩陣，選擇低相關性的資產組合
    """)

# ==================== 視覺化模組 ====================
def plot_efficient_frontier(efficient_df, optimal_portfolio, individual_assets, tickers, risk_free_rate=0.02):
    """繪製有效前沿、CAL線、最優組合和單個資產"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. 有效前沿
    if not efficient_df.empty:
        ax.plot(efficient_df['volatility'], efficient_df['return'], 
                'b-', linewidth=2, label='Efficient Frontier')
    
    # 2. 單個資產
    for i, ticker in enumerate(tickers):
        ax.scatter(individual_assets['volatility'][i], individual_assets['return'][i], 
                  s=100, marker='o', alpha=0.7, label=f'{ticker}')
    
    # 3. 最優組合 - 修復：使用標準marker
    if optimal_portfolio:
        ax.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'], 
                  s=500, c='red', marker='*', zorder=5, label='Optimal Portfolio')
        ax.annotate(f'  Return: {optimal_portfolio["return"]:.2%}\n  Risk: {optimal_portfolio["volatility"]:.2%}',
                   (optimal_portfolio['volatility'], optimal_portfolio['return']), 
                   fontsize=9, ha='left', va='bottom')
    
    # 4. 資本市場線 (CAL)
    if optimal_portfolio and optimal_portfolio['volatility'] > 0:
        cal_x = np.array([0, optimal_portfolio['volatility'] * 1.5])
        cal_y = risk_free_rate + (optimal_portfolio['return'] - risk_free_rate) / optimal_portfolio['volatility'] * cal_x
        ax.plot(cal_x, cal_y, 'g--', linewidth=2, label=f'CAL (Sharpe = {optimal_portfolio["sharpe"]:.2f})')
        ax.scatter(0, risk_free_rate, s=150, c='darkgreen', marker='s', label='Risk-Free Asset', zorder=4)
    
    # 5. 格式化
    ax.set_xlabel('Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Expected Return', fontsize=12)
    ax.set_title('Mean-Variance Efficient Frontier & Capital Market Line', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1%}'))
    
    st.pyplot(fig)
    plt.close()

def plot_weights_pie(weights, tickers):
    """繪製權重餅圖"""
    fig, ax = plt.subplots(figsize=(6, 6))
    non_zero_weights = [(tickers[i], weights[i]) for i in range(len(weights)) if weights[i] > 0.001]
    
    if non_zero_weights:
        labels = [f'{ticker}\n({weight:.1%})' for ticker, weight in non_zero_weights]
        sizes = [weight for _, weight in non_zero_weights]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title('Asset Allocation', fontsize=12, fontweight='bold')
        st.pyplot(fig)
    plt.close()

def plot_correlation_heatmap(corr_matrix, tickers):
    """繪製相關性熱力圖"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(tickers, fontsize=10)
    
    # 新增數值標籤
    for i in range(min(len(tickers), len(corr_matrix))):
        for j in range(min(len(tickers), len(corr_matrix))):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.7 else "white", 
                          fontsize=9, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Correlation Coefficient', fraction=0.046, pad=0.04)
    ax.set_title('Asset Correlation Matrix', fontsize=12, fontweight='bold')
    st.pyplot(fig)
    plt.close()

# ==================== 主應用 ====================
def main():
    st.title("📊 智慧投資組合最佳化器")
    st.markdown("基於**均值-方差模型**與**效用理論**的投資組合最佳化工具")
    st.markdown("---")
    
    # 建立Tab頁面
    tab1, tab2, tab3 = st.tabs(["📈 投資組合最佳化", "📊 個股統計分析", "📐 最佳化公式說明"])
    
    # ==================== Tab 1: 投資組合最佳化 ====================
    with tab1:
        # 側邊欄配置
        with st.sidebar:
            st.header("⚙️ 引數配置")
            
            # 股票選擇
            default_tickers = "AAPL,MSFT,GOOGL,AMZN,TSLA"
            tickers_input = st.text_input("股票程式碼 (英文逗號分隔)", default_tickers)
            tickers = [t.strip().upper() for t in tickers_input.split(',')]
            
            # 日期範圍
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
            with col2:
                end_date = st.date_input("結束日期", pd.to_datetime("today"))
            
            # 風險引數
            st.subheader("📈 風險引數")
            risk_free_rate = st.number_input("無風險利率 (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.5) / 100
            
            # 最佳化目標選擇
            st.subheader("🎯 最佳化目標")
            objective_options = {
                "📈 最大化夏普比率 (Max Sharpe)": "sharpe",
                "🛡️ 最小化波動率 (Min Risk)": "min_risk",
                "⚖️ 自定義效用函式 (Custom Utility)": "utility"
            }
            objective = st.selectbox("選擇最佳化函式", list(objective_options.keys()))
            
            # 風險厭惡係數
            risk_aversion = 5.0
            if objective == "⚖️ 自定義效用函式 (Custom Utility)":
                risk_aversion = st.slider("風險厭惡係數 λ", 0.5, 20.0, 5.0, 0.5)
                st.caption("λ 越大 → 越厭惡風險 → 組合越保守")
            
            # 執行按鈕
            st.markdown("---")
            run_optimization = st.button("🚀 執行最佳化", type="primary", use_container_width=True)
        
        # 主介面 - 顯示資料概覽
        if run_optimization:
            with st.spinner("正在載入資料並最佳化..."):
                # 載入資料
                data_dict = load_data(tickers, start_date, end_date)
                
                if data_dict is None or data_dict['returns'].empty:
                    st.error("資料載入失敗，請檢查股票程式碼是否正確或網路連線")
                    return
                
                # 提取資料
                returns = data_dict['returns']
                annual_returns = data_dict['annual_returns'].values
                annual_cov = data_dict['annual_cov'].values
                corr_matrix = data_dict['corr_matrix'].values
                
                # 單個資產的表現
                individual_assets = pd.DataFrame({
                    'return': annual_returns,
                    'volatility': data_dict['annual_vol']
                }, index=tickers)
                individual_assets['sharpe'] = (individual_assets['return'] - risk_free_rate) / individual_assets['volatility']
                
                # 執行最佳化
                opt_type = objective_options[objective]
                
                if opt_type == "sharpe":
                    weights, ret, vol, sharpe = optimize_max_sharpe(returns, annual_returns, annual_cov, risk_free_rate)
                    optimal_portfolio = {'return': ret, 'volatility': vol, 'sharpe': sharpe}
                    utility = None
                    if weights is not None:
                        st.success(f"✅ 最佳化完成！最優夏普比率: {sharpe:.3f}")
                    
                elif opt_type == "min_risk":
                    weights, ret, vol, sharpe = optimize_min_volatility(returns, annual_returns, annual_cov)
                    optimal_portfolio = {'return': ret, 'volatility': vol, 'sharpe': sharpe}
                    utility = None
                    if weights is not None:
                        st.success(f"✅ 最佳化完成！最小波動率: {vol:.2%}")
                        
                else:  # utility
                    weights, ret, vol, utility = maximize_utility(returns, annual_returns, annual_cov, risk_aversion)
                    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
                    optimal_portfolio = {'return': ret, 'volatility': vol, 'sharpe': sharpe}
                    if weights is not None:
                        st.success(f"✅ 最佳化完成！效用值: {utility:.4f}")
                
                if weights is None:
                    st.error("最佳化失敗，請嘗試其他引數或股票組合")
                    return
                
                # 生成有效前沿
                efficient_frontier = generate_efficient_frontier(returns, annual_returns, annual_cov)
                
                # ========== 結果展示 ==========
                st.markdown("## 📊 最佳化結果")
                
                # 第一行：關鍵指標
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("預期年化收益率", f"{ret:.2%}")
                with col2:
                    st.metric("預期年化波動率", f"{vol:.2%}")
                with col3:
                    st.metric("夏普比率", f"{sharpe:.2f}")
                with col4:
                    if utility:
                        st.metric("效用值", f"{utility:.4f}")
                    else:
                        st.metric("風險調整收益", f"{sharpe * vol:.2%}")
                
                # 第二行：圖表展示
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("📈 有效前沿與資本市場線")
                    plot_efficient_frontier(efficient_frontier, optimal_portfolio, individual_assets, tickers, risk_free_rate)
                
                with col2:
                    st.subheader("🥧 資產配置")
                    plot_weights_pie(weights, tickers)
                    
                    # 顯示詳細權重表
                    st.markdown("**詳細權重表**")
                    weight_df = pd.DataFrame({
                        '資產': tickers,
                        '權重 (%)': [f"{w*100:.2f}%" for w in weights]
                    })
                    st.dataframe(weight_df, hide_index=True, use_container_width=True)
                
                # 第三行：相關性分析和回撤
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔗 資產相關性矩陣")
                    plot_correlation_heatmap(corr_matrix, tickers)
                
                with col2:
                    st.subheader("📋 各資產表現對比")
                    comparison_df = individual_assets.copy()
                    comparison_df.columns = ['年化收益率', '年化波動率', '夏普比率']
                    comparison_df = comparison_df.sort_values('夏普比率', ascending=False)
                    st.dataframe(comparison_df.style.format({
                        '年化收益率': '{:.2%}',
                        '年化波動率': '{:.2%}',
                        '夏普比率': '{:.3f}'
                    }), use_container_width=True)
                    
                    # 額外說明
                    st.info("""
                    **💡 解讀提示:**
                    - **夏普比率 > 1**: 優秀的風險調整後收益
                    - **相關係數 < 0.3**: 分散化效果好
                    - **風險厭惡係數 λ**: 越大組合越保守
                    """)
                
                # 第四行：收益率歷史走勢
                st.subheader("📉 資產價格歷史走勢")
                st.line_chart(data_dict['prices'])
        else:
            # 未執行最佳化時的佔位符
            st.info("👈 請在左側配置引數後點選「執行最佳化」")
            
            # 顯示示例說明
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ### 🎯 功能特性
                - ✅ **均值-方差最佳化**: 最大化夏普比率
                - ✅ **最小風險組合**: 全域性最小方差
                - ✅ **效用函式最佳化**: U = E[R] - 0.5·λ·σ²
                - ✅ **有效前沿繪製**: Markowitz 前沿
                - ✅ **資本市場線**: 包含無風險資產
                - ✅ **相關性分析**: 分散化效果評估
                """)
            
            with col2:
                st.markdown("""
                ### 📦 支援的資料
                - **資料來源**: Yahoo Finance (實時)
                - **資產型別**: 股票、ETF、指數
                - **時間範圍**: 可自定義起止日期
                - **最佳化約束**: 權重 ≥ 0, 和 = 1
                - **頻率**: 日度 → 年化 (252天)
                """)
            
            st.markdown("---")
            st.caption("💡 示例股票程式碼: AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, JNJ, WMT, NVDA, META")
    
    # ==================== Tab 2: 個股統計分析 ====================
    with tab2:
        st.header("📊 個股統計分析")
        st.markdown("計算各資產的基本統計量：收益率、標準差、協方差、相關性等")
        st.markdown("---")
        
        # 個股統計的股票選擇
        col1, col2 = st.columns([2, 1])
        with col1:
            stats_tickers_input = st.text_input("股票程式碼 (英文逗號分隔)", "AAPL,MSFT,GOOGL,AMZN,TSLA", key="stats_tickers")
            stats_tickers = [t.strip().upper() for t in stats_tickers_input.split(',')]
        
        with col2:
            stats_start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"), key="stats_start")
            stats_end_date = st.date_input("結束日期", pd.to_datetime("today"), key="stats_end")
        
        if st.button("📊 計算統計量", key="calc_stats"):
            with st.spinner("正在計算統計量..."):
                data_dict = load_data(stats_tickers, stats_start_date, stats_end_date)
                
                if data_dict is None or data_dict['returns'].empty:
                    st.error("資料載入失敗，請檢查股票程式碼")
                else:
                    returns = data_dict['returns']
                    annual_returns = data_dict['annual_returns']
                    annual_vol = data_dict['annual_vol']
                    
                    # 1. 基礎統計量表
                    st.subheader("📋 個股基礎統計量")
                    stats_df = calculate_individual_stats(returns, annual_returns, annual_vol, stats_tickers)
                    st.dataframe(stats_df.style.format({
                        '日收益率均值 (%)': '{:.4f}',
                        '日收益率標準差 (%)': '{:.4f}',
                        '年化收益率 (%)': '{:.2f}',
                        '年化波動率 (%)': '{:.2f}',
                        '夏普比率': '{:.3f}',
                        '最大回撤 (%)': '{:.2f}',
                        '偏度': '{:.3f}',
                        '峰度': '{:.3f}',
                        '正收益機率 (%)': '{:.1f}'
                    }), use_container_width=True)
                    
                    # 2. 協方差矩陣
                    st.subheader("📐 協方差矩陣（年化）")
                    cov_matrix = data_dict['annual_cov']
                    st.dataframe(cov_matrix.style.format("{:.6f}"), use_container_width=True)
                    
                    # 3. 相關性矩陣
                    st.subheader("🔗 相關性矩陣")
                    corr_matrix = data_dict['corr_matrix']
                    st.dataframe(corr_matrix.style.format("{:.4f}").background_gradient(cmap='coolwarm', axis=None), 
                               use_container_width=True)
                    
                    # 4. 收益率分佈圖
                    st.subheader("📈 收益率分佈圖")
                    if len(stats_tickers) >= 1:
                        plot_individual_returns_distribution(returns, stats_tickers)
                    
                    # 5. 累計收益對比
                    st.subheader("📉 累計收益對比（標準化）")
                    cumulative_returns = (1 + returns).cumprod()
                    cumulative_returns_normalized = cumulative_returns / cumulative_returns.iloc[0] * 100
                    st.line_chart(cumulative_returns_normalized)
                    
                    # 6. 滾動波動率
                    st.subheader("📊 滾動30天波動率")
                    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
                    st.line_chart(rolling_vol * 100)
                    
                    # 7. 解釋說明
                    with st.expander("📖 統計量解讀說明"):
                        st.markdown("""
                        **統計量含義：**
                        - **偏度 (Skewness)**：衡量收益率分佈的不對稱性
                          - 正偏度：右尾長，表示有較大正收益的可能性
                          - 負偏度：左尾長，表示有較大負收益的風險
                        - **峰度 (Kurtosis)**：衡量收益率分佈的尾部風險
                          - 高峰度：極端值出現機率高（肥尾風險）
                          - 低峰度：收益分佈更集中
                        - **最大回撤**：歷史最高點到最低點的最大跌幅
                        - **正收益機率**：投資獲利的機率
                        """)
    
    # ==================== Tab 3: 最佳化公式說明 ====================
    with tab3:
        display_formulas()

if __name__ == "__main__":
    main()