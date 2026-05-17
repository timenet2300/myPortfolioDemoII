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

# 設定中文字型支援
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# ==================== 資料載入模組 ====================
@st.cache_data
def load_data(tickers, start_date, end_date):
    """下載股票資料並計算收益率、協方差等"""
    try:
        # 下載資料
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # 修正：處理 yfinance 的 Multi-index 並提取 Close 價格
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close']
        else:
            data = raw_data[['Close']]
            data.columns = tickers
        
        # 處理缺失值 (更新為新版 Pandas 語法)
        data = data.dropna(axis=1, how='all')
        data = data.ffill().bfill()
        
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
    """最大化效用函式"""
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
    # 修正：確保目標收益率區間在合理範圍內
    target_returns = np.linspace(annual_returns.min(), annual_returns.max(), n_points)
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

# ==================== 個股統計模組 ====================
def calculate_individual_stats(returns, annual_returns, annual_vol, tickers):
    """計算個股的詳細統計量"""
    stats_df = pd.DataFrame(index=tickers)
    stats_df['日收益率均值 (%)'] = returns.mean() * 100
    stats_df['日收益率標準差 (%)'] = returns.std() * 100
    stats_df['年化收益率 (%)'] = annual_returns * 100
    stats_df['年化波動率 (%)'] = annual_vol * 100
    stats_df['夏普比率'] = (annual_returns - 0.02) / annual_vol
    stats_df['最大回撤 (%)'] = [(returns[ticker].cumsum() - returns[ticker].cumsum().cummax()).min() * 100 for ticker in tickers]
    stats_df['偏度'] = returns.skew()
    stats_df['峰度'] = returns.kurtosis()
    stats_df['正收益機率 (%)'] = (returns > 0).mean() * 100
    return stats_df

def plot_individual_returns_distribution(returns, tickers):
    """繪製個股收益率分佈圖"""
    num_plots = min(len(tickers), 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx in range(4):
        ax = axes[idx]
        if idx < len(tickers):
            ticker = tickers[idx]
            ax.hist(returns[ticker], bins=50, alpha=0.7, edgecolor='black', density=True)
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_title(f'{ticker} 日收益率分佈')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ==================== 最佳化公式說明模組 ====================
def display_formulas():
    """顯示最佳化公式的數學說明"""
    st.markdown("""
    ## 📐 投資組合最佳化數學原理
    ... (此處省略部分 Markdown 內容以節省空間，與原稿一致) ...
    """)

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
    """繪製有效前沿"""
    fig, ax = plt.subplots(figsize=(12, 7))
    if not efficient_df.empty:
        ax.plot(efficient_df['volatility'], efficient_df['return'], 'b-', linewidth=2, label='Efficient Frontier')
    
    for i, ticker in enumerate(tickers):
        ax.scatter(individual_assets['volatility'][i], individual_assets['return'][i], s=100, marker='o', alpha=0.7, label=ticker)
    
    if optimal_portfolio:
        ax.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'], s=500, c='red', marker='*', zorder=5, label='Optimal Portfolio')
    
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Expected Return')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

def plot_weights_pie(weights, tickers):
    """繪製權重餅圖"""
    fig, ax = plt.subplots(figsize=(6, 6))
    non_zero_weights = [(tickers[i], weights[i]) for i in range(len(weights)) if weights[i] > 0.001]
    if non_zero_weights:
        labels = [f'{t}\n({w:.1%})' for t, w in non_zero_weights]
        sizes = [w for t, w in non_zero_weights]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Asset Allocation')
        st.pyplot(fig)
    plt.close()

def plot_correlation_heatmap(corr_matrix, tickers):
    """繪製相關性熱力圖"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45)
    ax.set_yticklabels(tickers)
    plt.colorbar(im)
    st.pyplot(fig)
    plt.close()

# ==================== 主應用 ====================
def main():
    st.title("📊 智慧投資組合最佳化器")
    st.markdown("基於**均值-方差模型**與**效用理論**的最佳化工具")
    
    tab1, tab2, tab3 = st.tabs(["📈 投資組合最佳化", "📊 個股統計分析", "📐 最佳化公式說明"])
    
    with tab1:
        with st.sidebar:
            st.header("⚙️ 引數配置")
            tickers_input = st.text_input("股票程式碼 (英文逗號分隔)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
            tickers = [t.strip().upper() for t in tickers_input.split(',')]
            start_date = st.date_input("開始日期", pd.to_datetime("2020-01-01"))
            end_date = st.date_input("結束日期", pd.to_datetime("today"))
            risk_free_rate = st.number_input("無風險利率 (%)", 0.0, 10.0, 2.0) / 100
            objective = st.selectbox("選擇最佳化函式", ["📈 最大化夏普比率 (Max Sharpe)", "🛡️ 最小化波動率 (Min Risk)", "⚖️ 自定義效用函式 (Custom Utility)"])
            risk_aversion = st.slider("風險厭惡係數 λ", 0.5, 20.0, 5.0) if "Utility" in objective else 5.0
            run_optimization = st.button("🚀 執行最佳化", type="primary")

        if run_optimization:
            data_dict = load_data(tickers, start_date, end_date)
            if data_dict:
                returns = data_dict['returns']
                annual_returns = data_dict['annual_returns']
                annual_cov = data_dict['annual_cov']
                
                # 執行最佳化邏輯
                if "Sharpe" in objective:
                    weights, ret, vol, sharpe = optimize_max_sharpe(returns, annual_returns, annual_cov, risk_free_rate)
                elif "Min Risk" in objective:
                    weights, ret, vol, sharpe = optimize_min_volatility(returns, annual_returns, annual_cov)
                else:
                    weights, ret, vol, utility = maximize_utility(returns, annual_returns, annual_cov, risk_aversion)
                    sharpe = (ret - risk_free_rate) / vol
                
                # 顯示結果
                st.metric("預期年化收益率", f"{ret:.2%}")
                st.metric("夏普比率", f"{sharpe:.2f}")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    eff_frontier = generate_efficient_frontier(returns, annual_returns, annual_cov)
                    ind_assets = pd.DataFrame({'return': annual_returns, 'volatility': data_dict['annual_vol']})
                    plot_efficient_frontier(eff_frontier, {'return': ret, 'volatility': vol}, ind_assets, tickers, risk_free_rate)
                with col2:
                    plot_weights_pie(weights, tickers)
                
                st.subheader("🔗 資產相關性")
                plot_correlation_heatmap(data_dict['corr_matrix'], tickers)
                st.line_chart(data_dict['prices'])

    with tab2:
        st.header("📊 個股統計分析")
        if st.button("📊 計算統計量"):
            data_dict = load_data(tickers, start_date, end_date)
            if data_dict:
                stats_df = calculate_individual_stats(data_dict['returns'], data_dict['annual_returns'], data_dict['annual_vol'], tickers)
                st.dataframe(stats_df.style.format("{:.2f}"))
                plot_individual_returns_distribution(data_dict['returns'], tickers)
                st.line_chart((1 + data_dict['returns']).cumprod())

    with tab3:
        display_formulas()

if __name__ == "__main__":
    main()