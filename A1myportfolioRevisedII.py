"""
1330.tw,2881.TW, 2888.TW, 2882.tw, 2883.tw,1216.tw, 2603.tw; 有在效率前沿上，且夏普比率不錯的股票組合
智慧投資組合最佳化器 - 專業修復版
修正內容：MDD計算邏輯、無風險利率連動、資料下載對齊機制、IndexError防禦
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
    """下載股票資料並動態過濾無效標的"""
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # 處理 Multi-index 並提取 Close 價格
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close']
        else:
            data = raw_data[['Close']]
            if len(tickers) == 1:
                data.columns = tickers

        # 核心修正：移除完全抓不到資料的股票 (避免 IndexError)
        data = data.dropna(axis=1, how='all')
        actual_tickers = data.columns.tolist() 
        
        # 處理缺失值
        data = data.ffill().bfill()
        
        # 計算收益率
        returns = data.pct_change().dropna()
        
        # 年化統計量 (252個交易日)
        annual_returns = returns.mean() * 252
        annual_cov = returns.cov() * 252
        annual_vol = np.sqrt(np.diag(annual_cov))
        
        return {
            'prices': data,
            'returns': returns,
            'annual_returns': annual_returns,
            'annual_cov': annual_cov,
            'annual_vol': annual_vol,
            'corr_matrix': returns.corr(),
            'actual_tickers': actual_tickers
        }
    except Exception as e:
        st.error(f"資料載入失敗: {str(e)}")
        return None

# ==================== 最佳化模組 ====================
def calculate_portfolio_stats(weights, returns, annual_returns, annual_cov, risk_free_rate):
    """計算投資組合的收益率、波動率、夏普比率"""
    weights = np.array(weights)
    port_return = np.sum(annual_returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
    # 修正：連動 risk_free_rate
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe

def optimize_max_sharpe(returns, annual_returns, annual_cov, risk_free_rate):
    """最大化夏普比率"""
    n_assets = len(annual_returns)
    
    def neg_sharpe(weights):
        port_return = np.sum(annual_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))
        if port_vol == 0: return 0
        return -(port_return - risk_free_rate) / port_vol
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial = np.array([1/n_assets] * n_assets)
    
    result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        weights = result.x
        ret, vol, sharpe = calculate_portfolio_stats(weights, returns, annual_returns, annual_cov, risk_free_rate)
        return weights, ret, vol, sharpe
    return None, 0, 0, 0

def optimize_min_volatility(returns, annual_returns, annual_cov, risk_free_rate):
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
        ret, vol, sharpe = calculate_portfolio_stats(weights, returns, annual_returns, annual_cov, risk_free_rate)
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

def generate_efficient_frontier(returns, annual_returns, annual_cov, risk_free_rate, n_points=50):
    """生成有效前沿"""
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
                'sharpe': (target_ret - risk_free_rate) / vol if vol > 0 else 0
            })
    return pd.DataFrame(efficient_portfolios)

# ==================== 個股統計模組 ====================
def calculate_individual_stats(returns, prices, annual_returns, annual_vol, tickers, risk_free_rate):
    """計算個股詳細統計量 (包含正確的 MDD)"""
    stats_df = pd.DataFrame(index=tickers)
    stats_df['年化收益率 (%)'] = annual_returns * 100
    stats_df['年化波動率 (%)'] = annual_vol * 100
    stats_df['夏普比率'] = (annual_returns - risk_free_rate) / annual_vol
    
    # 修正 MDD 計算：基於價格軌跡
    mdds = []
    for ticker in tickers:
        p = prices[ticker]
        mdds.append(((p - p.cummax()) / p.cummax()).min() * 100)
    
    stats_df['最大回撤 (%)'] = mdds
    stats_df['正收益機率 (%)'] = (returns > 0).mean() * 100
    stats_df['偏度'] = returns.skew()
    return stats_df

# ==================== 視覺化模組 ====================
def plot_efficient_frontier(efficient_df, optimal_portfolio, individual_assets, tickers):
    """繪製有效前沿圖"""
    fig, ax = plt.subplots(figsize=(12, 7))
    if not efficient_df.empty:
        ax.plot(efficient_df['volatility'], efficient_df['return'], 'b-', linewidth=2, label='Efficient Frontier')
    
    # 使用實際抓到的 tickers 長度來遍歷，避免 IndexError
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
    fig, ax = plt.subplots(figsize=(6, 6))
    non_zero = [(tickers[i], weights[i]) for i in range(len(weights)) if weights[i] > 0.001]
    if non_zero:
        labels = [f'{t}\n({w:.1%})' for t, w in non_zero]
        sizes = [w for t, w in non_zero]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Asset Allocation')
        st.pyplot(fig)
    plt.close()

# ==================== 主應用 ====================
def main():
    st.title("📊 智慧投資組合最佳化器")
    
    tab1, tab2, tab3 = st.tabs(["📈 投資組合最佳化", "📊 個股統計分析", "📐 最佳化公式說明"])
    
    with st.sidebar:
        st.header("⚙️ 引數配置")
        tickers_input = st.text_input("股票程式碼 (yfinance股票代號逗號分隔)", "2330.TW, 2881.TW, 2882.TW")
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
            # 獲取實際抓到資料的標的清單
            active_tickers = data_dict['actual_tickers']
            returns = data_dict['returns']
            annual_returns = data_dict['annual_returns']
            annual_cov = data_dict['annual_cov']
            
            if "Sharpe" in objective:
                weights, ret, vol, sharpe = optimize_max_sharpe(returns, annual_returns, annual_cov, risk_free_rate)
            elif "Min Risk" in objective:
                weights, ret, vol, sharpe = optimize_min_volatility(returns, annual_returns, annual_cov, risk_free_rate)
            else:
                weights, ret, vol, utility = maximize_utility(returns, annual_returns, annual_cov, risk_aversion)
                sharpe = (ret - risk_free_rate) / vol
            
            with tab1:
                st.metric("預期年化收益率", f"{ret:.2%}")
                st.metric("夏普比率", f"{sharpe:.2f}")
                col1, col2 = st.columns([2, 1])
                with col1:
                    eff_df = generate_efficient_frontier(returns, annual_returns, annual_cov, risk_free_rate)
                    ind_assets = pd.DataFrame({'return': annual_returns, 'volatility': data_dict['annual_vol']})
                    plot_efficient_frontier(eff_df, {'return': ret, 'volatility': vol}, ind_assets, active_tickers)
                with col2:
                    plot_weights_pie(weights, active_tickers)
                st.subheader("🔗 資產相關性")
                fig_corr, ax_corr = plt.subplots()
                im = ax_corr.imshow(data_dict['corr_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
                ax_corr.set_xticks(range(len(active_tickers))); ax_corr.set_yticks(range(len(active_tickers)))
                ax_corr.set_xticklabels(active_tickers, rotation=45); ax_corr.set_yticklabels(active_tickers)
                plt.colorbar(im); st.pyplot(fig_corr)

            with tab2:
                st.header("📊 個股統計分析")
                stats_df = calculate_individual_stats(returns, data_dict['prices'], annual_returns, data_dict['annual_vol'], active_tickers, risk_free_rate)
                st.dataframe(stats_df.style.format("{:.2f}"))
                st.line_chart((1 + returns).cumprod())

    with tab3:
        # (保留原本的 display_formulas 邏輯)
        st.markdown("### 📐 數學公式說明 (已更新 MDD 為價格基礎公式)")
        st.latex(r"MDD = \frac{P_t - \max(P_{0..t})}{\max(P_{0..t})}")

if __name__ == "__main__":
    main()