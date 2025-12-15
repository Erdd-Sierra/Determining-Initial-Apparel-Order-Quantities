"""
サプライチェーン・マネジメント最適化ツール
アパレル商品の発注において、香港工場と中国工場のそれぞれでどれくらい初期生産すべきかを
モンテカルロシミュレーションを用いて決定するStreamlitアプリケーション
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pulp import LpMinimize, LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
import scipy.stats as stats

# ============================================================================
# 設定データ（CONFIG）
# ============================================================================

# 商品データ
PRODUCTS = [
    {"Style": "Gail",      "Price": 110, "Mean": 1017, "StdDev": 388,   "UnderageCost": 26.40, "OverageCost": 8.80},
    {"Style": "Isis",      "Price": 99,  "Mean": 1042, "StdDev": 646,   "UnderageCost": 23.76, "OverageCost": 7.92},
    {"Style": "Entice",    "Price": 80,  "Mean": 1358, "StdDev": 496,   "UnderageCost": 19.20, "OverageCost": 6.40},
    {"Style": "Assault",   "Price": 90,  "Mean": 2525, "StdDev": 680,   "UnderageCost": 21.60, "OverageCost": 7.20},
    {"Style": "Teri",      "Price": 123, "Mean": 1100, "StdDev": 762,   "UnderageCost": 29.52, "OverageCost": 9.84},
    {"Style": "Electra",   "Price": 173, "Mean": 2150, "StdDev": 807,   "UnderageCost": 41.52, "OverageCost": 13.84},
    {"Style": "Stephanie", "Price": 133, "Mean": 1113, "StdDev": 1048,  "UnderageCost": 31.92, "OverageCost": 10.64},
    {"Style": "Seduced",   "Price": 73,  "Mean": 4017, "StdDev": 1113,  "UnderageCost": 17.52, "OverageCost": 5.84},
    {"Style": "Anita",     "Price": 93,  "Mean": 3296, "StdDev": 2094,  "UnderageCost": 22.32, "OverageCost": 7.44},
    {"Style": "Daphne",    "Price": 148, "Mean": 2383, "StdDev": 1394,  "UnderageCost": 35.52, "OverageCost": 11.84},
]

# 工場設定
FACTORY_CONFIG = {
    "香港工場": {
        "MinLotSize": 600,  # 最小ロット数
    },
    "中国工場": {
        "MinLotSize": 1200,  # 最小ロット数
    }
}

# その他の設定
CONFIG = {
    "TotalOrder": 10000,  # 合計発注量（厳密に10000）
    "DefaultSimulations": 100,  # デフォルトのシミュレーション回数
}

# ============================================================================
# ヘルパー関数
# ============================================================================

def generate_triangular_demand(mean, stddev):
    """
    三角分布に従って需要を生成
    
    パラメータ:
        mean: 平均需要 (μ)
        stddev: 標準偏差 (σ)
    
    戻り値:
        生成された需要値
    """
    # 範囲: [μ - σ, μ + σ]
    left = max(0, mean - stddev)
    right = mean + stddev
    mode = mean  # 頂点
    
    # 三角分布からサンプリング
    # c: モード（最頻値）の相対位置 [0, 1]
    # c = (mode - left) / (right - left)
    if right > left:
        c = (mode - left) / (right - left)
    else:
        c = 0.5  # 範囲が0の場合のフォールバック
    
    demand = stats.triang.rvs(
        loc=left,
        scale=right - left,
        c=c
    )
    
    return max(0, demand)  # 負の値にならないように


def solve_optimization_problem(demands, products, config, min_lot_size):
    """
    最適化問題を解く（混合整数計画法）
    
    目的関数: 想定利益（価格 × 実際の売上数量）の最大化
    
    パラメータ:
        demands: 各商品の需要リスト
        products: 商品データのリスト
        config: その他の設定
        min_lot_size: 最小ロット数
    
    戻り値:
        (最適発注量リスト, 総利益額)
    """
    n_products = len(products)
    
    # 最適化問題の定義
    prob = LpProblem("OrderOptimization", LpMaximize)
    
    # ========================================================================
    # 決定変数の定義
    # ========================================================================
    
    # バイナリ変数: 各商品を発注するかどうか
    # y[i] = 1: 商品iを発注する, y[i] = 0: 商品iを発注しない
    y = [LpVariable(f"y_{i}", cat="Binary") for i in range(n_products)]
    
    # 連続変数: 各商品の発注量
    # 制約: x[i] = 0 または x[i] >= min_lot_size
    # これを実現するためにBig-M法を使用
    M = config["TotalOrder"]  # 十分大きな数（合計発注量を上限とする）
    x = [LpVariable(f"x_{i}", lowBound=0, cat="Continuous") 
         for i in range(n_products)]
    
    # 補助変数: 不足量と余剰量
    shortage = [LpVariable(f"shortage_{i}", lowBound=0, cat="Continuous") 
                for i in range(n_products)]
    surplus = [LpVariable(f"surplus_{i}", lowBound=0, cat="Continuous") 
               for i in range(n_products)]
    
    # ========================================================================
    # 制約条件: 「0または最小ロット数以上」の実装（Big-M法）
    # ========================================================================
    # 各商品iについて、以下の2つの制約を追加：
    # 1. x[i] >= min_lot_size * y[i]
    #    - y[i] = 1 の場合: x[i] >= min_lot_size（最小ロット数以上）
    #    - y[i] = 0 の場合: x[i] >= 0（非負制約のみ）
    # 2. x[i] <= M * y[i]
    #    - y[i] = 1 の場合: x[i] <= M（上限制約）
    #    - y[i] = 0 の場合: x[i] <= 0（つまり x[i] = 0）
    # 
    # これにより、x[i] = 0 または x[i] >= min_lot_size が保証される
    for i in range(n_products):
        prob += x[i] >= min_lot_size * y[i], f"MinLot_{i}"
        prob += x[i] <= M * y[i], f"MaxLot_{i}"
    
    # 制約条件: 不足量と余剰量の定義
    # x - demand = shortage - surplus
    # つまり: shortage = max(0, demand - x), surplus = max(0, x - demand)
    for i in range(n_products):
        prob += x[i] - demands[i] == shortage[i] - surplus[i], f"Balance_{i}"
    
    # 目的関数: 総利益の最大化
    # 実際の売上数量 = min(需要, 発注量) = 需要 - 不足量 = demands[i] - shortage[i]
    # 利益 = 価格 × 実際の売上数量 = Price[i] × (demands[i] - shortage[i])
    prob += lpSum([products[i]["Price"] * (demands[i] - shortage[i])
                   for i in range(n_products)]), "TotalProfit"
    
    # 制約条件: 合計発注量は厳密に10000
    prob += lpSum(x) == config["TotalOrder"], "TotalOrder"
    
    # 最適化実行
    prob.solve()
    
    if LpStatus[prob.status] == "Optimal":
        optimal_orders = [x[i].varValue if x[i].varValue is not None else 0 
                         for i in range(n_products)]
        total_profit = prob.objective.value()
        return optimal_orders, total_profit
    else:
        return None, None


def run_monte_carlo_simulation(products, config, min_lot_size, n_simulations):
    """
    モンテカルロシミュレーションを実行
    
    パラメータ:
        products: 商品データのリスト
        config: その他の設定
        min_lot_size: 最小ロット数
        n_simulations: シミュレーション回数
    
    戻り値:
        (推奨発注量リスト, 利益額リスト, 全シミュレーション結果)
    """
    n_products = len(products)
    all_orders = []
    all_profits = []
    all_results = []
    
    for sim in range(n_simulations):
        # 需要生成
        demands = [generate_triangular_demand(p["Mean"], p["StdDev"]) 
                   for p in products]
        
        # 最適化問題を解く
        optimal_orders, total_profit = solve_optimization_problem(
            demands, products, config, min_lot_size
        )
        
        if optimal_orders is not None:
            all_orders.append(optimal_orders)
            all_profits.append(total_profit)
            all_results.append({
                "simulation": sim + 1,
                "demands": demands,
                "orders": optimal_orders,
                "profit": total_profit
            })
    
    if len(all_orders) == 0:
        return None, None, None
    
    # 推奨発注量: 平均値
    recommended_orders = np.mean(all_orders, axis=0).tolist()
    
    return recommended_orders, all_profits, all_results


# ============================================================================
# Streamlitアプリケーション
# ============================================================================

def main():
    st.set_page_config(page_title="SCM最適化ツール", layout="wide")
    st.title("アパレル商品発注最適化ツール")
    st.markdown("---")
    
    # サイドバー: パラメータ設定
    with st.sidebar:
        st.header("パラメータ設定")
        n_simulations = st.number_input(
            "シミュレーション回数",
            min_value=10,
            max_value=1000,
            value=CONFIG["DefaultSimulations"],
            step=10
        )
        st.info(f"合計発注量: {CONFIG['TotalOrder']:,}（固定）")
    
    # タブの作成
    tab1, tab2 = st.tabs(["📊 香港工場", "📊 中国工場"])
    
    # 香港工場のタブ
    with tab1:
        st.header("香港工場の分析")
        factory_name = "香港工場"
        min_lot_size = FACTORY_CONFIG[factory_name]["MinLotSize"]
        st.info(f"目的関数: 想定利益（価格 × 実際の売上数量）の最大化 | 最小ロット数: {min_lot_size}")
        
        if st.button("シミュレーション実行", key="hk_button"):
            with st.spinner("シミュレーション実行中..."):
                recommended_orders, profits, results = run_monte_carlo_simulation(
                    PRODUCTS, CONFIG, min_lot_size, n_simulations
                )
                
                if recommended_orders is not None:
                    st.success(f"シミュレーション完了 ({len(profits)}回成功)")
                    
                    # 結果の表示
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("推奨発注量")
                        df_orders = pd.DataFrame({
                            "商品": [p["Style"] for p in PRODUCTS],
                            "推奨発注量": [round(x, 2) for x in recommended_orders],
                            "平均需要": [p["Mean"] for p in PRODUCTS],
                            "標準偏差": [p["StdDev"] for p in PRODUCTS]
                        })
                        st.dataframe(df_orders, use_container_width=True)
                        total_order = sum(recommended_orders)
                        st.metric("合計発注量", f"{total_order:,.0f}", 
                                 delta=f"{total_order - CONFIG['TotalOrder']:,.0f}" if abs(total_order - CONFIG['TotalOrder']) > 0.01 else "目標達成")
                    
                    with col2:
                        st.subheader("利益統計")
                        if profits:
                            avg_profit = np.mean(profits)
                            min_profit = np.min(profits)
                            max_profit = np.max(profits)
                            std_profit = np.std(profits)
                            
                            st.metric("平均利益", f"${avg_profit:,.2f}")
                            st.metric("最小利益", f"${min_profit:,.2f}")
                            st.metric("最大利益", f"${max_profit:,.2f}")
                            st.metric("標準偏差", f"${std_profit:,.2f}")
                    
                    # 利益分布のヒストグラム
                    st.subheader("利益分布")
                    fig_hist = px.histogram(
                        x=profits,
                        nbins=30,
                        labels={"x": "利益額 ($)", "y": "頻度"},
                        title="利益額の分布"
                    )
                    fig_hist.update_layout(showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # 発注量の可視化
                    st.subheader("発注量の比較")
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        name="推奨発注量",
                        x=[p["Style"] for p in PRODUCTS],
                        y=recommended_orders,
                        marker_color="steelblue"
                    ))
                    fig_bar.add_trace(go.Bar(
                        name="平均需要",
                        x=[p["Style"] for p in PRODUCTS],
                        y=[p["Mean"] for p in PRODUCTS],
                        marker_color="lightcoral"
                    ))
                    fig_bar.update_layout(
                        title="推奨発注量 vs 平均需要",
                        xaxis_title="商品",
                        yaxis_title="数量",
                        barmode="group"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.error("シミュレーションに失敗しました。")
    
    # 中国工場のタブ
    with tab2:
        st.header("中国工場の分析")
        factory_name = "中国工場"
        min_lot_size = FACTORY_CONFIG[factory_name]["MinLotSize"]
        st.info(f"目的関数: 想定利益（価格 × 実際の売上数量）の最大化 | 最小ロット数: {min_lot_size}")
        
        if st.button("シミュレーション実行", key="cn_button"):
            with st.spinner("シミュレーション実行中..."):
                recommended_orders, profits, results = run_monte_carlo_simulation(
                    PRODUCTS, CONFIG, min_lot_size, n_simulations
                )
                
                if recommended_orders is not None:
                    st.success(f"シミュレーション完了 ({len(profits)}回成功)")
                    
                    # 結果の表示
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("推奨発注量")
                        df_orders = pd.DataFrame({
                            "商品": [p["Style"] for p in PRODUCTS],
                            "推奨発注量": [round(x, 2) for x in recommended_orders],
                            "平均需要": [p["Mean"] for p in PRODUCTS],
                            "標準偏差": [p["StdDev"] for p in PRODUCTS]
                        })
                        st.dataframe(df_orders, use_container_width=True)
                        total_order = sum(recommended_orders)
                        st.metric("合計発注量", f"{total_order:,.0f}", 
                                 delta=f"{total_order - CONFIG['TotalOrder']:,.0f}" if abs(total_order - CONFIG['TotalOrder']) > 0.01 else "目標達成")
                    
                    with col2:
                        st.subheader("利益統計")
                        if profits:
                            avg_profit = np.mean(profits)
                            min_profit = np.min(profits)
                            max_profit = np.max(profits)
                            std_profit = np.std(profits)
                            
                            st.metric("平均利益", f"${avg_profit:,.2f}")
                            st.metric("最小利益", f"${min_profit:,.2f}")
                            st.metric("最大利益", f"${max_profit:,.2f}")
                            st.metric("標準偏差", f"${std_profit:,.2f}")
                    
                    # 利益分布のヒストグラム
                    st.subheader("利益分布")
                    fig_hist = px.histogram(
                        x=profits,
                        nbins=30,
                        labels={"x": "利益額 ($)", "y": "頻度"},
                        title="利益額の分布"
                    )
                    fig_hist.update_layout(showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # 発注量の可視化
                    st.subheader("発注量の比較")
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        name="推奨発注量",
                        x=[p["Style"] for p in PRODUCTS],
                        y=recommended_orders,
                        marker_color="steelblue"
                    ))
                    fig_bar.add_trace(go.Bar(
                        name="平均需要",
                        x=[p["Style"] for p in PRODUCTS],
                        y=[p["Mean"] for p in PRODUCTS],
                        marker_color="lightcoral"
                    ))
                    fig_bar.update_layout(
                        title="推奨発注量 vs 平均需要",
                        xaxis_title="商品",
                        yaxis_title="数量",
                        barmode="group"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.error("シミュレーションに失敗しました。")


if __name__ == "__main__":
    main()

