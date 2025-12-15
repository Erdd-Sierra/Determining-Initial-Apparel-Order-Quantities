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
    {"Style": "Gail",      "Price": 110, "Mean": 508.5, "StdDev": 388,   "UnderageCost": 26.40, "OverageCost": 8.80},
    {"Style": "Isis",      "Price": 99,  "Mean": 521,   "StdDev": 646,   "UnderageCost": 23.76, "OverageCost": 7.92},
    {"Style": "Entice",    "Price": 80,  "Mean": 679,   "StdDev": 496,   "UnderageCost": 19.20, "OverageCost": 6.40},
    {"Style": "Assault",   "Price": 90,  "Mean": 1262.5,"StdDev": 680,   "UnderageCost": 21.60, "OverageCost": 7.20},
    {"Style": "Teri",      "Price": 123, "Mean": 550,   "StdDev": 762,   "UnderageCost": 29.52, "OverageCost": 9.84},
    {"Style": "Electra",   "Price": 173, "Mean": 1075,  "StdDev": 807,   "UnderageCost": 41.52, "OverageCost": 13.84},
    {"Style": "Stephanie", "Price": 133, "Mean": 556.5, "StdDev": 1048,  "UnderageCost": 31.92, "OverageCost": 10.64},
    {"Style": "Seduced",   "Price": 73,  "Mean": 2008.5,"StdDev": 1113,  "UnderageCost": 17.52, "OverageCost": 5.84},
    {"Style": "Anita",     "Price": 93,  "Mean": 1648,  "StdDev": 2094,  "UnderageCost": 22.32, "OverageCost": 7.44},
    {"Style": "Daphne",    "Price": 148, "Mean": 1191.5,"StdDev": 1394,  "UnderageCost": 35.52, "OverageCost": 11.84},
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
    
    目的関数:
        実売上 = min(発注量, 需要)
        利益 = 実売上 × 卸売価格 − UnderageCost × 不足量 − OverageCost × 余剰量
        を最大化する
    
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
    
    # 目的関数: 実売上利益 - 機会損失 - 売残損失
    # 実売上 = min(発注量, 需要) = demands[i] - shortage[i]
    # 利益 = 実売上 × 卸売価格 = Price × (demands[i] - shortage[i])
    # 機会損失 = UnderageCost × 不足量
    # 売残損失 = OverageCost × 余剰量
    prob += lpSum([
        products[i]["Price"] * (demands[i] - shortage[i])
        - products[i]["UnderageCost"] * shortage[i]
        - products[i]["OverageCost"] * surplus[i]
        for i in range(n_products)
    ]), "TotalProfit"
    
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
        (全シミュレーション結果のDataFrame, 利益額リスト, 統計情報)
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
    
    # 全シミュレーション結果をDataFrameに変換
    orders_array = np.array(all_orders)
    product_names = [p["Style"] for p in products]
    
    # 各シミュレーションの発注量を表形式で整理
    df_all_results = pd.DataFrame(
        orders_array,
        columns=product_names,
        index=[f"シミュレーション {i+1}" for i in range(len(all_orders))]
    )
    
    # 統計情報を計算
    stats_info = {}
    for i, product_name in enumerate(product_names):
        orders_for_product = orders_array[:, i]
        stats_info[product_name] = {
            "平均": np.mean(orders_for_product),
            "中央値": np.median(orders_for_product),
            "最小": np.min(orders_for_product),
            "最大": np.max(orders_for_product),
            "標準偏差": np.std(orders_for_product),
            "最小ロット数以上": np.sum(orders_for_product >= min_lot_size),
            "最小ロット数以上率": np.sum(orders_for_product >= min_lot_size) / len(orders_for_product) * 100,
            "0発注": np.sum(orders_for_product < 0.01),  # 実質的に0
            "0発注率": np.sum(orders_for_product < 0.01) / len(orders_for_product) * 100
        }
    
    return df_all_results, all_profits, stats_info


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
        st.info(
            "目的関数: min(発注量, 需要) × 卸売価格 − (機会損失 + 売残損失) を最大化 "
            f"| 最小ロット数: {min_lot_size}"
        )
        
        if st.button("シミュレーション実行", key="hk_button"):
            with st.spinner("シミュレーション実行中..."):
                df_all_results, profits, stats_info = run_monte_carlo_simulation(
                    PRODUCTS, CONFIG, min_lot_size, n_simulations
                )
                
                if df_all_results is not None:
                    st.success(f"シミュレーション完了 ({len(profits)}回成功)")
                    
                    # タブで結果を整理
                    tab_summary, tab_details, tab_stats, tab_visualization = st.tabs([
                        "📋 サマリー", "📊 全シミュレーション結果", "📈 統計情報", "📉 可視化"
                    ])
                    
                    with tab_summary:
                        st.subheader("推奨発注量（複数の指標）")
                        
                        # 推奨方法の選択
                        recommendation_method = st.radio(
                            "推奨方法を選択",
                            ["中央値", "平均値", "最頻値（四捨五入後）"],
                            horizontal=True,
                            key="hk_recommendation_method"
                        )
                        
                        # 推奨発注量を計算
                        if recommendation_method == "中央値":
                            recommended_orders = [stats_info[p["Style"]]["中央値"] for p in PRODUCTS]
                        elif recommendation_method == "平均値":
                            recommended_orders = [stats_info[p["Style"]]["平均"] for p in PRODUCTS]
                        else:  # 最頻値
                            # 各商品の発注量を四捨五入して最頻値を計算
                            recommended_orders = []
                            for p in PRODUCTS:
                                orders = df_all_results[p["Style"]].values
                                rounded_orders = np.round(orders)
                                # 0を除外して最頻値を計算
                                non_zero = rounded_orders[rounded_orders > 0]
                                if len(non_zero) > 0:
                                    values, counts = np.unique(non_zero, return_counts=True)
                                    mode_idx = np.argmax(counts)
                                    recommended_orders.append(values[mode_idx])
                                else:
                                    recommended_orders.append(0)
                        
                        # 最小ロット数制約を考慮して調整
                        adjusted_orders = []
                        for i, order in enumerate(recommended_orders):
                            if order > 0 and order < min_lot_size:
                                adjusted_orders.append(min_lot_size)
                            else:
                                adjusted_orders.append(order)
                        
                        # 合計が10000になるように調整（比例配分）
                        total_adjusted = sum(adjusted_orders)
                        if total_adjusted > 0:
                            scale_factor = CONFIG["TotalOrder"] / total_adjusted
                            final_orders = [x * scale_factor for x in adjusted_orders]
                        else:
                            final_orders = adjusted_orders
                        
                        df_summary = pd.DataFrame({
                            "商品": [p["Style"] for p in PRODUCTS],
                            f"推奨発注量({recommendation_method})": [round(x, 2) for x in recommended_orders],
                            "最小ロット数調整後": [round(x, 2) for x in adjusted_orders],
                            "最終推奨発注量": [round(x, 2) for x in final_orders],
                            "平均需要": [p["Mean"] for p in PRODUCTS],
                            "最小ロット数以上率(%)": [round(stats_info[p["Style"]]["最小ロット数以上率"], 1) for p in PRODUCTS]
                        })
                        st.dataframe(df_summary, use_container_width=True)
                        
                        total_final = sum(final_orders)
                        st.metric("最終合計発注量", f"{total_final:,.2f}", 
                                 delta=f"{total_final - CONFIG['TotalOrder']:,.2f}" if abs(total_final - CONFIG['TotalOrder']) > 0.01 else "目標達成")
                        
                        # 利益統計
                        st.subheader("利益統計（損失控除後）")
                        col1, col2, col3, col4 = st.columns(4)
                        if profits:
                            avg_profit = np.mean(profits)
                            min_profit = np.min(profits)
                            max_profit = np.max(profits)
                            std_profit = np.std(profits)
                            
                            col1.metric("平均利益", f"${avg_profit:,.2f}")
                            col2.metric("最小利益", f"${min_profit:,.2f}")
                            col3.metric("最大利益", f"${max_profit:,.2f}")
                            col4.metric("標準偏差", f"${std_profit:,.2f}")
                    
                    with tab_details:
                        st.subheader("全シミュレーション結果（各シミュレーションの発注量）")
                        st.caption("各行が1回のシミュレーション結果を表します。各商品の発注量が表示されています。")
                        # 数値を小数点以下2桁で表示
                        df_display = df_all_results.round(2)
                        st.dataframe(df_display, use_container_width=True, height=400)
                        
                        # CSVダウンロード
                        csv = df_display.to_csv(index=True).encode('utf-8-sig')
                        st.download_button(
                            label="📥 CSV形式でダウンロード",
                            data=csv,
                            file_name=f"simulation_results_{factory_name}_{n_simulations}回.csv",
                            mime="text/csv",
                            key="hk_download_csv"
                        )
                    
                    with tab_stats:
                        st.subheader("各商品の統計情報")
                        
                        stats_data = []
                        for p in PRODUCTS:
                            stats_data.append({
                                "商品": p["Style"],
                                "平均発注量": round(stats_info[p["Style"]]["平均"], 2),
                                "中央値": round(stats_info[p["Style"]]["中央値"], 2),
                                "最小発注量": round(stats_info[p["Style"]]["最小"], 2),
                                "最大発注量": round(stats_info[p["Style"]]["最大"], 2),
                                "標準偏差": round(stats_info[p["Style"]]["標準偏差"], 2),
                                "最小ロット数以上回数": int(stats_info[p["Style"]]["最小ロット数以上"]),
                                "最小ロット数以上率(%)": round(stats_info[p["Style"]]["最小ロット数以上率"], 1),
                                "0発注回数": int(stats_info[p["Style"]]["0発注"]),
                                "0発注率(%)": round(stats_info[p["Style"]]["0発注率"], 1)
                            })
                        
                        df_stats = pd.DataFrame(stats_data)
                        st.dataframe(df_stats, use_container_width=True)
                        
                        # 最小ロット数遵守率の可視化
                        st.subheader("最小ロット数遵守率")
                        compliance_rates = [stats_info[p["Style"]]["最小ロット数以上率"] for p in PRODUCTS]
                        fig_compliance = px.bar(
                            x=[p["Style"] for p in PRODUCTS],
                            y=compliance_rates,
                            labels={"x": "商品", "y": "遵守率 (%)"},
                            title="各商品の最小ロット数遵守率",
                            color=compliance_rates,
                            color_continuous_scale="RdYlGn"
                        )
                        fig_compliance.add_hline(y=100, line_dash="dash", line_color="red", 
                                                annotation_text="100%目標")
                        st.plotly_chart(fig_compliance, use_container_width=True)
                    
                    with tab_visualization:
                        # 発注量分布の箱ひげ図
                        st.subheader("発注量分布（箱ひげ図）")
                        fig_box = go.Figure()
                        for p in PRODUCTS:
                            fig_box.add_trace(go.Box(
                                y=df_all_results[p["Style"]].values,
                                name=p["Style"],
                                boxmean='sd'
                            ))
                        fig_box.add_hline(y=min_lot_size, line_dash="dash", line_color="red",
                                         annotation_text=f"最小ロット数 ({min_lot_size})")
                        fig_box.update_layout(
                            title="各商品の発注量分布",
                            yaxis_title="発注量",
                            xaxis_title="商品"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                        
                        # 利益分布のヒストグラム
                        st.subheader("利益分布（損失控除後）")
                        fig_hist = px.histogram(
                            x=profits,
                            nbins=30,
                            labels={"x": "利益額 ($)", "y": "頻度"},
                            title="利益額の分布（損失控除後）"
                        )
                        fig_hist.update_layout(showlegend=False)
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # 発注量の比較（平均値、中央値、最小ロット数）
                        st.subheader("発注量の比較")
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            name="平均発注量",
                            x=[p["Style"] for p in PRODUCTS],
                            y=[stats_info[p["Style"]]["平均"] for p in PRODUCTS],
                            marker_color="steelblue"
                        ))
                        fig_bar.add_trace(go.Bar(
                            name="中央値",
                            x=[p["Style"] for p in PRODUCTS],
                            y=[stats_info[p["Style"]]["中央値"] for p in PRODUCTS],
                            marker_color="lightgreen"
                        ))
                        fig_bar.add_trace(go.Bar(
                            name="平均需要",
                            x=[p["Style"] for p in PRODUCTS],
                            y=[p["Mean"] for p in PRODUCTS],
                            marker_color="lightcoral"
                        ))
                        fig_bar.add_hline(y=min_lot_size, line_dash="dash", line_color="red",
                                         annotation_text=f"最小ロット数 ({min_lot_size})")
                        fig_bar.update_layout(
                            title="発注量の比較（平均・中央値・需要）",
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
        st.info(
            "目的関数: min(発注量, 需要) × 卸売価格 − (機会損失 + 売残損失) を最大化 "
            f"| 最小ロット数: {min_lot_size}"
        )
        
        if st.button("シミュレーション実行", key="cn_button"):
            with st.spinner("シミュレーション実行中..."):
                df_all_results, profits, stats_info = run_monte_carlo_simulation(
                    PRODUCTS, CONFIG, min_lot_size, n_simulations
                )
                
                if df_all_results is not None:
                    st.success(f"シミュレーション完了 ({len(profits)}回成功)")
                    
                    # タブで結果を整理
                    tab_summary, tab_details, tab_stats, tab_visualization = st.tabs([
                        "📋 サマリー", "📊 全シミュレーション結果", "📈 統計情報", "📉 可視化"
                    ])
                    
                    with tab_summary:
                        st.subheader("推奨発注量（複数の指標）")
                        
                        # 推奨方法の選択
                        recommendation_method = st.radio(
                            "推奨方法を選択",
                            ["中央値", "平均値", "最頻値（四捨五入後）"],
                            horizontal=True,
                            key="cn_recommendation_method"
                        )
                        
                        # 推奨発注量を計算
                        if recommendation_method == "中央値":
                            recommended_orders = [stats_info[p["Style"]]["中央値"] for p in PRODUCTS]
                        elif recommendation_method == "平均値":
                            recommended_orders = [stats_info[p["Style"]]["平均"] for p in PRODUCTS]
                        else:  # 最頻値
                            # 各商品の発注量を四捨五入して最頻値を計算
                            recommended_orders = []
                            for p in PRODUCTS:
                                orders = df_all_results[p["Style"]].values
                                rounded_orders = np.round(orders)
                                # 0を除外して最頻値を計算
                                non_zero = rounded_orders[rounded_orders > 0]
                                if len(non_zero) > 0:
                                    values, counts = np.unique(non_zero, return_counts=True)
                                    mode_idx = np.argmax(counts)
                                    recommended_orders.append(values[mode_idx])
                                else:
                                    recommended_orders.append(0)
                        
                        # 最小ロット数制約を考慮して調整
                        adjusted_orders = []
                        for i, order in enumerate(recommended_orders):
                            if order > 0 and order < min_lot_size:
                                adjusted_orders.append(min_lot_size)
                            else:
                                adjusted_orders.append(order)
                        
                        # 合計が10000になるように調整（比例配分）
                        total_adjusted = sum(adjusted_orders)
                        if total_adjusted > 0:
                            scale_factor = CONFIG["TotalOrder"] / total_adjusted
                            final_orders = [x * scale_factor for x in adjusted_orders]
                        else:
                            final_orders = adjusted_orders
                        
                        df_summary = pd.DataFrame({
                            "商品": [p["Style"] for p in PRODUCTS],
                            f"推奨発注量({recommendation_method})": [round(x, 2) for x in recommended_orders],
                            "最小ロット数調整後": [round(x, 2) for x in adjusted_orders],
                            "最終推奨発注量": [round(x, 2) for x in final_orders],
                            "平均需要": [p["Mean"] for p in PRODUCTS],
                            "最小ロット数以上率(%)": [round(stats_info[p["Style"]]["最小ロット数以上率"], 1) for p in PRODUCTS]
                        })
                        st.dataframe(df_summary, use_container_width=True)
                        
                        total_final = sum(final_orders)
                        st.metric("最終合計発注量", f"{total_final:,.2f}", 
                                 delta=f"{total_final - CONFIG['TotalOrder']:,.2f}" if abs(total_final - CONFIG['TotalOrder']) > 0.01 else "目標達成")
                        
                        # 利益統計
                        st.subheader("利益統計（損失控除後）")
                        col1, col2, col3, col4 = st.columns(4)
                        if profits:
                            avg_profit = np.mean(profits)
                            min_profit = np.min(profits)
                            max_profit = np.max(profits)
                            std_profit = np.std(profits)
                            
                            col1.metric("平均利益", f"${avg_profit:,.2f}")
                            col2.metric("最小利益", f"${min_profit:,.2f}")
                            col3.metric("最大利益", f"${max_profit:,.2f}")
                            col4.metric("標準偏差", f"${std_profit:,.2f}")
                    
                    with tab_details:
                        st.subheader("全シミュレーション結果（各シミュレーションの発注量）")
                        st.caption("各行が1回のシミュレーション結果を表します。各商品の発注量が表示されています。")
                        # 数値を小数点以下2桁で表示
                        df_display = df_all_results.round(2)
                        st.dataframe(df_display, use_container_width=True, height=400)
                        
                        # CSVダウンロード
                        csv = df_display.to_csv(index=True).encode('utf-8-sig')
                        st.download_button(
                            label="📥 CSV形式でダウンロード",
                            data=csv,
                            file_name=f"simulation_results_{factory_name}_{n_simulations}回.csv",
                            mime="text/csv",
                            key="cn_download_csv"
                        )
                    
                    with tab_stats:
                        st.subheader("各商品の統計情報")
                        
                        stats_data = []
                        for p in PRODUCTS:
                            stats_data.append({
                                "商品": p["Style"],
                                "平均発注量": round(stats_info[p["Style"]]["平均"], 2),
                                "中央値": round(stats_info[p["Style"]]["中央値"], 2),
                                "最小発注量": round(stats_info[p["Style"]]["最小"], 2),
                                "最大発注量": round(stats_info[p["Style"]]["最大"], 2),
                                "標準偏差": round(stats_info[p["Style"]]["標準偏差"], 2),
                                "最小ロット数以上回数": int(stats_info[p["Style"]]["最小ロット数以上"]),
                                "最小ロット数以上率(%)": round(stats_info[p["Style"]]["最小ロット数以上率"], 1),
                                "0発注回数": int(stats_info[p["Style"]]["0発注"]),
                                "0発注率(%)": round(stats_info[p["Style"]]["0発注率"], 1)
                            })
                        
                        df_stats = pd.DataFrame(stats_data)
                        st.dataframe(df_stats, use_container_width=True)
                        
                        # 最小ロット数遵守率の可視化
                        st.subheader("最小ロット数遵守率")
                        compliance_rates = [stats_info[p["Style"]]["最小ロット数以上率"] for p in PRODUCTS]
                        fig_compliance = px.bar(
                            x=[p["Style"] for p in PRODUCTS],
                            y=compliance_rates,
                            labels={"x": "商品", "y": "遵守率 (%)"},
                            title="各商品の最小ロット数遵守率",
                            color=compliance_rates,
                            color_continuous_scale="RdYlGn"
                        )
                        fig_compliance.add_hline(y=100, line_dash="dash", line_color="red", 
                                                annotation_text="100%目標")
                        st.plotly_chart(fig_compliance, use_container_width=True)
                    
                    with tab_visualization:
                        # 発注量分布の箱ひげ図
                        st.subheader("発注量分布（箱ひげ図）")
                        fig_box = go.Figure()
                        for p in PRODUCTS:
                            fig_box.add_trace(go.Box(
                                y=df_all_results[p["Style"]].values,
                                name=p["Style"],
                                boxmean='sd'
                            ))
                        fig_box.add_hline(y=min_lot_size, line_dash="dash", line_color="red",
                                         annotation_text=f"最小ロット数 ({min_lot_size})")
                        fig_box.update_layout(
                            title="各商品の発注量分布",
                            yaxis_title="発注量",
                            xaxis_title="商品"
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                        
                        # 利益分布のヒストグラム
                        st.subheader("利益分布（損失控除後）")
                        fig_hist = px.histogram(
                            x=profits,
                            nbins=30,
                            labels={"x": "利益額 ($)", "y": "頻度"},
                            title="利益額の分布（損失控除後）"
                        )
                        fig_hist.update_layout(showlegend=False)
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # 発注量の比較（平均値、中央値、最小ロット数）
                        st.subheader("発注量の比較")
                        fig_bar = go.Figure()
                        fig_bar.add_trace(go.Bar(
                            name="平均発注量",
                            x=[p["Style"] for p in PRODUCTS],
                            y=[stats_info[p["Style"]]["平均"] for p in PRODUCTS],
                            marker_color="steelblue"
                        ))
                        fig_bar.add_trace(go.Bar(
                            name="中央値",
                            x=[p["Style"] for p in PRODUCTS],
                            y=[stats_info[p["Style"]]["中央値"] for p in PRODUCTS],
                            marker_color="lightgreen"
                        ))
                        fig_bar.add_trace(go.Bar(
                            name="平均需要",
                            x=[p["Style"] for p in PRODUCTS],
                            y=[p["Mean"] for p in PRODUCTS],
                            marker_color="lightcoral"
                        ))
                        fig_bar.add_hline(y=min_lot_size, line_dash="dash", line_color="red",
                                         annotation_text=f"最小ロット数 ({min_lot_size})")
                        fig_bar.update_layout(
                            title="発注量の比較（平均・中央値・需要）",
                            xaxis_title="商品",
                            yaxis_title="数量",
                            barmode="group"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.error("シミュレーションに失敗しました。")


if __name__ == "__main__":
    main()

