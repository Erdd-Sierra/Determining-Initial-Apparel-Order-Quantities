"""
線形計画問題を解くPythonプログラム
scipy.optimize.linprogを使用

必要なライブラリのインストール:
    pip install numpy scipy

または:
    pip install -r requirements.txt
"""

import numpy as np
from scipy.optimize import linprog

def solve_linear_programming(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, 
                            bounds=None, method='highs'):
    """
    線形計画問題を解く関数
    
    最小化: c^T * x
    制約条件:
        A_ub * x <= b_ub  (不等式制約)
        A_eq * x == b_eq  (等式制約)
        bounds            (変数の範囲)
    
    パラメータ:
        c: 目的関数の係数ベクトル
        A_ub: 不等式制約の係数行列 (オプション)
        b_ub: 不等式制約の右辺ベクトル (オプション)
        A_eq: 等式制約の係数行列 (オプション)
        b_eq: 等式制約の右辺ベクトル (オプション)
        bounds: 変数の範囲 [(min, max), ...] (オプション)
        method: 解法 ('highs', 'simplex', 'interior-point')
    
    戻り値:
        最適解と最適値
    """
    # 線形計画問題を解く
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method=method)
    
    if result.success:
        print("最適解が見つかりました！")
        print(f"最適値: {result.fun:.4f}")
        print(f"最適解: {result.x}")
        return result.x, result.fun
    else:
        print("最適解が見つかりませんでした。")
        print(f"メッセージ: {result.message}")
        return None, None


# 例1: 基本的な線形計画問題
# 最小化: -3x1 - 2x2
# 制約条件:
#   x1 + x2 <= 10
#   2x1 + x2 <= 15
#   x1 >= 0, x2 >= 0
print("=" * 50)
print("例1: 基本的な線形計画問題")
print("=" * 50)

# 目的関数の係数（最小化なので負の値）
c = np.array([-3, -2])

# 不等式制約: A_ub * x <= b_ub
A_ub = np.array([[1, 1],
                 [2, 1]])
b_ub = np.array([10, 15])

# 変数の範囲（非負制約）
bounds = [(0, None), (0, None)]

x_opt, f_opt = solve_linear_programming(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)


# 例2: 等式制約を含む問題
# 最小化: x1 + 2x2 + 3x3
# 制約条件:
#   x1 + x2 + x3 = 10
#   x1 + 2x2 <= 8
#   x1 >= 0, x2 >= 0, x3 >= 0
print("\n" + "=" * 50)
print("例2: 等式制約を含む線形計画問題")
print("=" * 50)

c2 = np.array([1, 2, 3])

# 等式制約
A_eq = np.array([[1, 1, 1]])
b_eq = np.array([10])

# 不等式制約
A_ub2 = np.array([[1, 2, 0]])
b_ub2 = np.array([8])

bounds2 = [(0, None), (0, None), (0, None)]

x_opt2, f_opt2 = solve_linear_programming(c2, A_ub=A_ub2, b_ub=b_ub2,
                                           A_eq=A_eq, b_eq=b_eq, bounds=bounds2)


# 例3: 最大化問題（最小化に変換）
# 最大化: 5x1 + 4x2
# 制約条件:
#   2x1 + 3x2 <= 12
#   4x1 + x2 <= 8
#   x1 >= 0, x2 >= 0
print("\n" + "=" * 50)
print("例3: 最大化問題（最小化に変換）")
print("=" * 50)

# 最大化は目的関数の符号を反転して最小化に変換
c3 = np.array([-5, -4])  # 負の値にして最小化

A_ub3 = np.array([[2, 3],
                  [4, 1]])
b_ub3 = np.array([12, 8])

bounds3 = [(0, None), (0, None)]

x_opt3, f_opt3 = solve_linear_programming(c3, A_ub=A_ub3, b_ub=b_ub3, bounds=bounds3)

if f_opt3 is not None:
    print(f"最大値: {-f_opt3:.4f}")  # 符号を戻す


# カスタム問題を解く関数
def solve_custom_problem():
    """
    ユーザーがカスタム問題を解くための関数
    ここを編集して自分の問題を解けます
    """
    print("\n" + "=" * 50)
    print("カスタム問題")
    print("=" * 50)
    
    # ここを編集してください
    # 例: 最小化: 2x1 + 3x2
    c_custom = np.array([2, 3])
    
    # 制約条件: x1 + x2 >= 5  →  -x1 - x2 <= -5
    A_ub_custom = np.array([[-1, -1]])
    b_ub_custom = np.array([-5])
    
    bounds_custom = [(0, None), (0, None)]
    
    x_opt_custom, f_opt_custom = solve_linear_programming(
        c_custom, A_ub=A_ub_custom, b_ub=b_ub_custom, bounds=bounds_custom
    )
    
    return x_opt_custom, f_opt_custom


# カスタム問題を実行（コメントアウトを外して使用）
# solve_custom_problem()

