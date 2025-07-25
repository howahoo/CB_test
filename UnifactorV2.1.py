import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import plotly.graph_objects as go


def spearmanr_cheat(df):
    """计算换手率与成交量的斯皮尔曼相关系数"""
    df['换手率'] = df['成交额'] / df['流通市值']
    stat, p_value = spearmanr(df['换手率'], df['成交量'])
    print(f"斯皮尔曼相关系数：{stat:.4f}，p值：{p_value:.4f}")
    return 0


def calc_beta1(x, y):
    covariance = np.cov(x, y)[0, 1]
    variance_x = np.var(x, ddof=1)
    beta1 = covariance / variance_x
    beta0 = y.mean() - beta1 * x.mean()
    return beta1, beta0


def main(df):
    # 定义因变量和自变量
    y = df['收盘价']
    x = df['成交量'] / 1000000  # 成交量单位转换
    
    # 计算回归系数
    beta1, beta0 = calc_beta1(x, y)
    
    # 计算预测值和残差
    df['beta1'] = beta1
    df['beta0'] = beta0
    df['y_pre'] = beta0 + beta1 * x
    df['n0'] = y - df['y_pre']  # 残差
    
    # 输出结果
    print(f"因子暴露度:\n{beta1}")
    print(f"残差:\n{df['n0']}")
    
    # 计算拟合优度
    sse = np.sum((df['y_pre'] - y) ** 2)
    ssr = np.sum((df['y_pre'] - y.mean()) ** 2)
    sst = sse + ssr
    R_squared = 1 - sse / sst 
    print(f"拟合优度{R_squared:.4f}")
    
    return df, x, y


def plot_scatter_points(df, x, y, title="回归散点图",
                       ylabel="股价", xlabel="成交量"):
    # 创建图表
    fig = go.Figure()
    
    # 添加实际值散点
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='实际值',
        marker=dict(
            color='blue',
            size=8,
            symbol='circle',
            opacity=0.7,
            line=dict(width=1, color='white')
        )
    ))
    
    # 添加预测值散点
    fig.add_trace(go.Scatter(
        x=x,
        y=df['y_pre'],
        mode='markers',
        name='预测值',
        marker=dict(
            color='red',
            size=8,
            symbol='circle',
            opacity=0.7,
            line=dict(width=1, color='white')
        )
    ))

    # 更新布局
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.show()
    return fig  


if __name__ == "__main__":
    # 文件路径处理
    stock_file_start = r'C:\Users\Administrator\Desktop\economy\Data\stock-trading-data-pro\stock-trading-data-pro'
    stock_file_end = 'sz301611'
    stock_file_end = stock_file_end if stock_file_end.endswith('.csv') else f"{stock_file_end}.csv"
    file_path = os.path.join(stock_file_start, stock_file_end)
    
    # 数据处理与分析
    df = pd.read_csv(file_path, encoding='gbk', header=1)
    df, x, y = main(df)
    spearmanr_cheat(df)

    # 可视化
    plot_scatter_points(df, x, y)