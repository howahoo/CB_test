import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

import plotly.express as px
import plotly.graph_objects as go


#连续随机变量相关性检测
def spearmanr_cheat(df):

    df['换手率'] = df['成交额'] / df['流通市值']
    stat, p_value = spearmanr(df['换手率'], df['成交量'])
    print(f"斯皮尔曼相关系数：{stat:.4f}，p值：{p_value:.4f}")
    '''
    若 p>0.05，则无法拒绝 “两变量独立” 的原假设（即认为可能独立）；
    若 p≤0.05，则拒绝原假设（认为不独立）。
    '''
    return 0

def calc_beta1(df):
    x = df['成交量']
    y = df['收盘价']
    #calc covariance
    covariance = np.cov(x, y)[0, 1]
    #variance
    variancex = np.var(x, ddof=1)
    #beta$
    beta1 = covariance / variancex
    beta0 = y.mean() - beta1 * x.mean()

    return beta1,beta0,x,y
 
def main(df):
    #simple regresssion model
    #n0为残差
    y = df['收盘价']
    x = df['成交量']

    n0 = 0

    beta1,beta0,x,y= calc_beta1(df)
    y_pre = beta0 + beta1 * x + n0

    n0=y-y_pre

    result_df = df.copy()
    result_df['y_pred'] = y_pre
    result_df['n0'] = n0
    
    #goodness of fit
    SSE = np.sum((y_pre - y) ** 2)

    SSR = np.sum((y_pre - y.mean()) ** 2)

    SST = SSE + SSR

    R = 1 - SSR/SST 

    print(f"拟合优度{R:.4f}")

    return df,y,y_pre,x

def plot_scatter_points(y, y_pre, x, title="回归散点图",
                       ylabel="股价", xlabel="成交量", x_is_date=True):
    
    y = np.array(y)
    y_pre = np.array(y_pre)
    x = np.array(x)
    
    # 创建图表
    fig = go.Figure()
    
    # 添加实际值散点
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',  # 散点图模式
        name='实际值',
        marker=dict(
            color='blue',
            size=8,
            symbol='circle',
            opacity=0.7,  # 增加透明度，避免点重叠时看不清
            line=dict(width=1, color='white')  # 点边缘白色边框
        )
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y_pre,
        mode='markers',  # 散点图模式
        name='预测值',
        marker=dict(
            color='red',
            size=8,
            symbol='x',  # 使用X形状区分预测值
            opacity=0.7,
            line=dict(width=1, color='white')
        )
    ))
    
    xaxis_config = dict(
        showgrid=True,
        gridcolor='lightgray'
    )
    # 更新布局
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        plot_bgcolor='white',
        xaxis=xaxis_config,
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
    # stock_file_start = os.path.join(os.path.dirname(__file__), 'Data', 'juejin_1min', 'mix')

    stock_file_end = 'sz301611'
    if not stock_file_end.endswith('.csv'):
        stock_file_end += '.csv'
    file_path = os.path.join(stock_file_start, stock_file_end)
    
    df = pd.read_csv(file_path, encoding='gbk', header=1)

    df,y,y_pre,x = main(df)
    spearmanr_cheat(df)

    plot_scatter_points(y, y_pre,x)