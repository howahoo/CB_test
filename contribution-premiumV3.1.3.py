import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import KNNImputer

import os
from scipy import interpolate
from scipy.signal import savgol_filter
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest


import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def isolation_Forest(df):
    # 选择需要检测异常的特征列
    feature_columns = ['溢价率', '正股_1m_成交均价', '转债_1m_成交均价']
    
    # 复制数据，避免修改原始数据
    df_clean = df.copy()
    
    for col in feature_columns:
        if df_clean[col].isnull().any():
            # 使用插值或统计方法填充缺失值
            if col in ['溢价率', '正股_1m_成交均价', '转债_1m_成交均价']:
                # 使用已有插值逻辑或重新插值
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
            else:
                # 使用稳健统计量填充（中位数比均值更抗异常值）
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 特征工程：添加衍生特征（有助于提高异常检测精度）
    df_clean['溢价率变化'] = df_clean['溢价率'].pct_change().fillna(0)
    df_clean['正股价格变化'] = df_clean['正股_1m_成交均价'].pct_change().fillna(0)
    df_clean['转债价格变化'] = df_clean['转债_1m_成交均价'].pct_change().fillna(0)
    
    # 构建特征矩阵
    X = df_clean[feature_columns].values
    
    # 训练Isolation Forest模型
    model = IsolationForest(
        n_estimators=100,             # 增加树的数量提高精度
        contamination=0.05,           # 降低异常值比例估计，专注于极端异常
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    # 预测异常值并获取异常分数
    df_clean['anomaly'] = model.fit_predict(X)
    df_clean['anomaly_score'] = model.decision_function(X)
    
    # 标记异常值（-1表示异常）
    df_clean['is_anomaly'] = df_clean['anomaly'] == -1
    
    # 识别极端异常值（分数极低的样本）
    extreme_threshold = np.percentile(df_clean['anomaly_score'], 5)  # 分数最低的5%
    df_clean['is_extreme'] = df_clean['anomaly_score'] < extreme_threshold
    
    # 处理异常值
    for col in feature_columns:
        # 获取正常值的统计信息（使用中位数和IQR更稳健）
        normal_data = df_clean.loc[~df_clean['is_anomaly'], col]
        median = normal_data.median()
        q1, q3 = np.percentile(normal_data, [25, 75])
        iqr = q3 - q1
        
        # 定义更严格的边界（使用IQR而不是标准差）
        lower_bound = max(median - 3 * iqr, normal_data.min())
        upper_bound = min(median + 3 * iqr, normal_data.max())
        
        # 对极端异常值使用更保守的替换策略
        extreme_mask = df_clean['is_extreme']
        df_clean.loc[extreme_mask, col] = median  # 用中位数替换极端异常
        
        # 对一般异常值使用区间限制
        normal_anomaly_mask = df_clean['is_anomaly'] & ~extreme_mask
        df_clean.loc[normal_anomaly_mask, col] = np.clip(
            df_clean.loc[normal_anomaly_mask, col], lower_bound, upper_bound
        )
    
    return df_clean

def interpolate_premium_rate(df):
    columns = ['溢价率', '正股_1m_成交均价', '转债_1m_成交均价']
    
    # 检查指定字段是否都在数据框中
    invalid_fields = [col for col in columns if col not in df.columns]
    if invalid_fields:
        raise ValueError(f"以下字段不存在于数据框中：{invalid_fields}")
    
    # 复制数据框，避免修改原始数据
    df = df.copy()
    
    # 第一步：用指定字段前5行的均值填充缺失值
    first_five = df.head(5)  # 获取前5行数据
    field_means = first_five[columns].mean()  # 计算前5行的均值
    df[columns] = df[columns].fillna(field_means)  # 填充缺失值
    
    # 第二步：对填充后的字段进行Z-score变换
    means = df[columns].mean()  # 填充后字段的均值
    stds = df[columns].std(ddof=0)  # 填充后字段的总体标准差
    # 处理标准差为0的情况（避免除以0）
    df[columns] = (df[columns] - means).div(stds.where(stds != 0, 1), axis=1)
    
    # 应用隔离森林进行异常值处理
    df = isolation_Forest(df)

    return df
def calculate_derivative(prices,times):
    # 转换为numpy数组，确保处理一致性
    prices = np.asarray(prices, dtype=np.float64)
    
    # 校验价格和时间长度是否一致（核心对齐前提）
    if len(prices) != len(times):
        raise ValueError(f"价格长度({len(prices)})与时间长度({len(times)})不匹配，无法对齐")
    
    # 检查价格中的零值（可能导致无效增长率）
    zero_indices = np.where(prices == 0)[0]
    if len(zero_indices) > 0:
        print(f"警告：价格序列包含{len(zero_indices)}个零值，可能导致无效增长率")
    
    n = len(prices)
    growth_rates = np.full(n, np.nan, dtype=np.float64)
    skip_until = -1  # 标记需要跳过的索引上限
    
    for i in range(1, n):
        # 如果当前索引在跳过范围内，继续跳过
        if i <= skip_until:
            continue
            
        # 检查前一个价格是否为0
        if prices[i-1] == 0:
            # 标记需要跳过当前行和下一行
            skip_until = i + 1
            continue
            
        # 正常计算增长率
        if prices[i-1] != 0:  # 再次确认避免除零
            growth_rates[i] = prices[i] / prices[i-1] - 1


    # 标记无效值（NaN/无穷大），并转换为NaN
    invalid_mask = np.isnan(growth_rates) | np.isinf(growth_rates)
    growth_rates[invalid_mask] = np.nan  # 统一用NaN表示无效值
    
    # 统计无效值并提示（排除首值的NaN）
    non_first_invalid = invalid_mask[1:]  # 从第二个元素开始统计
    if np.any(non_first_invalid):
        invalid_count = np.sum(non_first_invalid)
        first_invalid_idx = np.where(non_first_invalid)[0][0] + 1  # 加1还原原始索引
        print(f"警告：共{invalid_count}个无效增长率（非首值），第一个位于索引{first_invalid_idx}（时间：{times[first_invalid_idx]}）")
    
    # 转换为带时间索引的Series
    # 确保时间格式为DatetimeIndex，方便后续股债对齐
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.to_datetime(times)  # 统一转换为时间索引

    # 标记无效值（NaN/无穷大），并转换为NaN
    invalid_mask = np.isnan(growth_rates) | np.isinf(growth_rates)
    growth_rates[invalid_mask] = np.nan  # 统一用NaN表示无效值
    
    # 统计无效值并提示（排除首值的NaN）
    non_first_invalid = invalid_mask[1:]  # 从第二个元素开始统计
    if np.any(non_first_invalid):
        invalid_count = np.sum(non_first_invalid)
        first_invalid_idx = np.where(non_first_invalid)[0][0] + 1  # 加1还原原始索引
        print(f"警告：共{invalid_count}个无效增长率（非首值），第一个位于索引{first_invalid_idx}（时间：{times[first_invalid_idx]}）")
    
    # 新增：处理缺失值（线性插值）
    valid_indices = np.where(~np.isnan(growth_rates))[0]
    if len(valid_indices) >= 2:  # 确保有足够的有效点进行插值
        f = interpolate.interp1d(valid_indices, growth_rates[valid_indices], bounds_error=False, fill_value="extrapolate")
        growth_rates = f(np.arange(len(growth_rates)))
    
    # 确保时间格式为DatetimeIndex
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.to_datetime(times)




    return growth_rates

def calculate_missing_premium(df):
    """计算缺失的溢价率"""
    mask = df['溢价率'].isnull()
    if mask.sum()> 0:
        # 计算转股价值
        df.loc[mask, '转股价值'] = df.loc[mask, '正股_1m_成交均价'] * 100 / df.loc[mask, '转股价格']

        # 计算溢价率，避免除零错误
        valid_value_mask = (df['转股价值'] != 0) & (~df['转股价值'].isna())
        calculation_mask = mask & valid_value_mask

        # 确保索引一致
        valid_indices = df.index[calculation_mask]

        # 计算溢价率
        bond_price = df.loc[valid_indices, '转债_1m_成交均价']
        conversion_value = df.loc[valid_indices, '转股价值']
        premium_rate = (bond_price - conversion_value) / conversion_value * 100

        # 直接赋值
        df.loc[valid_indices, '溢价率'] = premium_rate

    return df

def calculate_contribution(df):
    """
    计算债贡献度和股贡献度，考虑股票空头、债券多头的持仓方向
    """
    df['eob'] = pd.to_datetime(df['eob'])
    df_sorted = df.sort_values('eob').copy()

    # 过滤转股价格为 0 的行（避免除零错误）
    df_sorted = df_sorted[df_sorted['转股价格'] > 0]

    # 获取价格和时间序列（转换为 numpy 数组以避免后续警告）
    stock_prices = df['正股_1m_成交均价']
    bond_prices = df['转债_1m_成交均价']
    times = df_sorted['eob']
    conversion_prices = df['转股价格']
    
    # 计算
    stock_derivative = calculate_derivative(stock_prices, times)
    bond_derivative = calculate_derivative(bond_prices, times)

    # 固定债开仓数量
    bond_amount = 10
    
    # 计算股开仓数量（考虑转股价格的变化）
    stock_amount = bond_amount * 100 / conversion_prices
    
    # 计算持仓价值的导数（考虑空头和多头方向）
    # 股票空头：价格下跌时盈利，导数为负
    stock_value_derivative = -1 * stock_derivative * stock_amount
    
    # 债券多头：价格上涨时盈利，导数为正
    bond_value_derivative = bond_derivative * bond_amount
    
    # 计算总变化率
    total_derivative = stock_value_derivative + bond_value_derivative
    
    # 计算贡献率，避免除零错误
    stock_contributions = np.zeros_like(stock_value_derivative)
    bond_contributions = np.zeros_like(bond_value_derivative)

    valid_mask = total_derivative != 0

    # 经过敞口对齐
    stock_contributions[valid_mask] = stock_value_derivative[valid_mask] 
    bond_contributions[valid_mask] = bond_value_derivative[valid_mask] 

    # 恢复原始索引顺序
    original_order = pd.Series(range(len(df)), index=df_sorted.index)
    stock_contributions = stock_contributions[original_order]
    bond_contributions = bond_contributions[original_order]
    
    return bond_contributions, stock_contributions

def create_visualization(df):
    """创建溢价率与贡献度关系的交互式可视化图表，支持分层显示不同数据"""

    # 先增加转股价值列
    df['转股价值'] = 100 / df['转股价格'] * df['正股_1m_成交均价']

    for col in ['溢价率', '转股价值']:
        if df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            print(f"警告：{col}存在{missing_count}个缺失值，将用插值法填充")
            df[col].interpolate(method='linear', inplace=True)  # 线性插值
            # 若仍有缺失（如首行），用均值填充
            df[col].fillna(df[col].mean(), inplace=True)

    X = df[['溢价率', '转股价值']].values

    # 创建基础2D图表
    fig2d = go.Figure()
    # 定义颜色
    bond_color = 'rgba(76, 114, 176, 0.6)'
    stock_color = 'rgba(221, 132, 82, 0.6)'
    bond_trend_color = 'rgb(37, 52, 148)'
    stock_trend_color = 'rgb(178, 37, 37)'
    # 计算y轴合理范围
    all_contributions = pd.concat([df['债增长率'], df['股增长率']])
    y_min, y_max = all_contributions.quantile([0.25, 0.75])
    margin = (y_max - y_min) * 0.05
    y_min -= margin
    y_max += margin
    # 创建所有可能的显示选项
    options = {
        '全部数据': {
            'scatter': [
                {
                    'y': '债增长率',
                    'name': '债增长率',
                    'marker': dict(size=5, opacity=0.4, color=bond_color),
                    'hovertemplate': '溢价率: %{x:.2f}<br>债增长率: %{y:.2f}<extra></extra>'
                },
                {
                    'y': '股增长率',
                    'name': '股增长率',
                    'marker': dict(size=5, opacity=0.4, color=stock_color),
                    'hovertemplate': '溢价率: %{x:.2f}<br>股增长率: %{y:.2f}<extra></extra>'
                }
            ],
            'trendline': [
                {
                    'y': '债增长率',
                    'name': '债增长率趋势',
                    'line': dict(color=bond_trend_color, width=2.5)
                },
                {
                    'y': '股增长率',
                    'name': '股增长率趋势',
                    'line': dict(color=stock_trend_color, width=2.5)
                }
            ]
        },
        '仅债增长率': {
            'scatter': [
                {
                    'y': '债增长率',
                    'name': '债增长率',
                    'marker': dict(size=5, opacity=0.4, color=bond_color),
                    'hovertemplate': '溢价率: %{x:.2f}<br>债增长率: %{y:.2f}<extra></extra>'
                }
            ],
            'trendline': [
                {
                    'y': '债增长率',
                    'name': '债增长率趋势',
                    'line': dict(color=bond_trend_color, width=2.5)
                }
            ]
        },
        '仅股增长率': {
            'scatter': [
                {
                    'y': '股增长率',
                    'name': '股增长率',
                    'marker': dict(size=5, opacity=0.4, color=stock_color),
                    'hovertemplate': '溢价率: %{x:.2f}<br>股增长率: %{y:.2f}<extra></extra>'
                }
            ],
            'trendline': [
                {
                    'y': '股增长率',
                    'name': '股增长率趋势',
                    'line': dict(color=stock_trend_color, width=2.5)
                }
            ]
        }
    }
    # 为每种选项创建trace
    for option_name, option_data in options.items():
        visible = (option_name == '全部数据')  # 默认只显示全部数据
        # 添加趋势线
        for trend_data in option_data['trendline']:
            trend_df = px.scatter(
                df, 
                x='溢价率', 
                y=trend_data['y'],
                trendline='lowess',
                trendline_options=dict(frac=0.3)
            )
            fig2d.add_trace(
                go.Scatter(
                    x=trend_df.data[1]['x'],
                    y=trend_df.data[1]['y'],
                    mode='lines',
                    name=trend_data['name'],
                    line=trend_data['line'],
                    visible=visible
                )
            )
    # 添加零线参考
    fig2d.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig2d.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray", opacity=0.5)
    # 设置y轴范围
    fig2d.update_yaxes(
        range=[y_min, y_max],
        title_text='绝对增长率',
        gridcolor='rgba(220, 220, 220, 0.5)',
        tickfont=dict(size=12),
        title_font=dict(size=14)
    )
    # 创建下拉菜单
    buttons = []
    start_index = 0
    for option_name in options.keys():
        traces_count = len(options[option_name]['scatter']) + len(options[option_name]['trendline'])
        visible = [False] * len(fig2d.data)
        visible[start_index:start_index+traces_count] = [True] * traces_count
        buttons.append(dict(
            label=option_name,
            method='update',
            args=[{'visible': visible}]
        ))
        start_index += traces_count
    # 更新布局
    fig2d.update_layout(
        title='可转债溢价率与股债增长率关系分析',
        xaxis_title='溢价率 (%)',
        legend_title='指标',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(248, 249, 250, 1.0)',
        font=dict(family="SimHei, WenQuanYi Micro Hei, Heiti TC"),  # 确保中文显示
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="x unified",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                active=0,
                x=0.05,
                y=1.15,
                buttons=buttons,
                showactive=True,
                bgcolor='rgba(240, 240, 240, 0.8)',
                bordercolor='rgba(200, 200, 200, 0.8)',
                font=dict(size=12)
            )
        ]
    )

    # ----------------- 新增三维三轴线图 -----------------
    # 默认显示债增长率

    fig3d = go.Figure()
    # 债增长率三维
    fig3d.add_trace(go.Scatter3d(
        x=df['溢价率'],
        y=df['债增长率'],
        z=df['转股价值'],
        mode='markers',
        marker=dict(
            size=1,  
            color='rgba(76, 114, 176, 0.7)',
            opacity=0.7,
            line=dict(width=0.5, color='darkblue')
        ),
        name='债增长率',
        text=df['eob'].astype(str),
        visible=True,
        hovertemplate='溢价率(%): %{x:.2f}<br>债增长率: %{y:.4f}<br>转股价值: %{z:.2f}<br>时间: %{text}'
    ))
    # 股增长率三维
    fig3d.add_trace(go.Scatter3d(
        x=df['溢价率'],
        y=df['股增长率'],
        z=df['转股价值'],
        mode='markers',
        marker=dict(
            size=1,  
            color='rgba(221, 132, 82, 0.3)',
            opacity=0.3,
            line=dict(width=0.5, color='darkred')
        ),
        name='股增长率',
        text=df['eob'].astype(str),
        visible=True,
        hovertemplate='溢价率(%): %{x:.2f}<br>股增长率: %{y:.4f}<br>转股价值: %{z:.2f}<br>时间: %{text}'
    ))
    # 布局同前
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(
                title='溢价率 (%)',
                backgroundcolor='rgba(248, 249, 250, 1.0)',
                gridcolor='rgba(220, 220, 220, 0.5)',
                showbackground=True,
                zerolinecolor='gray',
                showspikes=True,
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
            yaxis=dict(
                title='增长率',
                backgroundcolor='rgba(248, 249, 250, 1.0)',
                gridcolor='rgba(220, 220, 220, 0.5)',
                showbackground=True,
                zerolinecolor='gray',
                showspikes=True,
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
            zaxis=dict(
                title='转股价值',
                backgroundcolor='rgba(248, 249, 250, 1.0)',
                gridcolor='rgba(220, 220, 220, 0.5)',
                showbackground=True,
                zerolinecolor='gray',
                showspikes=True,
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
        ),
        title=dict(
            text='溢价率-增长率-转股价值 三维关系图',
            font=dict(size=18, family="SimHei, WenQuanYi Micro Hei, Heiti TC"),
            x=0.5
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        plot_bgcolor='rgba(248, 249, 250, 1.0)',
        font=dict(family="SimHei, WenQuanYi Micro Hei, Heiti TC"),
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="closest"
    )

    y_bond = df['债增长率'].values
    reg_bond = LinearRegression().fit(X, y_bond)

    # 拟合股增长率
    y_stock = df['股增长率'].values
    reg_stock = LinearRegression().fit(X, y_stock)

    # 构造网格
    x_range = np.linspace(df['溢价率'].min(), df['溢价率'].max(), 30)
    z_range = np.linspace(df['转股价值'].min(), df['转股价值'].max(), 30)
    xx, zz = np.meshgrid(x_range, z_range)
    X_grid = np.c_[xx.ravel(), zz.ravel()]

    # 预测回归面
    yy_bond = reg_bond.predict(X_grid).reshape(xx.shape)
    yy_stock = reg_stock.predict(X_grid).reshape(xx.shape)

    # 添加债增长率回归面
    fig3d.add_trace(go.Surface(
        x=xx, y=yy_bond, z=zz,
        colorscale='Blues',
        opacity=0.4,
        showscale=False,
        name='债增长率回归面'
    ))
    # 添加股增长率回归面
    fig3d.add_trace(go.Surface(
        x=xx, y=yy_stock, z=zz,
        colorscale='Oranges',
        opacity=0.3,
        showscale=False,
        name='股增长率回归面'
    ))

    return fig2d, fig3d

def create_3d_visualization_both(df):

    fig = go.Figure()

    # 债增长率
    fig.add_trace(go.Scatter3d(
        x=df['溢价率'],
        y=df['债增长率'],
        z=df['转股价值'],
        mode='markers',
        marker=dict(
            size=6,
            color='rgba(76, 114, 176, 0.7)',  # 蓝色
            opacity=0.7,
            line=dict(width=0.5, color='darkblue')
        ),
        name='债增长率',
        text=df['eob'].astype(str)
    ))

    # 股增长率
    fig.add_trace(go.Scatter3d(
        x=df['溢价率'],
        y=df['股增长率'],
        z=df['转股价值'],
        mode='markers',
        marker=dict(
            size=6,
            color='rgba(221, 132, 82, 0.7)',  # 橙色
            opacity=0.7,
            line=dict(width=0.5, color='darkred')
        ),
        name='股增长率',
        text=df['eob'].astype(str)
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='溢价率 (%)',
                backgroundcolor='rgba(248, 249, 250, 1.0)',
                gridcolor='rgba(220, 220, 220, 0.5)',
                showbackground=True,
                zerolinecolor='gray',
                showspikes=True,
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
            yaxis=dict(
                title='增长率',
                backgroundcolor='rgba(248, 249, 250, 1.0)',
                gridcolor='rgba(220, 220, 220, 0.5)',
                showbackground=True,
                zerolinecolor='gray',
                showspikes=True,
                tickfont=dict(size=12),
                title_font=dict(size=14),
                range=[-1, 1],  # 限制范围
                autorange=False  
            ),
            zaxis=dict(
                title='转股价值',
                backgroundcolor='rgba(248, 249, 250, 1.0)',
                gridcolor='rgba(220, 220, 220, 0.5)',
                showbackground=True,
                zerolinecolor='gray',
                showspikes=True,
                tickfont=dict(size=12),
                title_font=dict(size=14)
            ),
        ),
        title=dict(
            text='溢价率-增长率-转股价值 三维关系图',
            font=dict(size=18, family="SimHei, WenQuanYi Micro Hei, Heiti TC"),
            x=0.5
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        plot_bgcolor='rgba(248, 249, 250, 1.0)',
        font=dict(family="SimHei, WenQuanYi Micro Hei, Heiti TC"),
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="closest"
    )
    return fig

def main():
    # 文件路径处理
    # stock_file_start = r'C:\Users\Administrator\Desktop\economy\Data\mix_data\Data\juejin_1min\mix'
    stock_file_start = os.path.join(os.path.dirname(__file__), 'Data', 'juejin_1min', 'mix')

    stock_file_end = input("请输入股票数据文件名: ").strip()
    if not stock_file_end.endswith('.csv'):
        stock_file_end += '.csv'
    file_path = os.path.join(stock_file_start, stock_file_end)
    
    # 读取文件
    try:
        df = pd.read_csv(file_path, encoding='gbk')
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            print(f"文件读取错误: {e}")
            return
    except Exception as e:
        print(f"文件读取错误: {e}")
        return
    

    # 检查必要列
    required_columns = ['eob', '交易日期', '正股_1m_成交均价', '转债_1m_成交均价', '溢价率', '转股价格']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"缺少必要列: {', '.join(missing_columns)}")
        return
    
    # 处理日期格式
    df['eob'] = pd.to_datetime(df['eob'])
    df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y-%m-%d')

    # 计算缺失的溢价率
    df = calculate_missing_premium(df)

    # 获取数据
    bond_contributions, stock_contributions = calculate_contribution(df)

    interpolate_premium_rate(df)
    
    # 将贡献度添加到DataFrame
    df['债增长率'] = bond_contributions
    df['股增长率'] = stock_contributions
    
    # 日期区间筛选
    start_date = input("请输入起始日期（如20250101，回车跳过）: ").strip()
    end_date = input("请输入结束日期（如20250630，回车跳过）: ").strip()
    # 获取eob的时区
    tz = df['eob'].dt.tz

    if start_date:
        start_date = pd.to_datetime(start_date, format='%Y%m%d').tz_localize(tz)
        df = df[df['eob'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date, format='%Y%m%d').tz_localize(tz)
        df = df[df['eob'] <= end_date]
    
    # 创建图表
    fig2d, fig3d = create_visualization(df)
    # fig = create_3d_visualization_both(df)
    # 显示图表

    df.to_csv('data.csv',index=False)


    fig2d.show()
    fig3d.show()
    
if __name__ == "__main__":
    main()