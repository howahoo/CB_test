import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import os
from scipy import interpolate
from scipy.stats import gaussian_kde


def isolation_Forest(df):
    feature_columns = ['溢价率', '正股_1m_成交均价', '转债_1m_成交均价']
    df_clean = df.copy()
    
    for col in feature_columns:
        if df_clean[col].isnull().any():
            if col in ['溢价率', '正股_1m_成交均价', '转债_1m_成交均价']:
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    df_clean['溢价率变化'] = df_clean['溢价率'].pct_change().fillna(0)
    df_clean['正股价格变化'] = df_clean['正股_1m_成交均价'].pct_change().fillna(0)
    df_clean['转债价格变化'] = df_clean['转债_1m_成交均价'].pct_change().fillna(0)
    
    X = df_clean[feature_columns].values
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    df_clean['anomaly'] = model.fit_predict(X)
    df_clean['anomaly_score'] = model.decision_function(X)
    df_clean['is_anomaly'] = df_clean['anomaly'] == -1
    extreme_threshold = np.percentile(df_clean['anomaly_score'], 5)
    df_clean['is_extreme'] = df_clean['anomaly_score'] < extreme_threshold
    
    for col in feature_columns:
        normal_data = df_clean.loc[~df_clean['is_anomaly'], col]
        median = normal_data.median()
        q1, q3 = np.percentile(normal_data, [25, 75])
        iqr = q3 - q1
        lower_bound = max(median - 3 * iqr, normal_data.min())
        upper_bound = min(median + 3 * iqr, normal_data.max())
        
        extreme_mask = df_clean['is_extreme']
        df_clean.loc[extreme_mask, col] = median
        
        normal_anomaly_mask = df_clean['is_anomaly'] & ~extreme_mask
        df_clean.loc[normal_anomaly_mask, col] = np.clip(
            df_clean.loc[normal_anomaly_mask, col], lower_bound, upper_bound
        )
    
    return df_clean


def interpolate_premium_rate(df):
    columns = ['溢价率', '转债_1m_成交均价']
    invalid_fields = [col for col in columns if col not in df.columns]
    if invalid_fields:
        raise ValueError(f"以下字段不存在于数据框中：{invalid_fields}")
    
    df = df.copy()
    first_five = df.head(5)
    field_means = first_five[columns].mean()
    df[columns] = df[columns].fillna(field_means)
    
    means = df[columns].mean()
    stds = df[columns].std(ddof=0)
    df[columns] = (df[columns] - means).div(stds.where(stds != 0, 1), axis=1)
    
    df = isolation_Forest(df)
    return df


def calculate_derivative(prices,times):
    prices = np.asarray(prices, dtype=np.float64)
    
    if len(prices) != len(times):
        raise ValueError(f"价格长度({len(prices)})与时间长度({len(times)})不匹配，无法对齐")
    
    zero_indices = np.where(prices == 0)[0]
    if len(zero_indices) > 0:
        print(f"警告：价格序列包含{len(zero_indices)}个零值，可能导致无效增长率")
    
    n = len(prices)
    growth_rates = np.full(n, np.nan, dtype=np.float64)
    skip_until = -1
    
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.to_datetime(times)
    time_seconds = times.astype(np.int64) // 10**9
    
    for i in range(1, n):
        if i <= skip_until:
            continue
        if prices[i-1] == 0:
            skip_until = i + 1
            continue
        if prices[i-1] != 0:
            time_diff = time_seconds[i] - time_seconds[i-1]
            if time_diff != 0:
                growth_rates[i] = (prices[i] - prices[i-1]) / time_diff

    invalid_mask = np.isnan(growth_rates) | np.isinf(growth_rates)
    growth_rates[invalid_mask] = np.nan
    
    non_first_invalid = invalid_mask[1:]
    if np.any(non_first_invalid):
        invalid_count = np.sum(non_first_invalid)
        first_invalid_idx = np.where(non_first_invalid)[0][0] + 1
        print(f"警告：共{invalid_count}个无效增长率（非首值），第一个位于索引{first_invalid_idx}（时间：{times[first_invalid_idx]}）")
    
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.to_datetime(times)

    invalid_mask = np.isnan(growth_rates) | np.isinf(growth_rates)
    growth_rates[invalid_mask] = np.nan
    
    non_first_invalid = invalid_mask[1:]
    if np.any(non_first_invalid):
        invalid_count = np.sum(non_first_invalid)
        first_invalid_idx = np.where(non_first_invalid)[0][0] + 1
        print(f"警告：共{invalid_count}个无效增长率（非首值），第一个位于索引{first_invalid_idx}（时间：{times[first_invalid_idx]}）")
    
    valid_indices = np.where(~np.isnan(growth_rates))[0]
    if len(valid_indices) >= 2:
        f = interpolate.interp1d(valid_indices, growth_rates[valid_indices], bounds_error=False, fill_value="extrapolate")
        growth_rates = f(np.arange(len(growth_rates)))
    
    if not isinstance(times, pd.DatetimeIndex):
        times = pd.to_datetime(times)

    return growth_rates


def calculate_missing_premium(df):
    mask = df['溢价率'].isnull()
    if mask.sum() > 0:
        df.loc[mask, '转股价值'] = df.loc[mask, '正股_1m_成交均价'] * 100 / df.loc[mask, '转股价格']
        valid_value_mask = (df['转股价值'] != 0) & (~df['转股价值'].isna())
        calculation_mask = mask & valid_value_mask
        valid_indices = df.index[calculation_mask]
        
        bond_price = df.loc[valid_indices, '转债_1m_成交均价']
        conversion_value = df.loc[valid_indices, '转股价值']
        premium_rate = (bond_price - conversion_value) / conversion_value * 100
        df.loc[valid_indices, '溢价率'] = premium_rate
    
    return df


def calculate_contribution(df):
    df = df.copy()
    df['eob'] = pd.to_datetime(df['eob'])
    df_sorted = df.sort_values('eob')

    valid_mask = df_sorted['转股价格'] > 0
    df_valid = df_sorted[valid_mask].copy()
    
    stock_prices = df['正股_1m_成交均价']
    bond_prices = df['转债_1m_成交均价']
    times = df['eob'].values
    
    stock_derivative = calculate_derivative(stock_prices, times)
    bond_derivative = calculate_derivative(bond_prices, times)

    
    conversion_ratio = 100 / df_valid['转股价格']

    valid_derivative_mask = (stock_derivative != 0) & (~np.isnan(stock_derivative)) & (~np.isnan(bond_derivative))
    
    delta = np.zeros_like(stock_derivative)
    delta[valid_derivative_mask] = abs(bond_derivative[valid_derivative_mask] )/ abs(stock_derivative[valid_derivative_mask]) 
                                  
    
    delta = np.clip(delta, -8 * conversion_ratio, 8 * conversion_ratio)

    result = pd.Series(np.nan, index=df.index)
    result.loc[df_valid.index] = delta
    
    return result


def create_visualization(df):
    df = df.copy()
    
    # 检查是否存在合并后的数据列名
    if 'eob（当日最后时间）' in df.columns:
        time_column = 'eob（当日最后时间）'
    else:
        time_column = 'eob'  # 回退到原始列名（如果未合并）
    
    df['交易日期'] = pd.to_datetime(df[time_column])
    df['100_转股价比'] = 100 / df['转股价格'].replace(0, np.nan)

    df['开仓线'] = 1.5 * df['100_转股价比']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[time_column],  # 使用检测到的时间列
        y=df['动态delta'],
        mode='lines',
        name='动态delta',
        line=dict(color='rgba(76, 114, 176, 1)', width=2.5),
        hovertemplate=f'日期: %{{x}}<br>动态delta: %{{y:.4f}}<extra></extra>'
    )) 
    
    fig.add_trace(go.Scatter(
        x=df[time_column],  # 使用检测到的时间列
        y=df['100_转股价比'],
        mode='lines',
        name='100/转股价',
        line=dict(color='rgba(221, 132, 82, 1)', width=2.5, dash='dash'),
        hovertemplate=f'日期: %{{x}}<br>100/转股价: %{{y:.4f}}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df[time_column],  # 使用检测到的时间列
        y=df['开仓线'],
        mode='lines',
        name='开仓线',
        line=dict(color='rgba(0, 132, 82, 1)', width=2.5, dash='dash'),
        hovertemplate=f'日期: %{{x}}<br>开仓线: %{{y:.4f}}<extra></extra>'
    ))

    fig.update_layout(
        title='动态delta与100/转股价随时间变化趋势',
        xaxis_title='日期',
        yaxis_title='指标值',
        legend_title='指标类型',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(248, 249, 250, 1.0)',
        font=dict(family="SimHei, WenQuanYi Micro Hei, Heiti TC"),
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified"
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.5)',
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1天", step="day", stepmode="backward"),
                dict(count=7, label="1周", step="day", stepmode="backward"),
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(step="all", label="全部")
            ])
        ),
        rangeslider=dict(visible=True), 
        type="date"
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(220, 220, 220, 0.5)',
        title_standoff=15  
    )
    
    return fig

def time_chonice(df, start_time_choice='09:30:00', end_time_choice='09:33:00',
                 middle_time_start='11:30:00', middle_time_end='13:01:00',
                 finish_time_start='14:57:00', finish_time_end='15:00:00'):
    start_date = input_start_date
    end_date = input_end_date
    tz = df['eob'].dt.tz
    
    if start_date:
        start_date = pd.to_datetime(start_date, format='%Y%m%d').tz_localize(tz)
        df = df[df['eob'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date, format='%Y%m%d').tz_localize(tz)
        df = df[df['eob'] <= end_date]


    df['weekday'] = df['eob'].dt.weekday  
    df = df[df['weekday'] < 5]  
    df = df.drop('weekday', axis=1)
    
    datetime_column = 'eob'
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
        df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    df['time_part'] = df[datetime_column].dt.time
    mask = ((df['time_part'] >= pd.to_datetime(start_time_choice).time()) & 
            (df['time_part'] <= pd.to_datetime(end_time_choice).time())) | \
           ((df['time_part'] >= pd.to_datetime(middle_time_start).time()) & 
            (df['time_part'] <= pd.to_datetime(middle_time_end).time())) | \
           ((df['time_part'] >= pd.to_datetime(finish_time_start).time()) & 
            (df['time_part'] <= pd.to_datetime(finish_time_end).time()))
    
    df = df[~mask].drop('time_part', axis=1)
    return df


# 新增：将分钟数据合并为日数据
def merge_to_daily(df):
    """
    将分钟级数据按天合并，返回日级数据
    
    聚合规则：
    - 日期相关：保留日期（eob取当天最后一个时间戳）
    - 价格类：取每日平均值
    - 转股价格：取每日首个有效值（默认每日不变）
    - 动态delta：取每日平均值
    """
    df = df.copy()
    # 提取日期（从eob中提取日期部分）
    df['date'] = df['eob'].dt.date
    
    # 定义各列的聚合方式
    agg_dict = {
        'eob': 'last',  # 保留当天最后一个时间戳
        '正股_1m_成交均价': 'mean',  # 每日平均正股价格
        '转债_1m_成交均价': 'mean',  # 每日平均转债价格
        '溢价率': 'mean',  # 每日平均溢价率
        '转股价格': 'first',  # 转股价格每日不变，取首个值
        '动态delta': 'mean',  # 每日平均delta
        '交易日期': 'first'  # 交易日期取首个值（与date一致）
    }
    
    # 按日期分组聚合
    daily_df = df.groupby('date').agg(agg_dict).reset_index()
    
    # 调整列顺序，删除临时date列
    daily_df = daily_df.drop(columns=['date']).rename(columns={'eob': 'eob（当日最后时间）'})
    
    return daily_df


def main():
    # stock_file_start = r'C:\Users\Administrator\Desktop\economy\Data\mix_data\Data\juejin_1min\mix'
    stock_file_start = os.path.join(os.path.dirname(__file__), 'Data', 'juejin_1min', 'mix')

    stock_file_end = input_stock_file
    if not stock_file_end.endswith('.csv'):
        stock_file_end += '.csv'
    file_path = os.path.join(stock_file_start, stock_file_end)
    
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
    
    required_columns = ['eob', '交易日期', '正股_1m_成交均价', '转债_1m_成交均价', '溢价率', '转股价格']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"缺少必要列: {', '.join(missing_columns)}")
        return
    
    df['eob'] = pd.to_datetime(df['eob'])
    df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y-%m-%d')
    
    # 计算缺失溢价率
    df = calculate_missing_premium(df)
    
    # 计算股债贡献度（delta）
    delta = calculate_contribution(df)
    df['动态delta'] = delta

    # 时间筛选（过滤特定时间段的分钟数据）
    df = time_chonice(df)
    
    # 异常值处理与插值
    df = interpolate_premium_rate(df)

    # 关键步骤：将分钟数据合并为日数据
    daily_df = merge_to_daily(df)

    # 生成可视化图表（基于日数据）
    fig = create_visualization(daily_df)
    df.to_csv('test1.csv', index=False)
    fig.show()


if __name__ == "__main__":
    input_stock_file = "118016.csv"
    input_start_date = '20250101'
    input_end_date = '20250630'
    main()
    