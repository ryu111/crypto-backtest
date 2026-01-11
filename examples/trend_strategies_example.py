"""
趨勢策略使用範例

示範如何使用 MA Cross 和 Supertrend 策略。
"""

import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
from src.strategies import create_strategy, list_strategies

# ============================================================
# 範例 1: 查看可用策略
# ============================================================
print('可用的趨勢策略：')
all_strategies = list_strategies()
trend_strategies = [s for s in all_strategies if s.startswith('trend_')]
for strategy in trend_strategies:
    print(f'  - {strategy}')

# ============================================================
# 範例 2: 使用 MA Cross 策略
# ============================================================
print('\n範例 2: MA Cross 策略')
print('-' * 60)

# 建立策略實例
ma_cross = create_strategy(
    'trend_ma_cross',
    fast_period=10,     # 快線週期
    slow_period=30,     # 慢線週期
    stop_loss_atr=2.0,  # 止損 ATR 倍數
)

# 準備測試資料
np.random.seed(42)
n = 200
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
data = pd.DataFrame({
    'open': close + np.random.randn(n) * 0.3,
    'high': close + np.random.rand(n) * 0.8,
    'low': close - np.random.rand(n) * 0.8,
    'close': close,
    'volume': np.random.rand(n) * 1000 + 500
})

# 產生交易訊號
long_entry, long_exit, short_entry, short_exit = ma_cross.generate_signals(data)

print(f'訊號統計：')
print(f'  多單: {long_entry.sum()} 進 / {long_exit.sum()} 出')
print(f'  空單: {short_entry.sum()} 進 / {short_exit.sum()} 出')

# 取得第一筆多單訊號
if long_entry.sum() > 0:
    first_long_idx = data.index[long_entry][0]
    entry_price = data.loc[first_long_idx, 'close']

    # 計算止損
    stop_loss = ma_cross.calculate_stop_loss(
        data.loc[:first_long_idx],
        entry_price,
        'long'
    )

    print(f'\n第一筆多單：')
    print(f'  進場 K 線: {first_long_idx}')
    print(f'  進場價格: {entry_price:.2f}')
    print(f'  止損價格: {stop_loss:.2f}')
    print(f'  風險比例: {((entry_price - stop_loss) / entry_price * 100):.2f}%')

# ============================================================
# 範例 3: 使用 Supertrend 策略
# ============================================================
print('\n範例 3: Supertrend 策略')
print('-' * 60)

# 建立策略實例
supertrend = create_strategy(
    'trend_supertrend',
    period=10,          # ATR 週期
    multiplier=3.0,     # ATR 倍數
)

# 產生交易訊號
long_entry, long_exit, short_entry, short_exit = supertrend.generate_signals(data)

print(f'訊號統計：')
print(f'  多單: {long_entry.sum()} 進 / {long_exit.sum()} 出')
print(f'  空單: {short_entry.sum()} 進 / {short_exit.sum()} 出')

# 取得當前動態止損
current_stop = supertrend.get_stop_loss(data, 'long')
current_price = data['close'].iloc[-1]

print(f'\n當前狀態：')
print(f'  當前價格: {current_price:.2f}')
print(f'  動態止損: {current_stop:.2f}')
print(f'  距離止損: {((current_price - current_stop) / current_price * 100):.2f}%')

# ============================================================
# 範例 4: 使用過濾器
# ============================================================
print('\n範例 4: 使用趨勢過濾器')
print('-' * 60)

# 建立帶過濾器的策略
ma_cross_filtered = create_strategy(
    'trend_ma_cross',
    fast_period=10,
    slow_period=30,
    use_trend_filter=True,      # 啟用趨勢過濾
    trend_filter_period=50,     # 使用 50MA 過濾
)

long_entry_f, long_exit_f, short_entry_f, short_exit_f = ma_cross_filtered.generate_signals(data)

print(f'無過濾器: 多單 {long_entry.sum()} 次，空單 {short_entry.sum()} 次')
print(f'有過濾器: 多單 {long_entry_f.sum()} 次，空單 {short_entry_f.sum()} 次')
print(f'過濾效果: 減少 {(long_entry.sum() + short_entry.sum()) - (long_entry_f.sum() + short_entry_f.sum())} 個訊號')

# ============================================================
# 範例 5: 部位管理
# ============================================================
print('\n範例 5: 計算部位大小')
print('-' * 60)

capital = 10000  # 總資金 $10,000
risk_per_trade = 0.02  # 單筆風險 2%

if long_entry.sum() > 0:
    first_long_idx = data.index[long_entry][0]
    entry_price = data.loc[first_long_idx, 'close']
    stop_loss = ma_cross.calculate_stop_loss(
        data.loc[:first_long_idx],
        entry_price,
        'long'
    )

    # 計算部位大小
    position_size = ma_cross.position_size(
        capital=capital,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        risk_per_trade=risk_per_trade,
        max_position_pct=1.0
    )

    print(f'資金管理：')
    print(f'  總資金: ${capital:,.0f}')
    print(f'  風險比例: {risk_per_trade * 100}%')
    print(f'  風險金額: ${capital * risk_per_trade:,.0f}')
    print(f'  進場價格: ${entry_price:.2f}')
    print(f'  止損價格: ${stop_loss:.2f}')
    print(f'  部位大小: {position_size:.4f} 個合約')
    print(f'  部位價值: ${position_size * entry_price:,.2f}')

# ============================================================
# 範例 6: 策略比較
# ============================================================
print('\n範例 6: 策略訊號比較')
print('-' * 60)

# MA Cross 訊號
ma_long, ma_long_exit, ma_short, ma_short_exit = ma_cross.generate_signals(data)

# Supertrend 訊號
st_long, st_long_exit, st_short, st_short_exit = supertrend.generate_signals(data)

print(f'MA Cross 策略：')
print(f'  交易次數: {ma_long.sum() + ma_short.sum()}')
print(f'  多空比: {ma_long.sum()} : {ma_short.sum()}')

print(f'\nSupertrend 策略：')
print(f'  交易次數: {st_long.sum() + st_short.sum()}')
print(f'  多空比: {st_long.sum()} : {st_short.sum()}')

# 找出共同訊號
common_long = ma_long & st_long
common_short = ma_short & st_short

print(f'\n共同訊號：')
print(f'  共同做多: {common_long.sum()} 次')
print(f'  共同做空: {common_short.sum()} 次')

if common_long.sum() > 0:
    print(f'  共同做多位置: {data.index[common_long].tolist()}')

print('\n' + '=' * 60)
print('範例完成！')
print('=' * 60)
