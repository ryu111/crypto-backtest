"""
測試趨勢策略

驗證 MA Cross 和 Supertrend 策略的功能。
"""

import pandas as pd
import numpy as np
from src.strategies import create_strategy

# 產生測試資料（更長的資料集）
np.random.seed(42)
n = 300

# 模擬有明顯趨勢變化的價格
trend1 = np.linspace(100, 110, n//3)  # 上升趨勢
trend2 = np.linspace(110, 105, n//3)  # 下降趨勢
trend3 = np.linspace(105, 115, n//3)  # 再次上升

trend = np.concatenate([trend1, trend2, trend3])
noise = np.random.randn(n) * 0.8
close = trend + noise

# 確保 high >= close, low <= close
data = pd.DataFrame({
    'close': close,
    'high': close + np.random.rand(n) * 1.0,
    'low': close - np.random.rand(n) * 1.0,
    'volume': np.random.rand(n) * 2000 + 1000
})
data['open'] = data['close'].shift(1).fillna(data['close'][0])

print('=' * 60)
print('測試趨勢策略')
print('=' * 60)

# ============================================================
# 測試 1: MA Cross 策略
# ============================================================
print('\n1. MA Cross 策略測試')
print('-' * 60)

ma_cross = create_strategy(
    'trend_ma_cross',
    fast_period=10,
    slow_period=30,
    stop_loss_atr=2.0
)

print(f'策略名稱: {ma_cross.name}')
print(f'策略類型: {ma_cross.strategy_type}')
print(f'策略版本: {ma_cross.version}')
print(f'參數: {ma_cross.params}')

# 計算指標
indicators = ma_cross.calculate_indicators(data)
print(f'\n指標計算結果：')
print(f'  SMA Fast (最後 5 個): {indicators["sma_fast"].tail().values}')
print(f'  SMA Slow (最後 5 個): {indicators["sma_slow"].tail().values}')
print(f'  ATR (最後 5 個): {indicators["atr"].tail().values}')

# 產生訊號
long_entry, long_exit, short_entry, short_exit = ma_cross.generate_signals(data)
print(f'\n訊號統計：')
print(f'  多單進場: {long_entry.sum()} 次')
print(f'  多單出場: {long_exit.sum()} 次')
print(f'  空單進場: {short_entry.sum()} 次')
print(f'  空單出場: {short_exit.sum()} 次')

# 顯示訊號位置
if long_entry.sum() > 0:
    print(f'\n多單進場位置: {data.index[long_entry].tolist()}')
if short_entry.sum() > 0:
    print(f'空單進場位置: {data.index[short_entry].tolist()}')

# 測試止損計算
if long_entry.sum() > 0:
    first_entry_idx = data.index[long_entry][0]
    entry_price = data.loc[first_entry_idx, 'close']
    stop_loss = ma_cross.calculate_stop_loss(
        data.loc[:first_entry_idx],
        entry_price,
        'long'
    )
    print(f'\n止損測試（第一筆多單）：')
    print(f'  進場價格: {entry_price:.2f}')
    print(f'  止損價格: {stop_loss:.2f}')
    print(f'  止損距離: {entry_price - stop_loss:.2f} ({((entry_price - stop_loss) / entry_price * 100):.2f}%)')

# ============================================================
# 測試 2: Supertrend 策略
# ============================================================
print('\n' + '=' * 60)
print('2. Supertrend 策略測試')
print('-' * 60)

supertrend = create_strategy(
    'trend_supertrend',
    period=10,
    multiplier=3.0,
    use_volume_filter=False
)

print(f'策略名稱: {supertrend.name}')
print(f'策略類型: {supertrend.strategy_type}')
print(f'策略版本: {supertrend.version}')
print(f'參數: {supertrend.params}')

# 計算指標
indicators = supertrend.calculate_indicators(data)
print(f'\n指標計算結果：')
print(f'  Supertrend (最後 5 個): {indicators["supertrend"].tail().values}')
print(f'  Direction (最後 5 個): {indicators["direction"].tail().values}')
print(f'  ATR (最後 5 個): {indicators["atr"].tail().values}')

# 產生訊號
long_entry, long_exit, short_entry, short_exit = supertrend.generate_signals(data)
print(f'\n訊號統計：')
print(f'  多單進場: {long_entry.sum()} 次')
print(f'  多單出場: {long_exit.sum()} 次')
print(f'  空單進場: {short_entry.sum()} 次')
print(f'  空單出場: {short_exit.sum()} 次')

# 顯示訊號位置
if long_entry.sum() > 0:
    print(f'\n多單進場位置: {data.index[long_entry].tolist()}')
if short_entry.sum() > 0:
    print(f'空單進場位置: {data.index[short_entry].tolist()}')

# 測試動態止損
current_stop = supertrend.get_stop_loss(data, 'long')
print(f'\n動態止損（當前）：')
print(f'  當前價格: {data["close"].iloc[-1]:.2f}')
print(f'  Supertrend 止損: {current_stop:.2f}')

# ============================================================
# 測試 3: 參數驗證
# ============================================================
print('\n' + '=' * 60)
print('3. 參數驗證測試')
print('-' * 60)

# 測試無效參數
try:
    invalid_ma = create_strategy(
        'trend_ma_cross',
        fast_period=30,  # 快線大於慢線（無效）
        slow_period=10
    )
    print('❌ 參數驗證失敗：應該拒絕快線大於慢線')
except ValueError as e:
    print('✓ 參數驗證通過：正確拒絕無效參數')
    print(f'  錯誤訊息: {e}')

# ============================================================
# 測試 4: 過濾器測試
# ============================================================
print('\n' + '=' * 60)
print('4. 過濾器測試')
print('-' * 60)

# 測試趨勢過濾器
ma_cross_filtered = create_strategy(
    'trend_ma_cross',
    fast_period=10,
    slow_period=30,
    use_trend_filter=True,
    trend_filter_period=50
)

long_entry_f, long_exit_f, short_entry_f, short_exit_f = ma_cross_filtered.generate_signals(data)
print(f'MA Cross (有趨勢過濾)：')
print(f'  多單進場: {long_entry_f.sum()} 次')
print(f'  空單進場: {short_entry_f.sum()} 次')

# 測試成交量過濾器
supertrend_filtered = create_strategy(
    'trend_supertrend',
    period=10,
    multiplier=3.0,
    use_volume_filter=True,
    volume_threshold=1.2
)

long_entry_f, long_exit_f, short_entry_f, short_exit_f = supertrend_filtered.generate_signals(data)
print(f'\nSupertrend (有成交量過濾)：')
print(f'  多單進場: {long_entry_f.sum()} 次')
print(f'  空單進場: {short_entry_f.sum()} 次')

print('\n' + '=' * 60)
print('✓ 所有測試完成！')
print('=' * 60)
