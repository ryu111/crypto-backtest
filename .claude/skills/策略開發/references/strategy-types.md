# 策略類型詳解

## 趨勢跟隨策略

### 核心理念

「趨勢是你的朋友」—— 跟隨市場方向交易

### 特點

| 面向 | 說明 |
|------|------|
| 勝率 | 40-50%（較低） |
| 盈虧比 | 2:1 - 3:1（較高） |
| 持倉時間 | 中長期 |
| 適用市場 | 趨勢明確時期 |

### 常用策略

1. **雙均線交叉**
2. **突破策略**
3. **Supertrend**
4. **Channel Breakout**

### 範例：Donchian Channel

```python
class DonchianBreakout:
    params = {'period': 20}

    def generate_signals(self, data):
        high = data['high']
        low = data['low']
        close = data['close']

        upper = high.rolling(self.params['period']).max()
        lower = low.rolling(self.params['period']).min()

        long_entry = close > upper.shift(1)
        long_exit = close < lower.shift(1)
        short_entry = close < lower.shift(1)
        short_exit = close > upper.shift(1)

        return long_entry, long_exit, short_entry, short_exit
```

## 均值回歸策略

### 核心理念

價格終將回歸均值

### 特點

| 面向 | 說明 |
|------|------|
| 勝率 | 55-65%（較高） |
| 盈虧比 | 1:1 - 1.5:1 |
| 持倉時間 | 短期 |
| 適用市場 | 震盪盤整 |

### 常用策略

1. **RSI 超買超賣**
2. **布林帶回歸**
3. **價格偏離均線**

### 範例：布林帶均值回歸

```python
class BollingerMeanReversion:
    params = {'period': 20, 'std': 2.0}

    def generate_signals(self, data):
        close = data['close']

        ma = close.rolling(self.params['period']).mean()
        std = close.rolling(self.params['period']).std()

        upper = ma + self.params['std'] * std
        lower = ma - self.params['std'] * std

        # 觸及下軌做多，觸及上軌做空
        long_entry = close < lower
        long_exit = close > ma
        short_entry = close > upper
        short_exit = close < ma

        return long_entry, long_exit, short_entry, short_exit
```

## 動量策略

### 核心理念

強者恆強，弱者恆弱

### 特點

| 面向 | 說明 |
|------|------|
| 勝率 | 45-55% |
| 盈虧比 | 1.5:1 - 2:1 |
| 持倉時間 | 短中期 |
| 適用市場 | 波動市場 |

### 常用策略

1. **MACD 動量**
2. **ROC 突破**
3. **相對強弱**

### 範例：MACD 動量

```python
class MACDMomentum:
    params = {'fast': 12, 'slow': 26, 'signal': 9}

    def generate_signals(self, data):
        close = data['close']

        fast_ema = close.ewm(span=self.params['fast']).mean()
        slow_ema = close.ewm(span=self.params['slow']).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=self.params['signal']).mean()
        histogram = macd - signal

        # 柱狀圖由負轉正做多
        long_entry = (histogram > 0) & (histogram.shift(1) <= 0)
        long_exit = (histogram < 0) & (histogram.shift(1) >= 0)

        return long_entry, long_exit, None, None
```

## 統計套利策略

### 核心理念

利用統計關係獲利

### 特點

| 面向 | 說明 |
|------|------|
| 勝率 | 50-60% |
| 盈虧比 | 可變 |
| 持倉時間 | 極短到短期 |
| 適用市場 | 任何 |

### 常用策略

1. **配對交易**
2. **基差套利**
3. **資金費率套利**

### 範例：ETH/BTC 配對

```python
class ETHBTCPairs:
    params = {'period': 20, 'threshold': 2.0}

    def generate_signals(self, eth_data, btc_data):
        ratio = eth_data['close'] / btc_data['close']
        ma = ratio.rolling(self.params['period']).mean()
        std = ratio.rolling(self.params['period']).std()

        z_score = (ratio - ma) / std

        # 比率偏高：做空 ETH，做多 BTC
        short_eth = z_score > self.params['threshold']
        # 比率偏低：做多 ETH，做空 BTC
        long_eth = z_score < -self.params['threshold']

        return long_eth, short_eth
```

## 資金費率策略

### 核心理念

利用永續合約資金費率機制

### 特點

| 面向 | 說明 |
|------|------|
| 勝率 | 70%+（高） |
| 盈虧比 | 可變 |
| 持倉時間 | 8h 以上 |
| 適用市場 | 高費率期間 |

### 策略類型

1. **Delta Neutral 套利**
2. **費率預測交易**
3. **結算時點交易**

詳見 `perpetual-specific` skill

## 策略選擇指南

| 市場狀態 | 推薦策略 |
|----------|----------|
| 明確趨勢 | 趨勢跟隨 |
| 區間震盪 | 均值回歸 |
| 高波動 | 動量突破 |
| 任何時候 | 資金費率套利 |

## BTC vs ETH 策略差異

| 面向 | BTC | ETH |
|------|-----|-----|
| 趨勢策略 | 更穩定 | 波動更大 |
| 均值回歸 | 區間較寬 | 區間較窄 |
| 參數選擇 | 較長週期 | 可用較短 |
