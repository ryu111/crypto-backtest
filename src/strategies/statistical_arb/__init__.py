"""
統計套利策略模組

提供基於統計模型的套利策略。

策略列表：
- ETHBTCPairsStrategy: ETH/BTC 配對交易
- BasisArbStrategy: 永續/現貨基差套利
"""

from .eth_btc_pairs import ETHBTCPairsStrategy
from .basis_arb import BasisArbStrategy

__all__ = [
    'ETHBTCPairsStrategy',
    'BasisArbStrategy',
]
