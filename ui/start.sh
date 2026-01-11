#!/bin/bash

# AI 回測系統 UI 啟動腳本

echo "🚀 AI 回測系統 UI"
echo "=================="
echo ""

# 檢查依賴
echo "檢查依賴..."
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Streamlit 未安裝"
    echo "請執行: pip install -r ui/requirements.txt"
    exit 1
fi

python3 -c "import plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Plotly 未安裝"
    echo "請執行: pip install -r ui/requirements.txt"
    exit 1
fi

echo "✅ 依賴檢查通過"
echo ""

# 檢查資料
if [ ! -f "learning/experiments.json" ]; then
    echo "⚠️  未找到實驗資料 (learning/experiments.json)"
    echo "提示: 執行回測後會自動產生資料"
    echo ""
fi

# 啟動 Streamlit
echo "啟動 UI..."
echo "瀏覽器訪問: http://localhost:8501"
echo ""

streamlit run ui/Home.py --server.port 8501
