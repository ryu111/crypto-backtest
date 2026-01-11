#!/bin/bash
# Streamlit UI 啟動腳本

cd "$(dirname "$0")"

echo "正在啟動 AI 合約回測系統 UI..."
streamlit run ui/app.py
