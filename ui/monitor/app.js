/**
 * UltimateLoop Monitor - 前端應用
 */

class MonitorApp {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;

        this.state = {
            status: 'idle',
            currentIteration: 0,
            totalIterations: 0,
            startTime: null,
            sharpeHistory: [],
            bestSharpe: 0,
            avgSharpe: 0,
            successRate: 0,
            strategyDistribution: {},
        };

        this.chart = null;
        this.events = [];
        this.maxEvents = 50;

        this.init();
    }

    init() {
        this.initChart();
        this.connect();
        this.startTimer();
        // Fallback: 如果 WebSocket 連接失敗，使用 API 輪詢
        this.startPolling();
    }

    startPolling() {
        // 每 2 秒從 API 獲取狀態
        setInterval(async () => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                try {
                    const response = await fetch('/api/state');
                    const data = await response.json();
                    this.syncState(data);
                } catch (error) {
                    console.error('API 輪詢失敗:', error);
                }
            }
        }, 2000);

        // 每秒從 API 獲取系統狀態（CPU/RAM）
        setInterval(async () => {
            try {
                const response = await fetch('/api/system-stats');
                const data = await response.json();
                this.updateSystemStats(data);
            } catch (error) {
                // 靜默失敗
            }
        }, 1000);
    }

    // ============= WebSocket =============

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log('連接 WebSocket:', wsUrl);

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket 已連接');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                this.addEvent('系統已連接', 'success');
            };

            this.ws.onclose = () => {
                console.log('WebSocket 斷開');
                this.updateConnectionStatus(false);
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket 錯誤:', error);
            };

            this.ws.onmessage = (event) => {
                this.handleMessage(event.data);
            };
        } catch (error) {
            console.error('WebSocket 連接失敗:', error);
            this.scheduleReconnect();
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('已達最大重連次數');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        console.log(`${delay}ms 後重連 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(() => this.connect(), delay);
    }

    updateConnectionStatus(connected) {
        const el = document.getElementById('connection-status');
        const dot = el.querySelector('.status-dot');
        const text = el.querySelector('span:last-child');

        if (connected) {
            el.classList.remove('disconnected');
            el.classList.add('connected');
            dot.classList.remove('disconnected');
            text.textContent = '已連接';
        } else {
            el.classList.remove('connected');
            el.classList.add('disconnected');
            dot.classList.add('disconnected');
            text.textContent = '斷線';
        }

        // WebSocket 狀態
        const wsStatus = document.getElementById('ws-status');
        const wsText = document.getElementById('ws-status-text');
        if (connected) {
            wsStatus.classList.remove('disconnected');
            wsText.textContent = '連接中';
        } else {
            wsStatus.classList.add('disconnected');
            wsText.textContent = '斷線';
        }
    }

    // ============= 訊息處理 =============

    handleMessage(data) {
        if (data === 'ping') {
            this.ws.send('pong');
            return;
        }

        try {
            const msg = JSON.parse(data);

            switch (msg.type) {
                case 'state_sync':
                    this.syncState(msg.data);
                    break;
                case 'loop_start':
                    this.onLoopStart(msg.data);
                    break;
                case 'loop_complete':
                    this.onLoopComplete(msg.data);
                    break;
                case 'loop_error':
                    this.onLoopError(msg.data);
                    break;
                case 'iteration_start':
                    this.onIterationStart(msg.data);
                    break;
                case 'iteration_complete':
                    this.onIterationComplete(msg.data);
                    break;
                case 'iteration_error':
                    this.onIterationError(msg.data);
                    break;
                case 'system_stats':
                    this.updateSystemStats(msg.data);
                    break;
                case 'best_strategy_update':
                    this.onBestStrategyUpdate(msg.data);
                    break;
            }
        } catch (error) {
            console.error('訊息解析錯誤:', error);
        }
    }

    syncState(data) {
        this.state = { ...this.state, ...data };

        // 更新狀態徽章
        if (data.status) {
            this.updateStatusBadge(data.status);
            if (data.status === 'running') {
                this.updateEngineStatus(true);
            }
        }

        this.updateUI();

        // 恢復圖表資料
        if (data.sharpeHistory && data.sharpeHistory.length > 0) {
            this.chart.data.labels = data.sharpeHistory.map((_, i) => i + 1);
            this.chart.data.datasets[0].data = [...data.sharpeHistory];
            this.chart.update('none');
        }

        // 更新最佳策略
        if (data.bestSharpe > 0) {
            this.updateBestStrategy({
                iteration: data.bestIteration,
                strategyName: data.bestStrategyName,
                sharpe: data.bestSharpe,
                totalReturn: data.bestReturn,
                maxDrawdown: data.bestDrawdown,
                params: data.bestParams,
            });
        }

        // 更新策略分佈
        if (data.strategyDistribution && data.strategyDistribution.strategy_counts) {
            this.state.strategyDistribution = data.strategyDistribution.strategy_counts;
            this.updateStrategyDistribution();
        }
    }

    // ============= 事件處理 =============

    onLoopStart(data) {
        this.state.status = 'running';
        this.state.totalIterations = data.totalIterations;
        this.state.currentIteration = 0;
        this.state.startTime = new Date();
        this.state.sharpeHistory = [];

        this.chart.data.labels = [];
        this.chart.data.datasets[0].data = [];
        this.chart.update('none');

        this.updateStatusBadge('running');
        this.updateEngineStatus(true);
        this.addEvent(`開始執行 ${data.totalIterations} 次迭代`, 'success');
        this.updateUI();
    }

    onLoopComplete(data) {
        this.state.status = 'completed';
        this.updateStatusBadge('completed');
        this.updateEngineStatus(false);
        this.addEvent('回測完成!', 'success');
    }

    onLoopError(data) {
        this.state.status = 'error';
        this.updateStatusBadge('error');
        this.updateEngineStatus(false);
        this.addEvent(`錯誤: ${data.error}`, 'error');
    }

    onIterationStart(data) {
        this.state.currentIteration = data.iteration;
        this.updateProgress();
    }

    onIterationComplete(data) {
        this.state.currentIteration = data.iteration;
        this.state.sharpeHistory.push(data.sharpe);
        this.state.avgSharpe = data.avgSharpe || 0;
        this.state.successRate = data.successRate || 0;

        if (data.sharpe > this.state.bestSharpe) {
            this.state.bestSharpe = data.sharpe;
        }

        // 更新圖表
        this.addChartPoint(data.iteration, data.sharpe);

        // 更新策略分佈
        if (data.strategyName) {
            if (!this.state.strategyDistribution[data.strategyName]) {
                this.state.strategyDistribution[data.strategyName] = 0;
            }
            this.state.strategyDistribution[data.strategyName]++;
            this.updateStrategyDistribution();
        }

        // 事件
        let eventText = `#${data.iteration} ${data.strategyName} Sharpe=${data.sharpe.toFixed(2)}`;
        let eventType = 'success';

        if (data.isNewBest) {
            eventText += ' 🏆 新最佳!';
            eventType = 'new-best';
        }

        this.addEvent(eventText, eventType);
        this.updateUI();
    }

    onIterationError(data) {
        this.addEvent(`#${data.iteration} 錯誤: ${data.error}`, 'error');
    }

    onBestStrategyUpdate(data) {
        this.updateBestStrategy(data);
    }

    // ============= UI 更新 =============

    updateUI() {
        // 進度
        this.updateProgress();

        // 指標
        document.getElementById('best-sharpe').textContent = this.state.bestSharpe.toFixed(2);
        document.getElementById('success-rate').textContent = `${this.state.successRate.toFixed(1)}%`;

        // 平均報酬（從最佳策略或累積計算）
        // 如果值 > 1，說明已經是百分比形式；否則需要乘以 100
        if (this.state.bestReturn) {
            const returnVal = this.state.bestReturn > 1
                ? this.state.bestReturn
                : this.state.bestReturn * 100;
            document.getElementById('avg-return').textContent = `${returnVal.toFixed(1)}%`;
        }

        // 最大回撤（通常是 0-1 的小數）
        if (this.state.bestDrawdown) {
            const ddVal = this.state.bestDrawdown > 1
                ? this.state.bestDrawdown
                : this.state.bestDrawdown * 100;
            document.getElementById('max-drawdown').textContent = `${ddVal.toFixed(1)}%`;
        }
    }

    updateProgress() {
        const current = this.state.currentIteration;
        const total = this.state.totalIterations || 1;
        const percent = (current / total * 100).toFixed(1);

        document.getElementById('current-iteration').textContent = current;
        document.getElementById('total-iterations').textContent = total;
        document.getElementById('progress-percent').textContent = `${percent}%`;
        document.getElementById('progress-bar').style.width = `${percent}%`;
    }

    updateStatusBadge(status) {
        const badge = document.getElementById('status-badge');
        badge.className = `status-badge ${status}`;

        const statusText = {
            'idle': 'IDLE',
            'running': 'RUNNING',
            'completed': 'COMPLETED',
            'error': 'ERROR',
        };

        badge.textContent = statusText[status] || status.toUpperCase();
    }

    updateEngineStatus(active) {
        const dot = document.getElementById('engine-status');
        const text = document.getElementById('engine-status-text');

        if (active) {
            dot.classList.remove('disconnected');
            text.textContent = '運行中';
        } else {
            dot.classList.add('disconnected');
            text.textContent = '停止';
        }
    }

    updateSystemStats(data) {
        // CPU
        document.getElementById('cpu-bar').style.width = `${data.cpuPercent}%`;
        document.getElementById('cpu-value').textContent = `${Math.round(data.cpuPercent)}%`;

        // RAM
        document.getElementById('ram-bar').style.width = `${data.memoryPercent}%`;
        document.getElementById('ram-value').textContent = `${Math.round(data.memoryPercent)}%`;

        // GPU
        if (data.gpuPercent !== null && data.gpuPercent !== undefined) {
            document.getElementById('gpu-row').style.display = 'flex';
            document.getElementById('gpu-bar').style.width = `${data.gpuPercent}%`;
            document.getElementById('gpu-value').textContent = `${Math.round(data.gpuPercent)}%`;
        }
    }

    updateStrategyDistribution() {
        const list = document.getElementById('strategy-list');
        const strategies = Object.entries(this.state.strategyDistribution)
            .sort((a, b) => b[1] - a[1]);

        if (strategies.length === 0) return;

        const maxCount = strategies[0][1];

        list.innerHTML = strategies.map(([name, count]) => {
            const percent = (count / maxCount * 100).toFixed(0);
            return `
                <div class="strategy-item">
                    <div class="strategy-bar" style="width: ${percent}%"></div>
                    <span class="strategy-name">${name}</span>
                    <span class="strategy-count">${count}</span>
                </div>
            `;
        }).join('');
    }

    updateBestStrategy(data) {
        document.getElementById('no-best-strategy').style.display = 'none';
        document.getElementById('best-strategy-card').style.display = 'block';

        document.getElementById('best-strategy-name').textContent = data.strategyName || '-';
        document.getElementById('best-strategy-iteration').textContent = `#${data.iteration}`;
        document.getElementById('best-strategy-sharpe').textContent = (data.sharpe || 0).toFixed(2);

        // Return: 如果 > 1 說明已經是百分比形式，否則需要乘以 100
        const returnVal = data.totalReturn || 0;
        const displayReturn = returnVal > 1 ? returnVal : returnVal * 100;
        document.getElementById('best-strategy-return').textContent = `${displayReturn.toFixed(1)}%`;

        // Drawdown: 通常是 0-1 的小數，需要乘以 100
        const ddVal = data.maxDrawdown || 0;
        const displayDD = ddVal > 1 ? ddVal : ddVal * 100;
        document.getElementById('best-strategy-drawdown').textContent = `${displayDD.toFixed(1)}%`;

        const params = data.params || {};
        const paramsText = Object.entries(params)
            .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(4) : v}`)
            .join(', ');
        document.getElementById('best-strategy-params').textContent = paramsText || '-';

        // 更新全域狀態
        this.state.bestSharpe = data.sharpe;
        this.state.bestReturn = data.totalReturn;
        this.state.bestDrawdown = data.maxDrawdown;
    }

    // ============= 圖表 =============

    initChart() {
        const ctx = document.getElementById('sharpe-chart').getContext('2d');

        const gradient = ctx.createLinearGradient(0, 0, 0, 250);
        gradient.addColorStop(0, 'rgba(0, 245, 255, 0.3)');
        gradient.addColorStop(1, 'rgba(191, 0, 255, 0.0)');

        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Sharpe Ratio',
                    data: [],
                    borderColor: '#00f5ff',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#00f5ff',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false,
                    },
                    tooltip: {
                        backgroundColor: '#12121a',
                        titleColor: '#ffffff',
                        bodyColor: '#00f5ff',
                        borderColor: '#1e1e2e',
                        borderWidth: 1,
                        callbacks: {
                            label: (ctx) => `Sharpe: ${ctx.raw.toFixed(3)}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(30, 30, 46, 0.5)',
                        },
                        ticks: {
                            color: '#8888aa',
                            maxTicksLimit: 10,
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(30, 30, 46, 0.5)',
                        },
                        ticks: {
                            color: '#8888aa',
                        },
                        suggestedMin: 0,
                    }
                },
                animation: {
                    duration: 300,
                }
            }
        });
    }

    addChartPoint(iteration, sharpe) {
        this.chart.data.labels.push(iteration);
        this.chart.data.datasets[0].data.push(sharpe);

        // 限制資料點數量
        const maxPoints = 200;
        if (this.chart.data.labels.length > maxPoints) {
            this.chart.data.labels.shift();
            this.chart.data.datasets[0].data.shift();
        }

        this.chart.update('none');
    }

    // ============= 事件日誌 =============

    addEvent(content, type = 'success') {
        const now = new Date();
        const time = now.toTimeString().slice(0, 8);

        this.events.unshift({ time, content, type });

        if (this.events.length > this.maxEvents) {
            this.events.pop();
        }

        this.renderEvents();
    }

    renderEvents() {
        const list = document.getElementById('events-list');

        list.innerHTML = this.events.map(event => {
            let badge = '';
            if (event.type === 'new-best') {
                badge = '<span class="event-badge new-best">🏆</span>';
            } else if (event.type === 'error') {
                badge = '<span class="event-badge error">ERROR</span>';
            }

            return `
                <div class="event-item">
                    <span class="event-time">${event.time}</span>
                    <span class="event-content">${event.content}</span>
                    ${badge}
                </div>
            `;
        }).join('');
    }

    // ============= 計時器 =============

    startTimer() {
        setInterval(() => {
            if (this.state.status === 'running' && this.state.startTime) {
                const elapsed = Math.floor((new Date() - new Date(this.state.startTime)) / 1000);
                const hours = Math.floor(elapsed / 3600).toString().padStart(2, '0');
                const minutes = Math.floor((elapsed % 3600) / 60).toString().padStart(2, '0');
                const seconds = (elapsed % 60).toString().padStart(2, '0');
                document.getElementById('timer').textContent = `${hours}:${minutes}:${seconds}`;
            }
        }, 1000);
    }
}

// 啟動應用
const app = new MonitorApp();
