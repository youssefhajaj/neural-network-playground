/**
 * Visualization Module
 * Handles all canvas rendering for the neural network playground
 */

class Visualizer {
    constructor() {
        this.canvases = {};
        this.contexts = {};
        this.neuronPositions = [];
        this.hoveredNeuron = null;
        this.resolution = 'low';

        this.colors = {
            bg: '#030712',
            class0: '#ef4444',
            class1: '#6366f1',
            neuron: '#1e293b',
            connection: 'rgba(148, 163, 184, 0.2)',
            text: '#94a3b8',
            grid: 'rgba(148, 163, 184, 0.05)',
            loss: '#8b5cf6',
            accuracy: '#10b981'
        };
    }

    init() {
        this.canvases = {
            network: document.getElementById('network-canvas'),
            decision: document.getElementById('decision-canvas'),
            metrics: document.getElementById('metrics-canvas'),
            weights: document.getElementById('weights-canvas')
        };

        this.setupCanvases();
        this.setupEventListeners();
        this.drawDatasetPreviews();
    }

    setupCanvases() {
        for (const [name, canvas] of Object.entries(this.canvases)) {
            if (canvas) {
                // Get the actual displayed size
                const rect = canvas.getBoundingClientRect();
                const width = Math.max(rect.width, 200);
                const height = Math.max(rect.height, 150);
                const dpr = Math.min(window.devicePixelRatio || 1, 2); // Cap at 2x for performance

                canvas.width = width * dpr;
                canvas.height = height * dpr;

                const ctx = canvas.getContext('2d');
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                this.contexts[name] = ctx;
            }
        }
    }

    setupEventListeners() {
        window.addEventListener('resize', Utils.debounce(() => {
            this.setupCanvases();
        }, 200));

        if (this.canvases.network) {
            this.canvases.network.addEventListener('mousemove', (e) => this.handleNetworkHover(e));
            this.canvases.network.addEventListener('mouseleave', () => this.clearHover());
        }
    }

    drawDatasetPreviews() {
        const previews = document.querySelectorAll('.dataset-preview');
        previews.forEach(canvas => {
            const type = canvas.dataset.preview;
            if (type && Datasets[type]) {
                this.drawMiniDataset(canvas, type);
            }
        });
    }

    drawMiniDataset(canvas, type) {
        const size = 40;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = size * dpr;
        canvas.height = size * dpr;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        ctx.fillStyle = this.colors.bg;
        ctx.fillRect(0, 0, size, size);

        const data = Datasets[type](50, 0.05);

        for (let i = 0; i < data.data.length; i++) {
            const [x, y] = data.data[i];
            const label = data.labels[i][0];

            const px = (x + 1) / 2 * size;
            const py = (-y + 1) / 2 * size;

            ctx.beginPath();
            ctx.arc(px, py, 2, 0, Math.PI * 2);
            ctx.fillStyle = label > 0.5 ? this.colors.class1 : this.colors.class0;
            ctx.fill();
        }
    }

    handleNetworkHover(e) {
        if (!this.neuronPositions.length) return;

        const rect = this.canvases.network.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        let found = null;
        for (const pos of this.neuronPositions) {
            const dx = x - pos.x;
            const dy = y - pos.y;
            if (Math.sqrt(dx * dx + dy * dy) < pos.radius + 2) {
                found = pos;
                break;
            }
        }

        if (found) {
            this.hoveredNeuron = found;
            this.showInspector(found);
        }
    }

    clearHover() {
        this.hoveredNeuron = null;
        this.hideInspector();
    }

    showInspector(neuron) {
        const container = document.getElementById('inspector-content');
        if (!container) return;

        const layerName = neuron.layer === 0 ? 'Input' :
            neuron.layer === neuron.totalLayers - 1 ? 'Output' :
            `Hidden ${neuron.layer}`;

        container.innerHTML = `
            <div class="inspector-data">
                <div class="inspector-row">
                    <span>Layer</span>
                    <span>${layerName}</span>
                </div>
                <div class="inspector-row">
                    <span>Neuron</span>
                    <span>#${neuron.index + 1}</span>
                </div>
                <div class="inspector-row">
                    <span>Activation</span>
                    <span>${neuron.value !== undefined ? neuron.value.toFixed(4) : '—'}</span>
                </div>
            </div>
        `;
    }

    hideInspector() {
        const container = document.getElementById('inspector-content');
        if (!container) return;

        container.innerHTML = `
            <div class="inspector-placeholder">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
                <span>Hover over a neuron to inspect</span>
            </div>
        `;
    }

    // Draw network architecture
    drawNetwork(network, layerSizes) {
        const ctx = this.contexts.network;
        if (!ctx) return;

        const canvas = this.canvases.network;
        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);

        ctx.fillStyle = this.colors.bg;
        ctx.fillRect(0, 0, width, height);

        const numLayers = layerSizes.length;
        const maxNeurons = Math.max(...layerSizes);
        const layerSpacing = width / (numLayers + 1);
        const neuronRadius = Math.min(18, Math.min(height / (maxNeurons * 2.5), 20));

        const neuronValues = network ? network.getNeuronValues() : null;
        const weights = network ? network.getWeightInfo() : null;

        this.neuronPositions = [];

        // Draw connections
        if (weights) {
            for (let l = 0; l < numLayers - 1; l++) {
                const x1 = layerSpacing * (l + 1);
                const x2 = layerSpacing * (l + 2);

                for (let i = 0; i < layerSizes[l]; i++) {
                    const y1 = this.getNeuronY(i, layerSizes[l], height);

                    for (let j = 0; j < layerSizes[l + 1]; j++) {
                        const y2 = this.getNeuronY(j, layerSizes[l + 1], height);
                        const w = weights[l].data[j][i];

                        const absW = Math.abs(w);
                        const alpha = Math.min(0.7, absW * 0.4 + 0.1);
                        const lineWidth = Math.min(2.5, absW * 1.5 + 0.5);

                        ctx.strokeStyle = w > 0 ?
                            `rgba(99, 102, 241, ${alpha})` :
                            `rgba(239, 68, 68, ${alpha})`;
                        ctx.lineWidth = lineWidth;

                        ctx.beginPath();
                        ctx.moveTo(x1 + neuronRadius, y1);
                        ctx.lineTo(x2 - neuronRadius, y2);
                        ctx.stroke();
                    }
                }
            }
        }

        // Draw neurons
        for (let l = 0; l < numLayers; l++) {
            const x = layerSpacing * (l + 1);

            for (let i = 0; i < layerSizes[l]; i++) {
                const y = this.getNeuronY(i, layerSizes[l], height);

                let value = 0;
                if (neuronValues && neuronValues[l]) {
                    value = neuronValues[l][i] || 0;
                }

                // Neuron gradient
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, neuronRadius);

                const intensity = Math.abs(value);
                if (value > 0) {
                    gradient.addColorStop(0, `rgba(99, 102, 241, ${0.3 + intensity * 0.7})`);
                    gradient.addColorStop(1, `rgba(99, 102, 241, ${0.1 + intensity * 0.4})`);
                } else {
                    gradient.addColorStop(0, `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`);
                    gradient.addColorStop(1, `rgba(239, 68, 68, ${0.1 + intensity * 0.4})`);
                }

                ctx.beginPath();
                ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
                ctx.fillStyle = gradient;
                ctx.fill();

                ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();

                this.neuronPositions.push({
                    x, y, radius: neuronRadius,
                    layer: l, index: i, value,
                    totalLayers: numLayers
                });
            }
        }

        // Layer labels
        ctx.fillStyle = this.colors.text;
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';

        for (let l = 0; l < numLayers; l++) {
            const x = layerSpacing * (l + 1);
            const label = l === 0 ? 'Input' : l === numLayers - 1 ? 'Output' : `H${l}`;
            ctx.fillText(label, x, height - 8);
        }
    }

    getNeuronY(index, total, height) {
        const spacing = (height - 40) / (total + 1);
        return 20 + spacing * (index + 1);
    }

    // Draw decision boundary
    drawDecisionBoundary(network, trainData, testData) {
        const ctx = this.contexts.decision;
        if (!ctx) return;

        const canvas = this.canvases.decision;
        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);

        ctx.fillStyle = this.colors.bg;
        ctx.fillRect(0, 0, width, height);

        const resolutions = { low: 20, medium: 35, high: 50 };
        const resolution = resolutions[this.resolution] || 20;

        const cellW = width / resolution;
        const cellH = height / resolution;

        // Draw decision boundary
        if (network) {
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const x = (i / resolution) * 2 - 1;
                    const y = (j / resolution) * 2 - 1;

                    const output = network.predict([x, -y])[0];
                    const alpha = 0.4;

                    if (output > 0.5) {
                        const intensity = (output - 0.5) * 2;
                        ctx.fillStyle = `rgba(99, 102, 241, ${intensity * alpha})`;
                    } else {
                        const intensity = (0.5 - output) * 2;
                        ctx.fillStyle = `rgba(239, 68, 68, ${intensity * alpha})`;
                    }

                    ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
                }
            }
        }

        // Draw data points
        const drawPoints = (data, isTest = false) => {
            if (!data) return;

            for (let i = 0; i < data.data.length; i++) {
                const [x, y] = data.data[i];
                const label = data.labels[i][0];

                const px = (x + 1) / 2 * width;
                const py = (-y + 1) / 2 * height;

                ctx.beginPath();
                ctx.arc(px, py, isTest ? 5 : 6, 0, Math.PI * 2);

                if (isTest) {
                    ctx.strokeStyle = label > 0.5 ? this.colors.class1 : this.colors.class0;
                    ctx.lineWidth = 2;
                    ctx.stroke();
                } else {
                    ctx.fillStyle = label > 0.5 ? this.colors.class1 : this.colors.class0;
                    ctx.fill();
                    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        };

        drawPoints(trainData, false);
        drawPoints(testData, true);

        // Axes
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);

        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(width / 2, 0);
        ctx.lineTo(width / 2, height);
        ctx.stroke();

        ctx.setLineDash([]);
    }

    // Draw metrics chart
    drawMetrics(lossHistory, accuracyHistory, testLossHistory = [], testAccHistory = []) {
        const ctx = this.contexts.metrics;
        if (!ctx) return;

        const canvas = this.canvases.metrics;
        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);

        ctx.fillStyle = this.colors.bg;
        ctx.fillRect(0, 0, width, height);

        if (lossHistory.length < 2) {
            ctx.fillStyle = this.colors.text;
            ctx.font = '12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Training metrics will appear here...', width / 2, height / 2);
            return;
        }

        const padding = { left: 45, right: 45, top: 15, bottom: 35 };
        const chartW = width - padding.left - padding.right;
        const chartH = height - padding.top - padding.bottom;

        // Find scales
        const allLoss = [...lossHistory, ...(testLossHistory || [])];
        const maxLoss = allLoss.length > 0 ? Math.max(...allLoss) * 1.1 : 1;

        // Draw grid
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
        ctx.lineWidth = 1;

        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (i / 4) * chartH;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
        }

        // Draw train loss (solid purple)
        this.drawLine(ctx, lossHistory, padding, chartW, chartH, maxLoss, this.colors.loss, 2);

        // Draw test loss (dashed purple)
        if (testLossHistory && testLossHistory.length > 1) {
            this.drawLine(ctx, testLossHistory, padding, chartW, chartH, maxLoss, '#a78bfa', 1.5, true);
        }

        // Draw train accuracy (solid green)
        if (accuracyHistory.length > 1) {
            this.drawLine(ctx, accuracyHistory, padding, chartW, chartH, 1, this.colors.accuracy, 2);
        }

        // Draw test accuracy (dashed green)
        if (testAccHistory && testAccHistory.length > 1) {
            this.drawLine(ctx, testAccHistory, padding, chartW, chartH, 1, '#34d399', 1.5, true);
        }

        // Labels
        ctx.fillStyle = this.colors.text;
        ctx.font = '9px JetBrains Mono, monospace';

        // Left axis (loss)
        ctx.textAlign = 'right';
        ctx.fillStyle = this.colors.loss;
        ctx.fillText(maxLoss.toFixed(2), padding.left - 5, padding.top + 4);
        ctx.fillText('0', padding.left - 5, padding.top + chartH);

        // Right axis (accuracy)
        ctx.textAlign = 'left';
        ctx.fillStyle = this.colors.accuracy;
        ctx.fillText('100%', width - padding.right + 5, padding.top + 4);
        ctx.fillText('0%', width - padding.right + 5, padding.top + chartH);

        // Legend - two rows
        ctx.font = '9px Inter, sans-serif';
        const legendY1 = height - 22;
        const legendY2 = height - 8;

        // Train Loss (solid)
        ctx.fillStyle = this.colors.loss;
        ctx.fillRect(padding.left, legendY1 - 2, 12, 3);
        ctx.fillText('Train Loss', padding.left + 16, legendY1);

        // Test Loss (dashed)
        ctx.fillStyle = '#a78bfa';
        ctx.setLineDash([3, 2]);
        ctx.strokeStyle = '#a78bfa';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding.left + 90, legendY1 - 1);
        ctx.lineTo(padding.left + 102, legendY1 - 1);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillText('Test Loss', padding.left + 106, legendY1);

        // Train Acc (solid)
        ctx.fillStyle = this.colors.accuracy;
        ctx.fillRect(padding.left, legendY2 - 2, 12, 3);
        ctx.fillText('Train Acc', padding.left + 16, legendY2);

        // Test Acc (dashed)
        ctx.fillStyle = '#34d399';
        ctx.setLineDash([3, 2]);
        ctx.strokeStyle = '#34d399';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding.left + 90, legendY2 - 1);
        ctx.lineTo(padding.left + 102, legendY2 - 1);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillText('Test Acc', padding.left + 106, legendY2);

        // Update min/max displays
        const minLossEl = document.getElementById('min-loss');
        const maxAccEl = document.getElementById('max-accuracy');
        if (minLossEl && lossHistory.length > 0) {
            minLossEl.textContent = Math.min(...lossHistory).toFixed(4);
        }
        if (maxAccEl && accuracyHistory.length > 0) {
            maxAccEl.textContent = (Math.max(...accuracyHistory) * 100).toFixed(1) + '%';
        }
    }

    drawLine(ctx, data, padding, chartW, chartH, maxVal, color, lineWidth, dashed = false) {
        if (data.length < 2) return;

        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        if (dashed) ctx.setLineDash([4, 4]);

        ctx.beginPath();
        for (let i = 0; i < data.length; i++) {
            const x = padding.left + (i / (data.length - 1)) * chartW;
            const y = padding.top + (1 - data[i] / maxVal) * chartH;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw weight heatmap
    drawWeightHeatmap(network, selectedLayer = 'all') {
        const ctx = this.contexts.weights;
        if (!ctx || !network) return;

        const canvas = this.canvases.weights;
        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);

        ctx.fillStyle = this.colors.bg;
        ctx.fillRect(0, 0, width, height);

        const weights = network.getWeightInfo();
        if (!weights.length) return;

        // Determine which layers to show
        let layersToShow = weights;
        if (selectedLayer !== 'all') {
            const idx = parseInt(selectedLayer);
            if (!isNaN(idx) && weights[idx]) {
                layersToShow = [weights[idx]];
            }
        }

        const padding = 20;
        const availableWidth = width - padding * 2;
        const layerWidth = availableWidth / layersToShow.length;

        layersToShow.forEach((layer, li) => {
            const startX = padding + li * layerWidth;
            const layerPadding = 10;

            const cellW = (layerWidth - layerPadding * 2) / layer.cols;
            const cellH = Math.min(15, (height - 50) / layer.rows);

            const offsetY = (height - layer.rows * cellH) / 2;

            for (let r = 0; r < layer.rows; r++) {
                for (let c = 0; c < layer.cols; c++) {
                    const w = layer.data[r][c];
                    const color = Utils.getHeatmapColor(w);

                    ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
                    ctx.fillRect(
                        startX + layerPadding + c * cellW,
                        offsetY + r * cellH,
                        cellW - 1,
                        cellH - 1
                    );
                }
            }

            // Label
            ctx.fillStyle = this.colors.text;
            ctx.font = '9px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`L${layer.layer + 1}`, startX + layerWidth / 2, height - 8);
        });
    }

    setResolution(res) {
        this.resolution = res;
    }

    updateAll(network, trainData, testData, layerSizes, skipExpensive = false, selectedWeightLayer = 'all') {
        if (network && trainData) {
            network.predict(trainData.data[0]);
        }

        this.drawNetwork(network, layerSizes);

        // Decision boundary is expensive - skip on some frames during training
        if (!skipExpensive) {
            this.drawDecisionBoundary(network, trainData, testData);
            this.drawWeightHeatmap(network, selectedWeightLayer);
        }

        // Metrics chart is cheap, always update
        this.drawMetrics(
            network?.lossHistory || [],
            network?.accuracyHistory || [],
            network?.testLossHistory || [],
            network?.testAccuracyHistory || []
        );
    }
}

// Export
window.Visualizer = Visualizer;
