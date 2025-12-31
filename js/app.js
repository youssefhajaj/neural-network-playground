/**
 * Neural Network Playground - Main Application
 * Orchestrates all components and handles user interactions
 */

class App {
    constructor() {
        this.config = {
            hiddenLayers: [4, 4],
            learningRate: 0.03,
            batchSize: 32,
            activation: 'tanh',
            optimizer: 'adam',
            l2: 0,
            dropout: 0,
            datasetType: 'spiral',
            noise: 0.15,
            trainRatio: 0.8,
            speed: 3,
            maxEpochs: 150
        };

        this.network = null;
        this.visualizer = null;
        this.trainData = null;
        this.testData = null;

        this.isTraining = false;
        this.animationId = null;
        this.frameCount = 0;
        this.initialStats = null;
        this.trainingStartTime = null;
        this.trainingLog = [];
        this.selectedWeightLayer = 'all';

        this.presets = {
            simple: [4],
            medium: [4, 4],
            deep: [4, 4, 4, 4],
            wide: [8, 8]
        };

        this.init();
    }

    init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }

    setup() {
        // Setup event listeners first
        this.setupEventListeners();
        this.updateLayerBuilder();

        // Generate data and network
        this.generateDataset();
        this.createNetwork();

        // Delay visualization setup to ensure layout is ready
        setTimeout(() => {
            this.visualizer = new Visualizer();
            this.visualizer.init();

            // Force another canvas setup after a brief delay
            setTimeout(() => {
                this.visualizer.setupCanvases();
                this.updateStats(); // Show initial untrained stats
                this.render(true);
            }, 100);

            this.initMathRendering();
            this.loadTheme();
            console.log('Neural Network Playground initialized');
        }, 50);
    }

    setupEventListeners() {
        // Training controls
        document.getElementById('btn-train')?.addEventListener('click', () => this.startTraining());
        document.getElementById('btn-pause')?.addEventListener('click', () => this.pauseTraining());
        document.getElementById('btn-reset')?.addEventListener('click', () => this.reset());
        document.getElementById('btn-step')?.addEventListener('click', () => this.trainStep());

        // Dataset selection
        document.querySelectorAll('.dataset-card').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.dataset-card').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                this.config.datasetType = e.currentTarget.dataset.dataset;
                this.generateDataset();
                this.reset();
            });
        });

        // Regenerate data
        document.getElementById('btn-regenerate')?.addEventListener('click', () => {
            this.generateDataset();
            this.render();
        });

        // Noise slider
        this.setupSlider('noise-slider', 'noise-display', (val) => {
            this.config.noise = parseFloat(val);
            return val;
        }, () => {
            this.generateDataset();
            this.render();
        });

        // Train ratio slider
        this.setupSlider('ratio-slider', 'ratio-display', (val) => {
            this.config.trainRatio = parseInt(val) / 100;
            return val + '%';
        }, () => {
            this.generateDataset();
            this.render();
        });

        // Learning rate slider (log scale)
        this.setupSlider('lr-slider', 'lr-display', (val) => {
            this.config.learningRate = Math.pow(10, parseFloat(val));
            if (this.network) this.network.setLearningRate(this.config.learningRate);
            return this.config.learningRate.toFixed(3);
        });

        // Batch size slider (powers of 2)
        this.setupSlider('batch-slider', 'batch-display', (val) => {
            this.config.batchSize = Math.pow(2, parseInt(val));
            return this.config.batchSize;
        });

        // L2 slider
        this.setupSlider('l2-slider', 'l2-display', (val) => {
            this.config.l2 = parseFloat(val);
            if (this.network) this.network.setRegularization(this.config.l2, this.config.dropout);
            return parseFloat(val).toFixed(3);
        });

        // Dropout slider
        this.setupSlider('dropout-slider', 'dropout-display', (val) => {
            this.config.dropout = parseFloat(val);
            if (this.network) this.network.setRegularization(this.config.l2, this.config.dropout);
            return Math.round(parseFloat(val) * 100) + '%';
        });

        // Activation select
        document.getElementById('activation-select')?.addEventListener('change', (e) => {
            this.config.activation = e.target.value;
            this.createNetwork();
            this.render();
        });

        // Optimizer select
        document.getElementById('optimizer-select')?.addEventListener('change', (e) => {
            this.config.optimizer = e.target.value;
            if (this.network) this.network.setOptimizer(this.config.optimizer);
        });

        // Layer controls
        document.getElementById('btn-add-layer')?.addEventListener('click', () => this.addLayer());
        document.getElementById('btn-remove-layer')?.addEventListener('click', () => this.removeLayer());

        // Presets
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                const preset = e.currentTarget.dataset.preset;
                if (this.presets[preset]) {
                    this.config.hiddenLayers = [...this.presets[preset]];
                    this.updateLayerBuilder();
                    this.createNetwork();
                    this.render();
                }
            });
        });

        // Speed buttons
        document.querySelectorAll('.speed-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                this.config.speed = parseInt(e.currentTarget.dataset.speed);
            });
        });

        // Max epochs slider
        this.setupSlider('max-epochs-slider', 'max-epochs-display', (val) => {
            this.config.maxEpochs = parseInt(val);
            return val;
        });

        // Training log buttons
        document.getElementById('btn-clear-log')?.addEventListener('click', () => this.clearLog());
        document.getElementById('btn-copy-log')?.addEventListener('click', () => this.copyLog());
        document.getElementById('btn-export-log')?.addEventListener('click', () => this.exportLog());
        document.getElementById('btn-export-python')?.addEventListener('click', () => this.exportToPython());

        // Weight layer selector
        document.getElementById('weight-layer-select')?.addEventListener('change', (e) => {
            this.selectedWeightLayer = e.target.value;
            this.render(true);
        });

        // Resolution buttons
        document.querySelectorAll('[data-resolution]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-resolution]').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                this.visualizer.setResolution(e.currentTarget.dataset.resolution);
                this.render();
            });
        });

        // Theme toggle
        document.getElementById('btn-theme')?.addEventListener('click', () => this.toggleTheme());

        // Help modal
        document.getElementById('btn-help')?.addEventListener('click', () => {
            document.getElementById('help-modal')?.classList.add('open');
        });

        document.querySelectorAll('.modal-close, .modal-backdrop').forEach(el => {
            el.addEventListener('click', () => {
                document.querySelectorAll('.modal').forEach(m => m.classList.remove('open'));
            });
        });

        // Window resize
        window.addEventListener('resize', Utils.debounce(() => {
            this.visualizer.setupCanvases();
            this.render();
        }, 200));
    }

    setupSlider(sliderId, displayId, formatFn, onChangeFn = null) {
        const slider = document.getElementById(sliderId);
        const display = document.getElementById(displayId);

        if (slider && display) {
            // Sync initial value from HTML to config
            formatFn(slider.value);

            slider.addEventListener('input', (e) => {
                display.textContent = formatFn(e.target.value);
            });

            if (onChangeFn) {
                slider.addEventListener('change', onChangeFn);
            }
        }
    }

    updateLayerBuilder() {
        const container = document.getElementById('layer-builder');
        if (!container) return;

        container.innerHTML = '';

        // Input layer
        container.innerHTML += `
            <div class="layer-row input">
                <span class="layer-label">Input Layer</span>
                <span class="layer-input" style="background:transparent;border:none;">2</span>
            </div>
        `;

        // Hidden layers
        this.config.hiddenLayers.forEach((neurons, i) => {
            container.innerHTML += `
                <div class="layer-row">
                    <span class="layer-label">Hidden ${i + 1}</span>
                    <input type="number" class="layer-input" min="1" max="16" value="${neurons}" data-layer="${i}">
                </div>
            `;
        });

        // Output layer
        container.innerHTML += `
            <div class="layer-row output">
                <span class="layer-label">Output Layer</span>
                <span class="layer-input" style="background:transparent;border:none;">1</span>
            </div>
        `;

        // Add event listeners
        container.querySelectorAll('input[data-layer]').forEach(input => {
            input.addEventListener('change', (e) => {
                const layer = parseInt(e.target.dataset.layer);
                const value = Utils.clamp(parseInt(e.target.value) || 4, 1, 16);
                e.target.value = value;
                this.config.hiddenLayers[layer] = value;
                this.createNetwork();
                this.render();
            });
        });

        this.updateParamsCount();
    }

    addLayer() {
        if (this.config.hiddenLayers.length < 6) {
            this.config.hiddenLayers.push(4);
            this.updateLayerBuilder();
            this.createNetwork();
            this.render();
        }
    }

    removeLayer() {
        if (this.config.hiddenLayers.length > 1) {
            this.config.hiddenLayers.pop();
            this.updateLayerBuilder();
            this.createNetwork();
            this.render();
        }
    }

    getLayerSizes() {
        return [2, ...this.config.hiddenLayers, 1];
    }

    createNetwork() {
        // Validate config before creating network
        const validation = this.validateConfig();
        if (!validation.valid) {
            this.showError(validation.message);
            // Apply fixes automatically
            if (validation.fixes) {
                Object.assign(this.config, validation.fixes);
            }
        }

        try {
            this.network = new NeuralNetwork({
                layers: this.getLayerSizes(),
                activation: this.config.activation,
                optimizer: this.config.optimizer,
                learningRate: this.config.learningRate,
                l2: this.config.l2,
                dropout: this.config.dropout
            });

            this.updateParamsCount();
            this.updateWeightLayerSelect();
        } catch (error) {
            this.showError('Failed to create network: ' + error.message);
            console.error('Network creation error:', error);
        }
    }

    validateConfig() {
        const issues = [];
        const fixes = {};

        // Check learning rate
        if (this.config.learningRate <= 0) {
            issues.push('Learning rate must be positive');
            fixes.learningRate = 0.01;
        } else if (this.config.learningRate > 1) {
            issues.push('Learning rate too high (>1), may cause instability');
        }

        // Check hidden layers
        if (this.config.hiddenLayers.length === 0) {
            issues.push('Network needs at least one hidden layer');
            fixes.hiddenLayers = [4];
        }

        // Check for very small network with complex dataset
        const totalNeurons = this.config.hiddenLayers.reduce((a, b) => a + b, 0);
        if (totalNeurons < 4 && ['spiral', 'rings'].includes(this.config.datasetType)) {
            issues.push('Network may be too small for this dataset');
        }

        // Check dropout
        if (this.config.dropout >= 0.9) {
            issues.push('Dropout rate too high (>=90%), network cannot learn');
            fixes.dropout = 0.5;
        }

        // Check batch size vs actual data size
        const trainSize = this.trainData ? this.trainData.data.length : 300 * this.config.trainRatio;
        if (this.config.batchSize > trainSize * 0.8) {
            issues.push('Batch size very large relative to training data, may slow convergence');
        }

        // Check L2 regularization
        if (this.config.l2 > 0.5) {
            issues.push('L2 regularization too high, may prevent learning');
            fixes.l2 = 0.01;
        }

        return {
            valid: issues.length === 0,
            message: issues.join('. '),
            fixes: Object.keys(fixes).length > 0 ? fixes : null
        };
    }

    showError(message) {
        // Create toast notification
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = 'toast toast-warning';
        toast.innerHTML = `
            <span class="toast-icon">⚠️</span>
            <span class="toast-message">${message}</span>
        `;
        container.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            toast.classList.add('toast-fade');
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    updateParamsCount() {
        const badge = document.getElementById('params-badge');
        if (badge && this.network) {
            const count = this.network.getParameterCount();
            badge.textContent = `${count} params`;
        }
    }

    updateWeightLayerSelect() {
        const select = document.getElementById('weight-layer-select');
        if (!select || !this.network) return;

        select.innerHTML = '<option value="all">All Layers</option>';
        for (let i = 0; i < this.network.weights.length; i++) {
            select.innerHTML += `<option value="${i}">Layer ${i + 1}</option>`;
        }
    }

    generateDataset() {
        const generator = Datasets[this.config.datasetType];
        if (!generator) return;

        const fullData = generator(300, this.config.noise);
        const split = Datasets.split(fullData, this.config.trainRatio);

        this.trainData = split.train;
        this.testData = split.test;
    }

    startTraining() {
        if (this.isTraining) return;

        // Ensure network and data exist
        if (!this.network) {
            this.createNetwork();
        }
        if (!this.trainData) {
            this.generateDataset();
        }

        // Check if we've already reached max epochs
        if (this.network.epoch >= this.config.maxEpochs) {
            this.showToast(`Already at max epochs (${this.config.maxEpochs}). Click Reset to start over.`, 'info');
            return;
        }

        this.isTraining = true;
        this.trainingStartTime = Date.now();
        this.updateTrainingStatus('training');

        const btnTrain = document.getElementById('btn-train');
        const btnPause = document.getElementById('btn-pause');
        if (btnTrain) btnTrain.disabled = true;
        if (btnPause) btnPause.disabled = false;

        this.trainingLoop();
    }

    pauseTraining() {
        this.isTraining = false;
        this.updateTrainingStatus('paused');

        const btnTrain = document.getElementById('btn-train');
        const btnPause = document.getElementById('btn-pause');
        if (btnTrain) btnTrain.disabled = false;
        if (btnPause) btnPause.disabled = true;

        if (this.animationId) {
            clearTimeout(this.animationId);
            this.animationId = null;
        }
    }

    trainStep() {
        if (!this.network || !this.trainData) return;

        // Check if we've reached max epochs
        if (this.network.epoch >= this.config.maxEpochs) {
            this.showToast('Max epochs reached. Click Reset to start over.', 'info');
            return;
        }

        const loss = this.network.trainEpoch(this.trainData, this.config.batchSize);
        const trainAcc = this.network.calculateAccuracy(this.trainData.data, this.trainData.labels);

        // Record metrics (consistent with trainingLoop - every step for manual stepping)
        this.network.lossHistory.push(loss);
        this.network.accuracyHistory.push(trainAcc);

        let testLoss = null, testAcc = null;
        if (this.testData) {
            testLoss = this.network.calculateLoss(this.testData.data, this.testData.labels);
            testAcc = this.network.calculateAccuracy(this.testData.data, this.testData.labels);
            this.network.testLossHistory.push(testLoss);
            this.network.testAccuracyHistory.push(testAcc);
        }

        // Add to training log every 10 epochs (consistent with trainingLoop)
        if (this.network.epoch % 10 === 0) {
            this.addLogEntry(this.network.epoch, loss, trainAcc, testLoss, testAcc);
        }

        this.updateStats();
        this.render();
    }

    trainingLoop() {
        if (!this.isTraining) return;

        const batchesPerFrame = this.config.speed;

        for (let i = 0; i < batchesPerFrame; i++) {
            // Check if we've reached max epochs
            if (this.network.epoch >= this.config.maxEpochs) {
                this.pauseTraining();
                this.updateTrainingStatus('completed');
                return;
            }

            const loss = this.network.trainEpoch(this.trainData, this.config.batchSize);

            // Record metrics periodically
            if (this.network.epoch % 5 === 0) {
                const trainAcc = this.network.calculateAccuracy(this.trainData.data, this.trainData.labels);
                this.network.lossHistory.push(loss);
                this.network.accuracyHistory.push(trainAcc);

                let testLoss = null, testAcc = null;
                if (this.testData) {
                    testLoss = this.network.calculateLoss(this.testData.data, this.testData.labels);
                    testAcc = this.network.calculateAccuracy(this.testData.data, this.testData.labels);
                    this.network.testLossHistory.push(testLoss);
                    this.network.testAccuracyHistory.push(testAcc);
                }

                // Log to table every 10 epochs
                if (this.network.epoch % 10 === 0) {
                    this.addLogEntry(this.network.epoch, loss, trainAcc, testLoss, testAcc);
                }
            }
        }

        this.updateStats();
        this.render();

        // Use setTimeout instead of requestAnimationFrame for lower CPU usage
        // 33ms = ~30fps which is sufficient for visualization
        this.animationId = setTimeout(() => this.trainingLoop(), 33);
    }

    updateStats() {
        if (!this.network || !this.trainData) return;

        // Header stats
        const epochBadge = document.getElementById('epoch-badge');
        const lossValue = document.getElementById('loss-value');
        const accuracyValue = document.getElementById('accuracy-value');

        if (epochBadge) epochBadge.textContent = `Epoch ${this.network.epoch}`;

        // Calculate current values
        let currentLoss, currentAcc;
        if (this.network.lossHistory.length) {
            currentLoss = this.network.lossHistory[this.network.lossHistory.length - 1];
            currentAcc = this.network.accuracyHistory[this.network.accuracyHistory.length - 1];
        } else {
            currentLoss = this.network.calculateLoss(this.trainData.data, this.trainData.labels);
            currentAcc = this.network.calculateAccuracy(this.trainData.data, this.trainData.labels);
        }

        if (lossValue) lossValue.textContent = currentLoss.toFixed(4);
        if (accuracyValue) accuracyValue.textContent = (currentAcc * 100).toFixed(1) + '%';

        // Statistics Panel
        const statEpoch = document.getElementById('stat-epoch');
        const statParams = document.getElementById('stat-params');
        const statTrainLoss = document.getElementById('stat-train-loss');
        const statTrainAcc = document.getElementById('stat-train-acc');
        const statTestLoss = document.getElementById('stat-test-loss');
        const statTestAcc = document.getElementById('stat-test-acc');
        const statInitialLoss = document.getElementById('stat-initial-loss');
        const statInitialAcc = document.getElementById('stat-initial-acc');
        const statBestLoss = document.getElementById('stat-best-loss');
        const statBestAcc = document.getElementById('stat-best-acc');
        const statsStatus = document.getElementById('stats-status');

        if (statEpoch) statEpoch.textContent = this.network.epoch;
        if (statParams) statParams.textContent = this.network.getParameterCount();
        if (statTrainLoss) statTrainLoss.textContent = currentLoss.toFixed(4);
        if (statTrainAcc) statTrainAcc.textContent = (currentAcc * 100).toFixed(1) + '%';

        // Test data stats
        if (this.testData) {
            let testLoss, testAcc;
            if (this.network.testLossHistory.length) {
                testLoss = this.network.testLossHistory[this.network.testLossHistory.length - 1];
                testAcc = this.network.testAccuracyHistory[this.network.testAccuracyHistory.length - 1];
            } else {
                testLoss = this.network.calculateLoss(this.testData.data, this.testData.labels);
                testAcc = this.network.calculateAccuracy(this.testData.data, this.testData.labels);
            }
            if (statTestLoss) statTestLoss.textContent = testLoss.toFixed(4);
            if (statTestAcc) statTestAcc.textContent = (testAcc * 100).toFixed(1) + '%';
        }

        // Initial stats (store on first call)
        if (!this.initialStats && this.trainData) {
            this.initialStats = {
                loss: this.network.calculateLoss(this.trainData.data, this.trainData.labels),
                acc: this.network.calculateAccuracy(this.trainData.data, this.trainData.labels)
            };
        }
        if (this.initialStats) {
            if (statInitialLoss) statInitialLoss.textContent = this.initialStats.loss.toFixed(4);
            if (statInitialAcc) statInitialAcc.textContent = (this.initialStats.acc * 100).toFixed(1) + '%';
        }

        // Best stats
        if (this.network.lossHistory.length > 0) {
            const minLoss = Math.min(...this.network.lossHistory);
            const maxAcc = Math.max(...this.network.accuracyHistory);
            if (statBestLoss) statBestLoss.textContent = minLoss.toFixed(4);
            if (statBestAcc) statBestAcc.textContent = (maxAcc * 100).toFixed(1) + '%';
        }

        // Status badge
        if (statsStatus) {
            if (this.isTraining) {
                statsStatus.textContent = 'Training';
                statsStatus.style.background = '#10b981';
            } else if (this.network.epoch > 0) {
                statsStatus.textContent = 'Paused';
                statsStatus.style.background = '#f59e0b';
            } else {
                statsStatus.textContent = 'Ready';
                statsStatus.style.background = '';
            }
        }
    }

    updateTrainingStatus(status) {
        const statusEl = document.getElementById('training-status');
        if (!statusEl) return;

        statusEl.className = 'training-status ' + status;

        const textEl = statusEl.querySelector('.status-text');
        if (textEl) {
            const texts = {
                training: 'Training...',
                paused: 'Paused',
                ready: 'Ready',
                completed: 'Completed'
            };
            textEl.textContent = texts[status] || 'Ready';
        }

        // Also update stats panel badge
        const statsStatus = document.getElementById('stats-status');
        if (statsStatus && status === 'completed') {
            statsStatus.textContent = 'Done';
            statsStatus.style.background = '#6366f1';
        }
    }

    reset() {
        this.pauseTraining();

        // Reset initial stats so they get recalculated
        this.initialStats = null;
        this.trainingStartTime = null;

        if (this.network) {
            this.network.reset();
        } else {
            this.createNetwork();
        }

        this.updateTrainingStatus('ready');

        // Reset best stats display
        const statBestLoss = document.getElementById('stat-best-loss');
        const statBestAcc = document.getElementById('stat-best-acc');
        if (statBestLoss) statBestLoss.textContent = '—';
        if (statBestAcc) statBestAcc.textContent = '—';

        // Clear training log
        this.clearLog();

        // Update all stats with new initial values
        this.updateStats();
        this.render();
    }

    render(forceFullRender = false) {
        if (!this.visualizer) return;

        this.frameCount++;

        // Only do full render (decision boundary) every 3rd frame during training
        const skipExpensive = this.isTraining && !forceFullRender && (this.frameCount % 3 !== 0);

        this.visualizer.updateAll(
            this.network,
            this.trainData,
            this.testData,
            this.getLayerSizes(),
            skipExpensive,
            this.selectedWeightLayer
        );
    }

    initMathRendering() {
        if (typeof renderMathInElement === 'function') {
            renderMathInElement(document.body, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false }
                ],
                throwOnError: false
            });
        }
    }

    toggleTheme() {
        const body = document.body;
        const isLight = body.classList.toggle('light-theme');

        // Toggle icons
        const sunIcon = document.querySelector('#btn-theme .icon-sun');
        const moonIcon = document.querySelector('#btn-theme .icon-moon');

        if (sunIcon && moonIcon) {
            sunIcon.style.display = isLight ? 'none' : 'block';
            moonIcon.style.display = isLight ? 'block' : 'none';
        }

        // Update visualizer colors for light mode
        if (this.visualizer) {
            this.visualizer.colors.bg = isLight ? '#f8fafc' : '#030712';
            this.visualizer.colors.text = isLight ? '#64748b' : '#94a3b8';
            this.render(true);
        }

        // Save preference
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
    }

    loadTheme() {
        const saved = localStorage.getItem('theme');
        if (saved === 'light') {
            this.toggleTheme();
        }
    }

    addLogEntry(epoch, trainLoss, trainAcc, testLoss, testAcc) {
        const logBody = document.getElementById('log-body');
        if (!logBody) return;

        // Clear placeholder on first entry
        const placeholder = logBody.querySelector('.log-empty');
        if (placeholder) placeholder.remove();

        const elapsed = this.trainingStartTime ?
            ((Date.now() - this.trainingStartTime) / 1000).toFixed(1) + 's' : '—';

        const entry = {
            epoch,
            trainLoss: trainLoss.toFixed(4),
            trainAcc: (trainAcc * 100).toFixed(1) + '%',
            testLoss: testLoss !== null ? testLoss.toFixed(4) : '—',
            testAcc: testAcc !== null ? (testAcc * 100).toFixed(1) + '%' : '—',
            time: elapsed
        };

        this.trainingLog.push(entry);

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${entry.epoch}</td>
            <td>${entry.trainLoss}</td>
            <td>${entry.trainAcc}</td>
            <td>${entry.testLoss}</td>
            <td>${entry.testAcc}</td>
            <td>${entry.time}</td>
        `;
        logBody.appendChild(row);

        // Auto-scroll to bottom
        const wrapper = logBody.closest('.log-table-wrapper');
        if (wrapper) {
            wrapper.scrollTop = wrapper.scrollHeight;
        }
    }

    clearLog() {
        this.trainingLog = [];
        const logBody = document.getElementById('log-body');
        if (logBody) {
            logBody.innerHTML = '<tr class="log-empty"><td colspan="6">Training log will appear here when you start training...</td></tr>';
        }
    }

    copyLog() {
        if (this.trainingLog.length === 0) {
            this.showToast('No training data to copy', 'info');
            return;
        }

        const headers = 'Epoch\tTrain Loss\tTrain Acc\tTest Loss\tTest Acc\tTime';
        const rows = this.trainingLog.map(e =>
            `${e.epoch}\t${e.trainLoss}\t${e.trainAcc}\t${e.testLoss}\t${e.testAcc}\t${e.time}`
        );

        const text = [headers, ...rows].join('\n');

        navigator.clipboard.writeText(text).then(() => {
            this.showToast('Copied to clipboard!', 'success');
        }).catch(() => {
            this.showToast('Failed to copy', 'error');
        });
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const icons = { success: '✓', error: '✗', info: 'ℹ', warning: '⚠️' };
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<span class="toast-icon">${icons[type] || ''}</span><span class="toast-message">${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-fade');
            setTimeout(() => toast.remove(), 300);
        }, 2000);
    }

    exportLog() {
        if (this.trainingLog.length === 0) {
            this.showToast('No training data to export.', 'info');
            return;
        }

        const headers = ['Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Time'];
        const rows = this.trainingLog.map(e =>
            [e.epoch, e.trainLoss, e.trainAcc, e.testLoss, e.testAcc, e.time].join(',')
        );

        const csv = [headers.join(','), ...rows].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `training_log_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();

        URL.revokeObjectURL(url);
    }

    exportToPython() {
        const layers = this.getLayerSizes();
        const activation = this.config.activation;
        const optimizer = this.config.optimizer;
        const lr = this.config.learningRate;
        const batchSize = this.config.batchSize;
        const l2 = this.config.l2;
        const dropout = this.config.dropout;
        const dataset = this.config.datasetType;
        const noise = this.config.noise;
        const maxEpochs = this.config.maxEpochs;

        const pythonCode = `"""
Neural Network from Scratch - Exported from NN Playground
No PyTorch, No TensorFlow - Pure NumPy Implementation
"""

import numpy as np
import matplotlib.pyplot as plt

# ============== Configuration ==============
LAYERS = ${JSON.stringify(layers)}  # Network architecture
ACTIVATION = "${activation}"
OPTIMIZER = "${optimizer}"
LEARNING_RATE = ${lr}
BATCH_SIZE = ${batchSize}
L2_REGULARIZATION = ${l2}
DROPOUT = ${dropout}
MAX_EPOCHS = ${maxEpochs}
DATASET = "${dataset}"
NOISE = ${noise}

# ============== Activation Functions ==============
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

ACTIVATIONS = {
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'leaky-relu': (leaky_relu, leaky_relu_derivative),
    'elu': (elu, elu_derivative),
    'swish': (swish, swish_derivative),
    'linear': (linear, linear_derivative)
}

# ============== Dataset Generators ==============
def generate_xor(n_points, noise=0.1):
    data, labels = [], []
    per_quadrant = n_points // 4
    quadrants = [(0.5, 0.5, 1), (-0.5, 0.5, 0), (-0.5, -0.5, 1), (0.5, -0.5, 0)]
    for cx, cy, label in quadrants:
        for _ in range(per_quadrant):
            x = cx + (np.random.random() - 0.5) * 0.8 + (np.random.random() - 0.5) * noise
            y = cy + (np.random.random() - 0.5) * 0.8 + (np.random.random() - 0.5) * noise
            data.append([x, y])
            labels.append([label])
    return np.array(data), np.array(labels)

def generate_circle(n_points, noise=0.1):
    data, labels = [], []
    radius = 0.5
    for _ in range(n_points):
        angle = np.random.random() * np.pi * 2
        is_inside = np.random.random() > 0.5
        r = np.random.random() * radius * 0.7 if is_inside else radius + 0.15 + np.random.random() * 0.35
        x = r * np.cos(angle) + (np.random.random() - 0.5) * noise
        y = r * np.sin(angle) + (np.random.random() - 0.5) * noise
        data.append([x, y])
        labels.append([1 if is_inside else 0])
    return np.array(data), np.array(labels)

def generate_spiral(n_points, noise=0.1):
    data, labels = [], []
    per_class = n_points // 2
    for class_idx in range(2):
        for i in range(per_class):
            t = i / per_class
            r = t * 0.8
            angle = t * np.pi * 2.5 + class_idx * np.pi
            x = r * np.cos(angle) + (np.random.random() - 0.5) * noise
            y = r * np.sin(angle) + (np.random.random() - 0.5) * noise
            data.append([x, y])
            labels.append([class_idx])
    return np.array(data), np.array(labels)

def generate_moons(n_points, noise=0.1):
    data, labels = [], []
    per_class = n_points // 2
    for i in range(per_class):
        angle = np.pi * i / per_class
        x = np.cos(angle) * 0.5 + (np.random.random() - 0.5) * noise
        y = np.sin(angle) * 0.5 + (np.random.random() - 0.5) * noise
        data.append([x, y])
        labels.append([0])
    for i in range(per_class):
        angle = np.pi + np.pi * i / per_class
        x = 0.25 + np.cos(angle) * 0.5 + (np.random.random() - 0.5) * noise
        y = -0.25 + np.sin(angle) * 0.5 + (np.random.random() - 0.5) * noise
        data.append([x, y])
        labels.append([1])
    return np.array(data), np.array(labels)

def generate_clusters(n_points, noise=0.1):
    data, labels = [], []
    per_class = n_points // 2
    centers = [(-0.4, -0.4), (0.4, 0.4)]
    spread = 0.2 + noise * 0.5
    for class_idx, (cx, cy) in enumerate(centers):
        for _ in range(per_class):
            u1, u2 = np.random.random(), np.random.random()
            z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            data.append([cx + z0 * spread, cy + z1 * spread])
            labels.append([class_idx])
    return np.array(data), np.array(labels)

def generate_rings(n_points, noise=0.1):
    data, labels = [], []
    per_ring = n_points // 3
    rings = [(0.2, 0), (0.5, 1), (0.8, 0)]
    for r, label in rings:
        for _ in range(per_ring):
            angle = np.random.random() * np.pi * 2
            r_noise = r + (np.random.random() - 0.5) * 0.1 * (1 + noise)
            x = r_noise * np.cos(angle) + (np.random.random() - 0.5) * noise * 0.5
            y = r_noise * np.sin(angle) + (np.random.random() - 0.5) * noise * 0.5
            data.append([x, y])
            labels.append([label])
    return np.array(data), np.array(labels)

DATASETS = {
    'xor': generate_xor,
    'circle': generate_circle,
    'spiral': generate_spiral,
    'moons': generate_moons,
    'clusters': generate_clusters,
    'rings': generate_rings
}

# ============== Neural Network ==============
class NeuralNetwork:
    def __init__(self, layers, activation='tanh', optimizer='adam', learning_rate=0.03, l2=0, dropout=0):
        self.layers = layers
        self.activation_name = activation
        self.activation, self.activation_deriv = ACTIVATIONS[activation]
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.l2 = l2
        self.dropout = dropout
        self.epoch = 0

        # Initialize weights with Xavier/He initialization
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i + 1]
            if activation in ['relu', 'leaky-relu', 'elu']:
                std = np.sqrt(2.0 / fan_in)  # He initialization
            else:
                std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier
            self.weights.append(np.random.randn(fan_in, fan_out) * std)
            self.biases.append(np.zeros((1, fan_out)))

        # Optimizer state (for Adam/RMSprop/Momentum)
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.t = 0

        # History
        self.loss_history = []
        self.accuracy_history = []

    def forward(self, X, training=False):
        self.activations = [X]
        self.z_values = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = self.activations[-1] @ W + b
            self.z_values.append(z)

            # Apply activation (sigmoid for output layer)
            if i == len(self.weights) - 1:
                a = sigmoid(z)
            else:
                a = self.activation(z)
                # Apply dropout during training
                if training and self.dropout > 0:
                    mask = np.random.binomial(1, 1 - self.dropout, a.shape) / (1 - self.dropout)
                    a = a * mask

            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        output = self.activations[-1]

        # Output layer gradient (binary cross-entropy derivative)
        delta = output - y

        gradients_w = []
        gradients_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients
            dW = self.activations[i].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Add L2 regularization
            if self.l2 > 0:
                dW += self.l2 * self.weights[i]

            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)

            # Propagate to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_deriv(self.z_values[i - 1])

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        self.t += 1

        for i in range(len(self.weights)):
            if self.optimizer == 'sgd':
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

            elif self.optimizer == 'momentum':
                beta = 0.9
                self.m_weights[i] = beta * self.m_weights[i] + (1 - beta) * gradients_w[i]
                self.m_biases[i] = beta * self.m_biases[i] + (1 - beta) * gradients_b[i]
                self.weights[i] -= self.learning_rate * self.m_weights[i]
                self.biases[i] -= self.learning_rate * self.m_biases[i]

            elif self.optimizer == 'rmsprop':
                beta = 0.999
                eps = 1e-8
                self.v_weights[i] = beta * self.v_weights[i] + (1 - beta) * gradients_w[i] ** 2
                self.v_biases[i] = beta * self.v_biases[i] + (1 - beta) * gradients_b[i] ** 2
                self.weights[i] -= self.learning_rate * gradients_w[i] / (np.sqrt(self.v_weights[i]) + eps)
                self.biases[i] -= self.learning_rate * gradients_b[i] / (np.sqrt(self.v_biases[i]) + eps)

            elif self.optimizer == 'adam':
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * gradients_w[i]
                self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * gradients_w[i] ** 2
                self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * gradients_b[i]
                self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * gradients_b[i] ** 2

                # Bias correction
                m_w_hat = self.m_weights[i] / (1 - beta1 ** self.t)
                v_w_hat = self.v_weights[i] / (1 - beta2 ** self.t)
                m_b_hat = self.m_biases[i] / (1 - beta1 ** self.t)
                v_b_hat = self.v_biases[i] / (1 - beta2 ** self.t)

                self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
                self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

    def train_epoch(self, X, y, batch_size=32):
        indices = np.random.permutation(len(X))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(X), batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch, y_batch = X[batch_idx], y[batch_idx]

            # Forward pass
            output = self.forward(X_batch, training=True)

            # Calculate loss
            eps = 1e-15
            loss = -np.mean(y_batch * np.log(output + eps) + (1 - y_batch) * np.log(1 - output + eps))
            total_loss += loss
            n_batches += 1

            # Backward pass
            gradients_w, gradients_b = self.backward(X_batch, y_batch)

            # Update weights
            self.update_weights(gradients_w, gradients_b)

        self.epoch += 1
        return total_loss / n_batches

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def calculate_loss(self, X, y):
        output = self.forward(X)
        eps = 1e-15
        return -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))

# ============== Visualization ==============
def plot_decision_boundary(nn, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(label='Prediction')
    plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='blue', edgecolors='white', label='Class 0')
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='red', edgecolors='white', label='Class 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_training_history(loss_history, accuracy_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot([a * 100 for a in accuracy_history], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

# ============== Main Training ==============
if __name__ == "__main__":
    # Generate dataset
    print(f"Generating {DATASET} dataset...")
    generator = DATASETS[DATASET]
    X, y = generator(300, NOISE)

    # Train/test split
    split_idx = int(len(X) * 0.8)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Create network
    print(f"\\nNetwork Architecture: {LAYERS}")
    print(f"Activation: {ACTIVATION}, Optimizer: {OPTIMIZER}")
    print(f"Learning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")

    nn = NeuralNetwork(
        layers=LAYERS,
        activation=ACTIVATION,
        optimizer=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        l2=L2_REGULARIZATION,
        dropout=DROPOUT
    )

    # Calculate initial stats
    initial_loss = nn.calculate_loss(X_train, y_train)
    initial_acc = nn.accuracy(X_train, y_train)
    print(f"\\nInitial - Loss: {initial_loss:.4f}, Accuracy: {initial_acc * 100:.1f}%")

    # Training loop
    print(f"\\nTraining for {MAX_EPOCHS} epochs...")
    print("-" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Test Loss':>10} | {'Test Acc':>10}")
    print("-" * 60)

    loss_history = []
    accuracy_history = []

    for epoch in range(MAX_EPOCHS):
        loss = nn.train_epoch(X_train, y_train, BATCH_SIZE)
        train_acc = nn.accuracy(X_train, y_train)
        test_loss = nn.calculate_loss(X_test, y_test)
        test_acc = nn.accuracy(X_test, y_test)

        loss_history.append(loss)
        accuracy_history.append(train_acc)

        if (epoch + 1) % 10 == 0:
            print(f"{epoch + 1:>6} | {loss:>10.4f} | {train_acc * 100:>9.1f}% | {test_loss:>10.4f} | {test_acc * 100:>9.1f}%")

    print("-" * 60)

    # Final results
    final_train_acc = nn.accuracy(X_train, y_train)
    final_test_acc = nn.accuracy(X_test, y_test)
    print(f"\\nFinal Results:")
    print(f"  Train Accuracy: {final_train_acc * 100:.1f}%")
    print(f"  Test Accuracy:  {final_test_acc * 100:.1f}%")

    # Visualize
    plot_decision_boundary(nn, X, y, f"Decision Boundary - {DATASET.upper()} Dataset")
    plot_training_history(loss_history, accuracy_history)
    plt.show()
`;

        const blob = new Blob([pythonCode], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `neural_network_${this.config.datasetType}_${new Date().toISOString().slice(0, 10)}.py`;
        a.click();

        URL.revokeObjectURL(url);
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
