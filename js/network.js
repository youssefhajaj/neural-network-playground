/**
 * Neural Network Implementation
 * Complete implementation with forward/backward propagation,
 * multiple optimizers, and regularization
 */

class NeuralNetwork {
    constructor(config = {}) {
        this.layerSizes = config.layers || [2, 4, 4, 1];
        this.activationName = config.activation || 'tanh';
        this.activation = Activations[this.activationName] || Activations.tanh;

        // Regularization
        this.l2Lambda = config.l2 || 0;
        this.dropoutRate = config.dropout || 0;

        // Initialize weights and biases
        this.weights = [];
        this.biases = [];
        this._initializeWeights();

        // Optimizer
        this.optimizerType = config.optimizer || 'adam';
        this.learningRate = config.learningRate || 0.03;
        this.optimizer = Optimizers.create(this.optimizerType, this.learningRate);

        // Training state
        this.epoch = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
        this.testLossHistory = [];
        this.testAccuracyHistory = [];

        // Cache for forward pass
        this.layerOutputs = [];
        this.layerInputs = [];
        this.dropoutMasks = [];

        // Gradient cache
        this.weightGrads = [];
        this.biasGrads = [];
    }

    _initializeWeights() {
        const numLayers = this.layerSizes.length;

        for (let i = 0; i < numLayers - 1; i++) {
            const fanIn = this.layerSizes[i];
            const fanOut = this.layerSizes[i + 1];

            const w = new Matrix(fanOut, fanIn);
            const b = new Matrix(fanOut, 1);

            // Use He initialization for ReLU variants, Xavier for others
            if (['relu', 'leaky-relu', 'elu'].includes(this.activationName)) {
                w.heInit(fanIn);
            } else {
                w.xavierInit(fanIn, fanOut);
            }

            b.zeros();

            this.weights.push(w);
            this.biases.push(b);
        }
    }

    /**
     * Get total parameter count
     */
    getParameterCount() {
        let count = 0;
        for (let i = 0; i < this.weights.length; i++) {
            count += this.weights[i].rows * this.weights[i].cols;
            count += this.biases[i].rows;
        }
        return count;
    }

    /**
     * Forward propagation
     */
    forward(input, training = false) {
        this.layerOutputs = [];
        this.layerInputs = [];
        this.dropoutMasks = [];

        let current = Matrix.fromArray(input);
        this.layerOutputs.push(current.clone());

        for (let i = 0; i < this.weights.length; i++) {
            // z = W * a + b
            const z = Matrix.add(
                Matrix.multiply(this.weights[i], current),
                this.biases[i]
            );

            // Clip z values to prevent extreme activations
            z.clip(-50, 50);
            this.layerInputs.push(z.clone());

            // a = activation(z)
            current = z.map(val => {
                const result = this.activation.fn(val);
                // Ensure valid output
                return isFinite(result) ? result : 0;
            });

            // Apply dropout (except for output layer)
            if (training && this.dropoutRate > 0 && i < this.weights.length - 1) {
                const mask = current.map(() => Math.random() > this.dropoutRate ? 1 : 0);
                this.dropoutMasks.push(mask);
                current = Matrix.hadamard(current, mask);
                current.scale(1 / (1 - this.dropoutRate));
            } else {
                this.dropoutMasks.push(null);
            }

            this.layerOutputs.push(current.clone());
        }

        return current.toArray();
    }

    /**
     * Backward propagation
     */
    backward(target) {
        const m = 1;
        const numLayers = this.weights.length;

        this.weightGrads = [];
        this.biasGrads = [];

        // Output layer error
        const output = this.layerOutputs[this.layerOutputs.length - 1];
        const targetMatrix = Matrix.fromArray(target);

        // dL/da for MSE loss: 2(a - y)
        let delta = Matrix.subtract(output, targetMatrix);
        delta.scale(2);

        // Apply activation derivative
        const zL = this.layerInputs[numLayers - 1];
        if (this.activation.needsInput) {
            delta = delta.map((val, i, j) =>
                val * this.activation.derivative(output.data[i][j], zL.data[i][j])
            );
        } else {
            delta = delta.map((val, i, j) =>
                val * this.activation.derivative(output.data[i][j])
            );
        }

        // Backpropagate through layers
        for (let l = numLayers - 1; l >= 0; l--) {
            const prevOutput = this.layerOutputs[l];

            // Weight gradient: delta * prevOutput^T
            const wGrad = Matrix.multiply(delta, Matrix.transpose(prevOutput));

            // Add L2 regularization gradient
            if (this.l2Lambda > 0) {
                const regGrad = Matrix.scale(this.weights[l], this.l2Lambda);
                wGrad.add(regGrad);
            }

            this.weightGrads.unshift(wGrad);
            this.biasGrads.unshift(delta.clone());

            if (l > 0) {
                // Propagate delta to previous layer
                delta = Matrix.multiply(Matrix.transpose(this.weights[l]), delta);

                // Apply dropout mask
                if (this.dropoutMasks[l - 1]) {
                    delta = Matrix.hadamard(delta, this.dropoutMasks[l - 1]);
                    delta.scale(1 / (1 - this.dropoutRate));
                }

                // Apply activation derivative
                const z = this.layerInputs[l - 1];
                const a = this.layerOutputs[l];

                if (this.activation.needsInput) {
                    delta = delta.map((val, i, j) =>
                        val * this.activation.derivative(a.data[i][j], z.data[i][j])
                    );
                } else {
                    delta = delta.map((val, i, j) =>
                        val * this.activation.derivative(a.data[i][j])
                    );
                }
            }
        }
    }

    /**
     * Train on a single batch
     */
    trainBatch(inputs, targets) {
        const batchSize = inputs.length;

        // Accumulate gradients
        const accWeightGrads = this.weights.map(w => new Matrix(w.rows, w.cols));
        const accBiasGrads = this.biases.map(b => new Matrix(b.rows, b.cols));

        let totalLoss = 0;

        for (let i = 0; i < batchSize; i++) {
            // Forward pass
            const output = this.forward(inputs[i], true);

            // Calculate loss
            const loss = this._calculateLoss(output, targets[i]);
            totalLoss += loss;

            // Backward pass
            this.backward(targets[i]);

            // Accumulate gradients
            for (let l = 0; l < this.weights.length; l++) {
                accWeightGrads[l].add(this.weightGrads[l]);
                accBiasGrads[l].add(this.biasGrads[l]);
            }
        }

        // Average gradients
        for (let l = 0; l < this.weights.length; l++) {
            accWeightGrads[l].scale(1 / batchSize);
            accBiasGrads[l].scale(1 / batchSize);
        }

        // Gradient clipping to prevent exploding gradients
        const maxGradNorm = 5.0;
        for (let l = 0; l < this.weights.length; l++) {
            accWeightGrads[l].clip(-maxGradNorm, maxGradNorm);
            accBiasGrads[l].clip(-maxGradNorm, maxGradNorm);
        }

        // Update weights using optimizer
        this.optimizer.update(this.weights, this.biases, accWeightGrads, accBiasGrads);

        return totalLoss / batchSize;
    }

    /**
     * Calculate loss for a single sample
     */
    _calculateLoss(output, target) {
        let loss = 0;
        for (let i = 0; i < target.length; i++) {
            const diff = output[i] - target[i];
            // Clip large differences to prevent NaN
            const clippedDiff = Math.max(-10, Math.min(10, diff));
            loss += clippedDiff * clippedDiff;
        }
        loss /= target.length;

        // Add L2 regularization term
        if (this.l2Lambda > 0) {
            let l2Loss = 0;
            for (const w of this.weights) {
                l2Loss += w.sumSquared();
            }
            loss += 0.5 * this.l2Lambda * l2Loss;
        }

        // Ensure loss is valid
        if (!isFinite(loss)) {
            console.warn('Loss became NaN/Infinity, resetting to large value');
            return 10;
        }

        return loss;
    }

    /**
     * Predict output for input
     */
    predict(input) {
        return this.forward(input, false);
    }

    /**
     * Calculate accuracy on dataset
     */
    calculateAccuracy(inputs, targets, threshold = 0.5) {
        let correct = 0;
        for (let i = 0; i < inputs.length; i++) {
            const output = this.predict(inputs[i]);
            const predicted = output[0] > threshold ? 1 : 0;
            const actual = targets[i][0] > threshold ? 1 : 0;
            if (predicted === actual) correct++;
        }
        return correct / inputs.length;
    }

    /**
     * Calculate loss on dataset
     */
    calculateLoss(inputs, targets) {
        let totalLoss = 0;
        for (let i = 0; i < inputs.length; i++) {
            const output = this.predict(inputs[i]);
            totalLoss += this._calculateLoss(output, targets[i]);
        }
        return totalLoss / inputs.length;
    }

    /**
     * Train for one epoch
     */
    trainEpoch(trainData, batchSize = 32) {
        const n = trainData.data.length;
        const shuffled = this._shuffleData(trainData);

        let totalLoss = 0;
        let numBatches = 0;

        for (let i = 0; i < n; i += batchSize) {
            const end = Math.min(i + batchSize, n);
            const batchInputs = shuffled.data.slice(i, end);
            const batchTargets = shuffled.labels.slice(i, end);

            const loss = this.trainBatch(batchInputs, batchTargets);
            totalLoss += loss;
            numBatches++;
        }

        this.epoch++;
        return totalLoss / numBatches;
    }

    /**
     * Shuffle data
     */
    _shuffleData(data) {
        const n = data.data.length;
        const indices = Array.from({ length: n }, (_, i) => i);

        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        return {
            data: indices.map(i => data.data[i]),
            labels: indices.map(i => data.labels[i])
        };
    }

    /**
     * Reset network weights
     */
    reset() {
        this.weights = [];
        this.biases = [];
        this._initializeWeights();

        this.optimizer = Optimizers.create(this.optimizerType, this.learningRate);

        this.epoch = 0;
        this.lossHistory = [];
        this.accuracyHistory = [];
        this.testLossHistory = [];
        this.testAccuracyHistory = [];

        this.layerOutputs = [];
        this.layerInputs = [];
    }

    /**
     * Set learning rate
     */
    setLearningRate(lr) {
        this.learningRate = lr;
        this.optimizer.setLearningRate(lr);
    }

    /**
     * Set optimizer
     */
    setOptimizer(type) {
        this.optimizerType = type;
        this.optimizer = Optimizers.create(type, this.learningRate);
    }

    /**
     * Set activation function
     */
    setActivation(name) {
        this.activationName = name;
        this.activation = Activations[name] || Activations.tanh;
    }

    /**
     * Set regularization
     */
    setRegularization(l2, dropout) {
        this.l2Lambda = l2;
        this.dropoutRate = dropout;
    }

    /**
     * Get neuron values for visualization
     */
    getNeuronValues() {
        return this.layerOutputs.map(layer => layer.toArray());
    }

    /**
     * Get weight info for visualization
     */
    getWeightInfo() {
        return this.weights.map((w, i) => ({
            layer: i,
            rows: w.rows,
            cols: w.cols,
            data: w.data,
            min: w.min(),
            max: w.max(),
            mean: w.mean()
        }));
    }

    /**
     * Export configuration
     */
    export() {
        return {
            layers: this.layerSizes,
            activation: this.activationName,
            optimizer: this.optimizerType,
            learningRate: this.learningRate,
            l2: this.l2Lambda,
            dropout: this.dropoutRate,
            weights: this.weights.map(w => w.data),
            biases: this.biases.map(b => b.data)
        };
    }

    /**
     * Import configuration
     */
    import(config) {
        this.layerSizes = config.layers;
        this.activationName = config.activation;
        this.activation = Activations[config.activation];
        this.optimizerType = config.optimizer;
        this.learningRate = config.learningRate;
        this.l2Lambda = config.l2 || 0;
        this.dropoutRate = config.dropout || 0;

        this.weights = config.weights.map((data, i) => {
            const m = new Matrix(data.length, data[0].length);
            m.data = data;
            return m;
        });

        this.biases = config.biases.map((data, i) => {
            const m = new Matrix(data.length, 1);
            m.data = data;
            return m;
        });

        this.optimizer = Optimizers.create(this.optimizerType, this.learningRate);
    }
}

// Export
window.NeuralNetwork = NeuralNetwork;
