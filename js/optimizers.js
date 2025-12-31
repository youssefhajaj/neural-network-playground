/**
 * Optimizers
 * Implementation of various gradient descent optimization algorithms
 */

class Optimizer {
    constructor(learningRate = 0.01) {
        this.learningRate = learningRate;
    }

    initialize(weights, biases) {
        // Override in subclasses if needed
    }

    update(weights, biases, weightGrads, biasGrads) {
        throw new Error('update() must be implemented by subclass');
    }

    setLearningRate(lr) {
        this.learningRate = lr;
    }
}

/**
 * Stochastic Gradient Descent
 * The simplest optimizer: W = W - lr * dW
 */
class SGD extends Optimizer {
    constructor(learningRate = 0.01) {
        super(learningRate);
        this.name = 'SGD';
    }

    update(weights, biases, weightGrads, biasGrads) {
        for (let i = 0; i < weights.length; i++) {
            // W = W - lr * dW
            for (let r = 0; r < weights[i].rows; r++) {
                for (let c = 0; c < weights[i].cols; c++) {
                    weights[i].data[r][c] -= this.learningRate * weightGrads[i].data[r][c];
                }
            }

            // b = b - lr * db
            for (let r = 0; r < biases[i].rows; r++) {
                biases[i].data[r][0] -= this.learningRate * biasGrads[i].data[r][0];
            }
        }
    }
}

/**
 * SGD with Momentum
 * Adds velocity to escape local minima and speed up convergence
 * v = momentum * v - lr * dW
 * W = W + v
 */
class Momentum extends Optimizer {
    constructor(learningRate = 0.01, momentum = 0.9) {
        super(learningRate);
        this.momentum = momentum;
        this.velocityW = null;
        this.velocityB = null;
        this.name = 'Momentum';
    }

    initialize(weights, biases) {
        this.velocityW = weights.map(w => new Matrix(w.rows, w.cols));
        this.velocityB = biases.map(b => new Matrix(b.rows, b.cols));
    }

    update(weights, biases, weightGrads, biasGrads) {
        if (!this.velocityW) {
            this.initialize(weights, biases);
        }

        for (let i = 0; i < weights.length; i++) {
            // Update weights
            for (let r = 0; r < weights[i].rows; r++) {
                for (let c = 0; c < weights[i].cols; c++) {
                    // v = momentum * v - lr * dW
                    this.velocityW[i].data[r][c] =
                        this.momentum * this.velocityW[i].data[r][c] -
                        this.learningRate * weightGrads[i].data[r][c];

                    // W = W + v
                    weights[i].data[r][c] += this.velocityW[i].data[r][c];
                }
            }

            // Update biases
            for (let r = 0; r < biases[i].rows; r++) {
                this.velocityB[i].data[r][0] =
                    this.momentum * this.velocityB[i].data[r][0] -
                    this.learningRate * biasGrads[i].data[r][0];

                biases[i].data[r][0] += this.velocityB[i].data[r][0];
            }
        }
    }
}

/**
 * RMSprop
 * Adapts learning rate based on running average of squared gradients
 * cache = decay * cache + (1 - decay) * dW^2
 * W = W - lr * dW / (sqrt(cache) + eps)
 */
class RMSprop extends Optimizer {
    constructor(learningRate = 0.001, decay = 0.9, epsilon = 1e-8) {
        super(learningRate);
        this.decay = decay;
        this.epsilon = epsilon;
        this.cacheW = null;
        this.cacheB = null;
        this.name = 'RMSprop';
    }

    initialize(weights, biases) {
        this.cacheW = weights.map(w => new Matrix(w.rows, w.cols));
        this.cacheB = biases.map(b => new Matrix(b.rows, b.cols));
    }

    update(weights, biases, weightGrads, biasGrads) {
        if (!this.cacheW) {
            this.initialize(weights, biases);
        }

        for (let i = 0; i < weights.length; i++) {
            // Update weights
            for (let r = 0; r < weights[i].rows; r++) {
                for (let c = 0; c < weights[i].cols; c++) {
                    const grad = weightGrads[i].data[r][c];

                    // cache = decay * cache + (1 - decay) * dW^2
                    this.cacheW[i].data[r][c] =
                        this.decay * this.cacheW[i].data[r][c] +
                        (1 - this.decay) * grad * grad;

                    // W = W - lr * dW / (sqrt(cache) + eps)
                    weights[i].data[r][c] -=
                        this.learningRate * grad /
                        (Math.sqrt(this.cacheW[i].data[r][c]) + this.epsilon);
                }
            }

            // Update biases
            for (let r = 0; r < biases[i].rows; r++) {
                const grad = biasGrads[i].data[r][0];

                this.cacheB[i].data[r][0] =
                    this.decay * this.cacheB[i].data[r][0] +
                    (1 - this.decay) * grad * grad;

                biases[i].data[r][0] -=
                    this.learningRate * grad /
                    (Math.sqrt(this.cacheB[i].data[r][0]) + this.epsilon);
            }
        }
    }
}

/**
 * Adam (Adaptive Moment Estimation)
 * Combines momentum and RMSprop with bias correction
 * m = beta1 * m + (1 - beta1) * dW          (first moment - momentum)
 * v = beta2 * v + (1 - beta2) * dW^2        (second moment - RMSprop)
 * m_hat = m / (1 - beta1^t)                  (bias correction)
 * v_hat = v / (1 - beta2^t)                  (bias correction)
 * W = W - lr * m_hat / (sqrt(v_hat) + eps)
 */
class Adam extends Optimizer {
    constructor(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.mW = null;  // First moment (momentum)
        this.mB = null;
        this.vW = null;  // Second moment (RMSprop)
        this.vB = null;
        this.t = 0;      // Timestep
        this.name = 'Adam';
    }

    initialize(weights, biases) {
        this.mW = weights.map(w => new Matrix(w.rows, w.cols));
        this.mB = biases.map(b => new Matrix(b.rows, b.cols));
        this.vW = weights.map(w => new Matrix(w.rows, w.cols));
        this.vB = biases.map(b => new Matrix(b.rows, b.cols));
        this.t = 0;
    }

    update(weights, biases, weightGrads, biasGrads) {
        if (!this.mW) {
            this.initialize(weights, biases);
        }

        this.t++;

        // Bias correction terms
        const beta1Correction = 1 - Math.pow(this.beta1, this.t);
        const beta2Correction = 1 - Math.pow(this.beta2, this.t);

        for (let i = 0; i < weights.length; i++) {
            // Update weights
            for (let r = 0; r < weights[i].rows; r++) {
                for (let c = 0; c < weights[i].cols; c++) {
                    const grad = weightGrads[i].data[r][c];

                    // m = beta1 * m + (1 - beta1) * dW
                    this.mW[i].data[r][c] =
                        this.beta1 * this.mW[i].data[r][c] +
                        (1 - this.beta1) * grad;

                    // v = beta2 * v + (1 - beta2) * dW^2
                    this.vW[i].data[r][c] =
                        this.beta2 * this.vW[i].data[r][c] +
                        (1 - this.beta2) * grad * grad;

                    // Bias-corrected estimates
                    const mHat = this.mW[i].data[r][c] / beta1Correction;
                    const vHat = this.vW[i].data[r][c] / beta2Correction;

                    // W = W - lr * m_hat / (sqrt(v_hat) + eps)
                    weights[i].data[r][c] -=
                        this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
                }
            }

            // Update biases
            for (let r = 0; r < biases[i].rows; r++) {
                const grad = biasGrads[i].data[r][0];

                this.mB[i].data[r][0] =
                    this.beta1 * this.mB[i].data[r][0] +
                    (1 - this.beta1) * grad;

                this.vB[i].data[r][0] =
                    this.beta2 * this.vB[i].data[r][0] +
                    (1 - this.beta2) * grad * grad;

                const mHat = this.mB[i].data[r][0] / beta1Correction;
                const vHat = this.vB[i].data[r][0] / beta2Correction;

                biases[i].data[r][0] -=
                    this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
            }
        }
    }

    reset() {
        this.mW = null;
        this.mB = null;
        this.vW = null;
        this.vB = null;
        this.t = 0;
    }
}

/**
 * Optimizer Factory
 */
const Optimizers = {
    sgd: (lr) => new SGD(lr),
    momentum: (lr) => new Momentum(lr, 0.9),
    rmsprop: (lr) => new RMSprop(lr, 0.9),
    adam: (lr) => new Adam(lr, 0.9, 0.999),

    create(type, learningRate) {
        const factory = this[type];
        if (!factory) {
            console.warn(`Unknown optimizer: ${type}, falling back to Adam`);
            return this.adam(learningRate);
        }
        return factory(learningRate);
    }
};

// Export
window.Optimizer = Optimizer;
window.SGD = SGD;
window.Momentum = Momentum;
window.RMSprop = RMSprop;
window.Adam = Adam;
window.Optimizers = Optimizers;
