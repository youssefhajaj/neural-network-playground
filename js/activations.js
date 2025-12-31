/**
 * Activation Functions
 * Each function includes both the activation and its derivative for backpropagation
 */

const Activations = {
    /**
     * Sigmoid: σ(x) = 1 / (1 + e^-x)
     * Range: (0, 1)
     * Good for: Output layer in binary classification
     */
    sigmoid: {
        name: 'Sigmoid',
        fn: (x) => {
            const clipped = Math.max(-500, Math.min(500, x));
            return 1 / (1 + Math.exp(-clipped));
        },
        derivative: (output) => output * (1 - output),
        range: [0, 1]
    },

    /**
     * Hyperbolic Tangent: tanh(x)
     * Range: (-1, 1)
     * Good for: Hidden layers, centered output
     */
    tanh: {
        name: 'Tanh',
        fn: (x) => Math.tanh(x),
        derivative: (output) => 1 - output * output,
        range: [-1, 1]
    },

    /**
     * ReLU: max(0, x)
     * Range: [0, ∞)
     * Good for: Hidden layers in deep networks
     */
    relu: {
        name: 'ReLU',
        fn: (x) => Math.max(0, x),
        derivative: (output, input) => input > 0 ? 1 : 0,
        range: [0, Infinity],
        needsInput: true
    },

    /**
     * Leaky ReLU: max(αx, x) where α = 0.01
     * Range: (-∞, ∞)
     * Good for: Avoiding dead neurons
     */
    'leaky-relu': {
        name: 'Leaky ReLU',
        fn: (x) => x > 0 ? x : 0.01 * x,
        derivative: (output, input) => input > 0 ? 1 : 0.01,
        range: [-Infinity, Infinity],
        needsInput: true
    },

    /**
     * ELU: x if x > 0, else α(e^x - 1)
     * Range: (-α, ∞)
     * Good for: Smooth approximation of ReLU
     */
    elu: {
        name: 'ELU',
        fn: (x, alpha = 1) => x > 0 ? x : alpha * (Math.exp(Math.min(x, 100)) - 1),
        derivative: (output, input, alpha = 1) => input > 0 ? 1 : output + alpha,
        range: [-1, Infinity],
        needsInput: true
    },

    /**
     * Swish: x * sigmoid(x)
     * Range: (-0.28, ∞)
     * Good for: Modern deep networks
     */
    swish: {
        name: 'Swish',
        fn: (x) => {
            const sig = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
            return x * sig;
        },
        derivative: (output, input) => {
            const sig = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, input))));
            return sig + input * sig * (1 - sig);
        },
        range: [-0.28, Infinity],
        needsInput: true
    },

    /**
     * Linear: f(x) = x
     * Range: (-∞, ∞)
     * Good for: Output layer in regression
     */
    linear: {
        name: 'Linear',
        fn: (x) => x,
        derivative: () => 1,
        range: [-Infinity, Infinity]
    },

    /**
     * Softplus: ln(1 + e^x)
     * Range: (0, ∞)
     * Good for: Smooth approximation of ReLU
     */
    softplus: {
        name: 'Softplus',
        fn: (x) => Math.log(1 + Math.exp(Math.min(100, x))),
        derivative: (output, input) => 1 / (1 + Math.exp(-input)),
        range: [0, Infinity],
        needsInput: true
    },

    /**
     * GELU: Gaussian Error Linear Unit
     * Range: (-0.17, ∞)
     * Good for: Transformer models
     */
    gelu: {
        name: 'GELU',
        fn: (x) => {
            // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            const c = Math.sqrt(2 / Math.PI);
            return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
        },
        derivative: (output, input) => {
            const c = Math.sqrt(2 / Math.PI);
            const x3 = input * input * input;
            const inner = c * (input + 0.044715 * x3);
            const tanhInner = Math.tanh(inner);
            const sech2 = 1 - tanhInner * tanhInner;
            return 0.5 * (1 + tanhInner) + 0.5 * input * sech2 * c * (1 + 0.134145 * input * input);
        },
        range: [-0.17, Infinity],
        needsInput: true
    }
};

/**
 * Loss Functions
 */
const LossFunctions = {
    /**
     * Mean Squared Error
     * L = (1/n) Σ(y - ŷ)²
     */
    mse: {
        name: 'Mean Squared Error',
        fn: (predicted, actual) => {
            let sum = 0;
            for (let i = 0; i < predicted.length; i++) {
                const diff = predicted[i] - actual[i];
                sum += diff * diff;
            }
            return sum / predicted.length;
        },
        derivative: (predicted, actual) => {
            return predicted.map((p, i) => 2 * (p - actual[i]) / predicted.length);
        }
    },

    /**
     * Binary Cross Entropy
     * L = -(1/n) Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
     */
    bce: {
        name: 'Binary Cross Entropy',
        fn: (predicted, actual) => {
            const eps = 1e-15;
            let sum = 0;
            for (let i = 0; i < predicted.length; i++) {
                const p = Math.max(eps, Math.min(1 - eps, predicted[i]));
                sum -= actual[i] * Math.log(p) + (1 - actual[i]) * Math.log(1 - p);
            }
            return sum / predicted.length;
        },
        derivative: (predicted, actual) => {
            const eps = 1e-15;
            return predicted.map((p, i) => {
                const pClip = Math.max(eps, Math.min(1 - eps, p));
                return (pClip - actual[i]) / (pClip * (1 - pClip) * predicted.length);
            });
        }
    }
};

// Export
window.Activations = Activations;
window.LossFunctions = LossFunctions;
