# Neural Network Playground

<div align="center">

![Neural Network Playground](https://img.shields.io/badge/Neural%20Network-Playground-blue?style=for-the-badge)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow?style=for-the-badge&logo=javascript)
![No Dependencies](https://img.shields.io/badge/Dependencies-Zero-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**An interactive neural network visualization tool built entirely from scratch.**

*No TensorFlow. No PyTorch. Just pure mathematics and JavaScript.*

[**Live Demo**](https://skemato.com/nn-playground/) | [**Report Bug**](https://github.com/youssefhajaj/neural-network-playground/issues) | [**Request Feature**](https://github.com/youssefhajaj/neural-network-playground/issues)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Live Demo](#live-demo)
- [Installation](#installation)
  - [Quick Start (No Installation)](#quick-start-no-installation)
  - [Local Development](#local-development)
  - [Using Docker](#using-docker)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [Mathematical Foundations](#mathematical-foundations)
  - [Forward Propagation](#forward-propagation)
  - [Backward Propagation](#backward-propagation)
  - [Activation Functions](#activation-functions)
  - [Loss Functions](#loss-functions)
  - [Optimization Algorithms](#optimization-algorithms)
  - [Weight Initialization](#weight-initialization)
  - [Regularization Techniques](#regularization-techniques)
- [Algorithm Complexity Analysis](#algorithm-complexity-analysis)
- [Code Structure](#code-structure)
- [Configuration Options](#configuration-options)
- [Python Export](#python-export)
- [Browser Compatibility](#browser-compatibility)
- [Performance Optimizations](#performance-optimizations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Neural Network Playground is a fully interactive, web-based tool for understanding and experimenting with neural networks. Unlike other educational tools that rely on machine learning libraries, this project implements **every single component from scratch** using pure JavaScript — from matrix operations to backpropagation, from gradient descent to Adam optimizer.

### Why This Project?

- **Educational**: Understand the inner workings of neural networks without library abstractions
- **Interactive**: Real-time visualization of training, decision boundaries, and weight distributions
- **Zero Dependencies**: No TensorFlow, PyTorch, or any ML library — just vanilla JavaScript
- **Comprehensive**: Implements multiple optimizers, activation functions, regularization techniques
- **Exportable**: Generate standalone Python code to reproduce your experiments

---

## Features

### Network Architecture
- **Dynamic Layer Builder**: Add/remove hidden layers (up to 6 layers)
- **Flexible Neurons**: 1-16 neurons per layer
- **Quick Presets**: Simple [4], Medium [4,4], Deep [4,4,4,4], Wide [8,8]
- **Real-time Parameter Count**: See total trainable parameters

### Training Controls
- **Train/Pause/Reset**: Full control over training
- **Step Mode**: Train one epoch at a time
- **Speed Control**: 1x, 3x, 10x, 50x (Max)
- **Max Epochs**: Configurable limit (10-1000)

### Datasets (6 Types)
| Dataset | Description | Difficulty |
|---------|-------------|------------|
| **XOR** | Classic non-linearly separable problem | Easy |
| **Circle** | Points inside vs outside a circle | Easy |
| **Spiral** | Two interleaving spirals | Hard |
| **Moons** | Two crescent moon shapes | Medium |
| **Clusters** | Gaussian blob clusters | Easy |
| **Rings** | Concentric circles | Medium |

### Hyperparameters
- **Learning Rate**: 0.001 - 1.0 (logarithmic scale)
- **Batch Size**: 1, 2, 4, 8, 16, 32, 64
- **Activation Functions**: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish
- **Optimizers**: SGD, Momentum, RMSprop, Adam
- **L2 Regularization**: 0 - 0.1
- **Dropout**: 0% - 50%

### Visualizations
- **Network Architecture**: Real-time neuron activations with color gradients
- **Decision Boundary**: Heatmap showing classification regions
- **Training Metrics**: Loss and accuracy curves
- **Weight Heatmap**: Visualize weight distributions per layer

### Additional Features
- **Dark/Light Theme**: Toggle between color schemes
- **Training Log**: Detailed epoch-by-epoch statistics
- **CSV Export**: Download training history
- **Python Export**: Generate standalone Python code
- **Neuron Inspector**: Hover to see individual neuron values

---

## Live Demo

**Try it now**: [https://skemato.com/nn-playground/](https://skemato.com/nn-playground/)

No installation required. Works in any modern browser.

---

## Installation

### Quick Start (No Installation)

Simply visit the [Live Demo](https://skemato.com/nn-playground/) — no installation required!

### Local Development

#### Prerequisites
- Any web server (Apache, Nginx, Python, Node.js, etc.)
- Modern web browser (Chrome, Firefox, Safari, Edge)

#### Option 1: Python HTTP Server

```bash
# Clone the repository
git clone https://github.com/youssefhajaj/neural-network-playground.git
cd neural-network-playground

# Start a local server (Python 3)
python -m http.server 8000

# Or with Python 2
python -m SimpleHTTPServer 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

#### Option 2: Node.js HTTP Server

```bash
# Clone the repository
git clone https://github.com/youssefhajaj/neural-network-playground.git
cd neural-network-playground

# Install http-server globally (one-time)
npm install -g http-server

# Start the server
http-server -p 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

#### Option 3: PHP Built-in Server

```bash
# Clone the repository
git clone https://github.com/youssefhajaj/neural-network-playground.git
cd neural-network-playground

# Start PHP server
php -S localhost:8000
```

#### Option 4: Live Server (VS Code)

1. Clone the repository
2. Open in VS Code
3. Install "Live Server" extension
4. Right-click `index.html` → "Open with Live Server"

### Using Docker

```bash
# Clone the repository
git clone https://github.com/youssefhajaj/neural-network-playground.git
cd neural-network-playground

# Build and run with Docker
docker run -d -p 8080:80 -v $(pwd):/usr/share/nginx/html:ro nginx:alpine
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

### Apache/Nginx Deployment

Simply copy all files to your web server's document root:

```bash
# For Apache (typical path)
cp -r * /var/www/html/nn-playground/

# For Nginx (typical path)
cp -r * /usr/share/nginx/html/nn-playground/
```

---

## Technologies Used

### Frontend Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **HTML5** | Structure & Canvas API | - |
| **CSS3** | Styling, Animations, Flexbox/Grid | - |
| **JavaScript (ES6+)** | Core logic, Classes, Modules | ES2015+ |
| **Canvas API** | Real-time visualizations | HTML5 |
| **KaTeX** | Mathematical equation rendering | 0.16.9 |

### Custom Implementations (No External Libraries)

| Component | Description | Lines of Code |
|-----------|-------------|---------------|
| **Matrix Library** | Full matrix algebra operations | ~420 |
| **Neural Network** | Forward/backward propagation | ~470 |
| **Optimizers** | SGD, Momentum, RMSprop, Adam | ~285 |
| **Activation Functions** | 9 functions with derivatives | ~190 |
| **Dataset Generators** | 6 procedural datasets | ~260 |
| **Visualizer** | Canvas-based rendering | ~610 |
| **Application** | UI orchestration | ~1350 |

**Total**: ~6,200 lines of hand-written code

---

## Architecture

```
neural-network-playground/
├── index.html              # Main HTML structure (671 lines)
├── css/
│   └── style.css           # Complete styling (1,720 lines)
└── js/
    ├── utils.js            # Utility functions (222 lines)
    ├── matrix.js           # Matrix operations library (421 lines)
    ├── activations.js      # Activation & loss functions (192 lines)
    ├── optimizers.js       # Optimization algorithms (285 lines)
    ├── datasets.js         # Dataset generators (260 lines)
    ├── network.js          # Neural network implementation (472 lines)
    ├── visualizer.js       # Canvas rendering (608 lines)
    └── app.js              # Main application orchestration (1,351 lines)
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              index.html                                  │
│                         (UI Structure & Layout)                          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │    style.css    │             │     app.js      │
          │   (Styling)     │             │  (Orchestrator) │
          └─────────────────┘             └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
                    ▼                              ▼                              ▼
          ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
          │   network.js    │            │  visualizer.js  │            │   datasets.js   │
          │ (Neural Network)│            │ (Canvas Render) │            │  (Data Gen)     │
          └────────┬────────┘            └─────────────────┘            └─────────────────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
┌─────────┐ ┌───────────┐ ┌───────────────┐
│matrix.js│ │optimizers │ │activations.js │
│(Algebra)│ │   .js     │ │ (Functions)   │
└─────────┘ └───────────┘ └───────────────┘
```

---

## Mathematical Foundations

### Forward Propagation

For each layer $l$ in the network:

$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$$

$$a^{[l]} = g(z^{[l]})$$

Where:
- $W^{[l]}$ = Weight matrix for layer $l$
- $b^{[l]}$ = Bias vector for layer $l$
- $a^{[l-1]}$ = Activation from previous layer (input for layer 0)
- $g(\cdot)$ = Activation function
- $z^{[l]}$ = Linear transformation (pre-activation)
- $a^{[l]}$ = Activation output

### Backward Propagation

**Output Layer Error (MSE Loss):**

$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial a^{[L]}} \odot g'(z^{[L]}) = 2(a^{[L]} - y) \odot g'(z^{[L]})$$

**Hidden Layer Error:**

$$\delta^{[l]} = (W^{[l+1]})^T \cdot \delta^{[l+1]} \odot g'(z^{[l]})$$

**Gradients:**

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} \cdot (a^{[l-1]})^T$$

$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$

### Activation Functions

| Function | Formula | Derivative | Use Case |
|----------|---------|------------|----------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1 + e^{-x}}$ | $\sigma(x)(1 - \sigma(x))$ | Output layer (binary) |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | Hidden layers |
| **ReLU** | $\max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | Deep networks |
| **Leaky ReLU** | $\max(0.01x, x)$ | $\begin{cases} 1 & x > 0 \\ 0.01 & x \leq 0 \end{cases}$ | Avoid dead neurons |
| **ELU** | $\begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$ | $\begin{cases} 1 & x > 0 \\ f(x) + \alpha & x \leq 0 \end{cases}$ | Smooth ReLU |
| **Swish** | $x \cdot \sigma(x)$ | $\sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))$ | Modern deep networks |

### Loss Functions

**Mean Squared Error (MSE):**

$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**With L2 Regularization:**

$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \frac{\lambda}{2} \sum_{l} \|W^{[l]}\|_F^2$$

Where $\|\cdot\|_F$ is the Frobenius norm.

### Optimization Algorithms

#### Stochastic Gradient Descent (SGD)

$$W = W - \eta \cdot \nabla_W \mathcal{L}$$

#### Momentum

$$v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla_W \mathcal{L}$$
$$W = W - v_t$$

Where $\beta = 0.9$ (momentum coefficient)

#### RMSprop

$$s_t = \rho \cdot s_{t-1} + (1 - \rho) \cdot (\nabla_W \mathcal{L})^2$$
$$W = W - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot \nabla_W \mathcal{L}$$

Where $\rho = 0.9$, $\epsilon = 10^{-8}$

#### Adam (Adaptive Moment Estimation)

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla_W \mathcal{L}$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla_W \mathcal{L})^2$$

**Bias Correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update Rule:**
$$W = W - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### Weight Initialization

#### Xavier/Glorot Initialization (for Sigmoid, Tanh)

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

#### He Initialization (for ReLU variants)

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

### Regularization Techniques

#### L2 Regularization (Weight Decay)

Adds penalty term to loss:
$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2} \sum_{l} \sum_{i,j} (W_{ij}^{[l]})^2$$

Gradient modification:
$$\frac{\partial \mathcal{L}_{reg}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial W^{[l]}} + \lambda W^{[l]}$$

#### Dropout (Inverted Dropout)

During training:
1. Generate mask $M$ where $M_{ij} \sim \text{Bernoulli}(1-p)$
2. Apply: $a^{[l]} = \frac{a^{[l]} \odot M}{1-p}$

During inference:
- No dropout applied (scaling already handled during training)

---

## Algorithm Complexity Analysis

### Time Complexity

| Operation | Complexity | Description |
|-----------|------------|-------------|
| **Forward Pass (per sample)** | $O(\sum_{l=1}^{L} n_l \cdot n_{l-1})$ | Matrix multiplication per layer |
| **Backward Pass (per sample)** | $O(\sum_{l=1}^{L} n_l \cdot n_{l-1})$ | Gradient computation per layer |
| **Single Epoch** | $O(N \cdot \sum_{l=1}^{L} n_l \cdot n_{l-1})$ | N = number of samples |
| **Matrix Multiplication** | $O(n \cdot m \cdot k)$ | For matrices (n×m) × (m×k) |
| **Decision Boundary Render** | $O(R^2 \cdot \sum_{l=1}^{L} n_l \cdot n_{l-1})$ | R = resolution (100×100) |

### Space Complexity

| Component | Complexity | Description |
|-----------|------------|-------------|
| **Weights** | $O(\sum_{l=1}^{L} n_l \cdot n_{l-1})$ | All weight matrices |
| **Biases** | $O(\sum_{l=1}^{L} n_l)$ | All bias vectors |
| **Activations Cache** | $O(\sum_{l=0}^{L} n_l)$ | For backpropagation |
| **Optimizer State (Adam)** | $O(2 \cdot \sum_{l=1}^{L} n_l \cdot n_{l-1})$ | m and v for each weight |
| **Dataset** | $O(N \cdot d)$ | N samples, d dimensions |

### Example Calculation

For a network `[2, 8, 8, 1]`:
- **Total Parameters**: $(2 \times 8 + 8) + (8 \times 8 + 8) + (8 \times 1 + 1) = 24 + 72 + 9 = 105$
- **Forward Pass FLOPs**: $2 \times 8 + 8 \times 8 + 8 \times 1 = 16 + 64 + 8 = 88$ multiplications per sample
- **Memory (Adam)**: $105 \times 3 \times 8$ bytes $\approx 2.5$ KB (weights + m + v, 64-bit floats)

---

## Code Structure

### Matrix Operations (`matrix.js`)

```javascript
class Matrix {
    // Creation
    static fromArray(arr)           // Create column vector from array
    clone()                         // Deep copy

    // Initialization
    xavierInit(fanIn, fanOut)       // Xavier/Glorot initialization
    heInit(fanIn)                   // He initialization
    randomize(scale)                // Random uniform [-scale, scale]
    zeros()                         // Fill with zeros

    // Arithmetic (Static - returns new matrix)
    static multiply(a, b)           // Matrix multiplication
    static add(a, b)                // Element-wise addition
    static subtract(a, b)           // Element-wise subtraction
    static hadamard(a, b)           // Element-wise multiplication
    static scale(m, scalar)         // Scalar multiplication
    static transpose(m)             // Matrix transpose

    // Arithmetic (Instance - modifies in place)
    add(m)                          // In-place addition
    subtract(m)                     // In-place subtraction
    scale(scalar)                   // In-place scaling

    // Utilities
    map(fn)                         // Apply function to each element
    clip(min, max)                  // Clamp values to range
    sum()                           // Sum all elements
    sumSquared()                    // Sum of squares (for L2)
    mean()                          // Average of all elements
    max() / min()                   // Maximum/minimum value
    toArray()                       // Flatten to 1D array
}
```

### Neural Network (`network.js`)

```javascript
class NeuralNetwork {
    constructor(config) {
        // config: { layers, activation, optimizer, learningRate, l2, dropout }
    }

    // Core Methods
    forward(input, training)        // Forward propagation
    backward(target)                // Backward propagation
    trainBatch(inputs, targets)     // Train on batch, return loss
    trainEpoch(trainData, batchSize) // Train full epoch

    // Prediction
    predict(input)                  // Get network output
    calculateAccuracy(inputs, targets) // Classification accuracy
    calculateLoss(inputs, targets)  // MSE loss on dataset

    // Configuration
    setLearningRate(lr)             // Update learning rate
    setOptimizer(type)              // Change optimizer
    setActivation(name)             // Change activation function
    setRegularization(l2, dropout)  // Set regularization params

    // State
    reset()                         // Reinitialize weights
    getParameterCount()             // Total trainable parameters
    getNeuronValues()               // Get all layer activations
    getWeightInfo()                 // Get weight statistics

    // Export/Import
    export()                        // Serialize network config
    import(config)                  // Load network config
}
```

### Optimizers (`optimizers.js`)

```javascript
// Base class
class Optimizer {
    constructor(learningRate)
    update(weights, biases, weightGrads, biasGrads)
    setLearningRate(lr)
}

// Implementations
class SGD extends Optimizer { }
class Momentum extends Optimizer { }
class RMSprop extends Optimizer { }
class Adam extends Optimizer { }

// Factory
Optimizers.create(type, learningRate)  // Create optimizer by name
```

### Activation Functions (`activations.js`)

```javascript
const Activations = {
    sigmoid: {
        fn: (x) => ...,
        derivative: (output) => ...,
        range: [0, 1]
    },
    tanh: { ... },
    relu: { ..., needsInput: true },
    'leaky-relu': { ..., needsInput: true },
    elu: { ..., needsInput: true },
    swish: { ..., needsInput: true },
    linear: { ... }
};
```

---

## Configuration Options

### Network Configuration

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `hiddenLayers` | 1-6 layers | `[4, 4]` | Hidden layer sizes |
| `neuronsPerLayer` | 1-16 | 4 | Neurons in each layer |
| `activation` | see below | `tanh` | Activation function |
| `optimizer` | see below | `adam` | Optimization algorithm |

### Training Configuration

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `learningRate` | 0.001-1.0 | 0.03 | Step size for updates |
| `batchSize` | 1-64 | 32 | Samples per update |
| `maxEpochs` | 10-1000 | 150 | Maximum training epochs |
| `speed` | 1/3/10/50 | 3 | Epochs per animation frame |

### Regularization

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `l2` | 0-0.1 | 0 | L2 regularization strength |
| `dropout` | 0-0.5 | 0 | Dropout probability |

### Dataset Configuration

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `datasetType` | 6 types | `spiral` | Dataset pattern |
| `noise` | 0-0.5 | 0.15 | Random noise level |
| `trainRatio` | 0.5-0.9 | 0.8 | Train/test split ratio |

---

## Python Export

The playground can export your configuration as a standalone Python script. The generated code:

- **Zero dependencies** except NumPy and Matplotlib
- **Identical implementation** to the JavaScript version
- **Complete training loop** with visualization
- **Reproducible results** (same hyperparameters)

### Example Export

```python
"""
Neural Network from Scratch - Exported from NN Playground
No PyTorch, No TensorFlow - Pure NumPy Implementation
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuration
LAYERS = [2, 4, 4, 1]
ACTIVATION = "tanh"
OPTIMIZER = "adam"
LEARNING_RATE = 0.03
BATCH_SIZE = 32
MAX_EPOCHS = 150

# ... full implementation included
```

---

## Browser Compatibility

| Browser | Version | Status |
|---------|---------|--------|
| Chrome | 80+ | ✅ Fully Supported |
| Firefox | 75+ | ✅ Fully Supported |
| Safari | 13+ | ✅ Fully Supported |
| Edge | 80+ | ✅ Fully Supported |
| Opera | 70+ | ✅ Fully Supported |
| IE 11 | - | ❌ Not Supported |

### Required Browser Features
- ES6+ JavaScript (Classes, Arrow Functions, Template Literals)
- Canvas API
- CSS Grid & Flexbox
- CSS Custom Properties (Variables)

---

## Performance Optimizations

### Implemented Optimizations

1. **Selective Rendering**: Decision boundary only updates every 3rd frame during training
2. **Resolution Options**: Low (50×50), Medium (75×75), High (100×100) for decision boundary
3. **Gradient Clipping**: Prevents exploding gradients (max norm = 5.0)
4. **Value Clipping**: Pre-activation values clipped to [-50, 50]
5. **Efficient Matrix Operations**: Direct array access instead of method calls
6. **Lazy Initialization**: Optimizers initialize state on first use
7. **Canvas Optimization**: Device pixel ratio handling for crisp rendering

### Numerical Stability

1. **Sigmoid Clipping**: Input clipped to [-500, 500] to prevent overflow
2. **Loss Clipping**: Difference values clipped to [-10, 10]
3. **NaN Handling**: Returns fallback value (10) if loss becomes NaN
4. **Finite Checks**: All activation outputs validated

---

## Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **Bug Reports**: Open an issue with reproduction steps
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve README or add code comments
5. **Testing**: Report browser compatibility issues

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/neural-network-playground.git
cd neural-network-playground

# Start local server
python -m http.server 8000

# Make changes and test
# Submit pull request
```

### Code Style

- Use ES6+ features
- Maintain consistent indentation (4 spaces)
- Add JSDoc comments for functions
- Keep functions focused and small
- Use meaningful variable names

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Youssef Hajaj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

- **Mathematical Foundations**: Based on standard deep learning theory and practices
- **Visualization Inspiration**: TensorFlow Playground by Google
- **Color Schemes**: Inspired by modern UI design principles

---

<div align="center">

**Built with passion for education and understanding**

Made by [Youssef Hajaj](https://github.com/youssefhajaj)

</div>
