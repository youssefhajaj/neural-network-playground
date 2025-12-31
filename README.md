<div align="center">

# 🧠 Neural Network Playground

<img src="https://img.shields.io/badge/Built%20From-Scratch-ff6b6b?style=for-the-badge" alt="Built From Scratch"/>
<img src="https://img.shields.io/badge/JavaScript-ES6+-f7df1e?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript"/>
<img src="https://img.shields.io/badge/Dependencies-Zero-4ecdc4?style=for-the-badge" alt="Zero Dependencies"/>
<img src="https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge" alt="MIT License"/>

<br/>
<br/>

**✨ An interactive neural network visualization tool built entirely from scratch ✨**

*No TensorFlow. No PyTorch. No NumPy. Just pure mathematics and vanilla JavaScript.*

<br/>

[<img src="https://img.shields.io/badge/🚀%20Live%20Demo-Visit%20Now-28a745?style=for-the-badge" alt="Live Demo"/>](https://skemato.com/nn-playground/)
&nbsp;&nbsp;
[<img src="https://img.shields.io/badge/🐛%20Report-Bug-dc3545?style=for-the-badge" alt="Report Bug"/>](https://github.com/youssefhajaj/neural-network-playground/issues)
&nbsp;&nbsp;
[<img src="https://img.shields.io/badge/💡%20Request-Feature-ffc107?style=for-the-badge" alt="Request Feature"/>](https://github.com/youssefhajaj/neural-network-playground/issues)

<br/>
<br/>

---

<img width="800" src="https://raw.githubusercontent.com/youssefhajaj/neural-network-playground/main/screenshot.png" alt="Neural Network Playground Screenshot"/>

*Design custom architectures • Train on 6 datasets • Watch decision boundaries form in real-time*

---

</div>

<br/>

## 📋 Table of Contents

<details>
<summary>Click to expand</summary>

- [Overview](#-overview)
- [Features](#-features)
- [Live Demo](#-live-demo)
- [Installation](#-installation)
- [Technologies](#-technologies)
- [Architecture](#-architecture)
- [Mathematical Foundations](#-mathematical-foundations)
- [Algorithm Complexity](#-algorithm-complexity)
- [Code Structure](#-code-structure)
- [Configuration](#%EF%B8%8F-configuration)
- [Python Export](#-python-export)
- [Browser Support](#-browser-support)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

</details>

<br/>

---

## 🎯 Overview

Neural Network Playground is a **fully interactive, web-based tool** for understanding and experimenting with neural networks. Unlike other educational tools that rely on machine learning libraries, this project implements **every single component from scratch** — from matrix operations to backpropagation, from gradient descent to Adam optimizer.

<table>
<tr>
<td width="50%">

### 🎓 Why This Project?

- **📚 Educational** — Understand neural networks without library abstractions
- **⚡ Interactive** — Real-time visualization of training progress
- **🔧 Zero Dependencies** — Pure vanilla JavaScript implementation
- **📊 Comprehensive** — Multiple optimizers, activations, regularization
- **🐍 Exportable** — Generate standalone Python code

</td>
<td width="50%">

### 🔢 By The Numbers

| Metric | Value |
|--------|-------|
| Lines of Code | **~6,200** |
| External ML Libraries | **0** |
| Activation Functions | **6** |
| Optimizers | **4** |
| Datasets | **6** |
| File Size | **~232 KB** |

</td>
</tr>
</table>

<br/>

---

## ✨ Features

<table>
<tr>
<td width="33%" valign="top">

### 🏗️ Network Architecture

- Dynamic layer builder (1-6 layers)
- 1-16 neurons per layer
- Quick presets:
  - Simple `[4]`
  - Medium `[4,4]`
  - Deep `[4,4,4,4]`
  - Wide `[8,8]`
- Real-time parameter count

</td>
<td width="33%" valign="top">

### 🎮 Training Controls

- ▶️ Train (continuous)
- ⏸️ Pause anytime
- ⏭️ Step (single epoch)
- 🔄 Reset weights
- ⚡ Speed: 1x, 3x, 10x, 50x
- 📊 Max epochs: 10-1000

</td>
<td width="33%" valign="top">

### 📈 Visualizations

- 🕸️ Network architecture
- 🗺️ Decision boundary heatmap
- 📉 Loss & accuracy curves
- 🎨 Weight distribution
- 🔍 Neuron inspector
- 📋 Training log table

</td>
</tr>
</table>

<br/>

### 🎲 Datasets

<table>
<tr>
<td align="center" width="16%">
<h4>🔀 XOR</h4>
<sub>Classic non-linear</sub><br/>
<img src="https://img.shields.io/badge/Difficulty-Easy-4ecdc4?style=flat-square"/>
</td>
<td align="center" width="16%">
<h4>⭕ Circle</h4>
<sub>Inside vs outside</sub><br/>
<img src="https://img.shields.io/badge/Difficulty-Easy-4ecdc4?style=flat-square"/>
</td>
<td align="center" width="16%">
<h4>🌀 Spiral</h4>
<sub>Interleaving spirals</sub><br/>
<img src="https://img.shields.io/badge/Difficulty-Hard-ff6b6b?style=flat-square"/>
</td>
<td align="center" width="16%">
<h4>🌙 Moons</h4>
<sub>Two crescents</sub><br/>
<img src="https://img.shields.io/badge/Difficulty-Medium-ffd93d?style=flat-square"/>
</td>
<td align="center" width="16%">
<h4>☁️ Clusters</h4>
<sub>Gaussian blobs</sub><br/>
<img src="https://img.shields.io/badge/Difficulty-Easy-4ecdc4?style=flat-square"/>
</td>
<td align="center" width="16%">
<h4>💫 Rings</h4>
<sub>Concentric circles</sub><br/>
<img src="https://img.shields.io/badge/Difficulty-Medium-ffd93d?style=flat-square"/>
</td>
</tr>
</table>

<br/>

### ⚙️ Hyperparameters

<table>
<tr>
<td width="50%">

| Parameter | Options |
|-----------|---------|
| **Activation** | Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish |
| **Optimizer** | SGD, Momentum, RMSprop, Adam |
| **Learning Rate** | 0.001 → 1.0 (log scale) |
| **Batch Size** | 1, 2, 4, 8, 16, 32, 64 |

</td>
<td width="50%">

| Parameter | Options |
|-----------|---------|
| **L2 Regularization** | 0 → 0.1 |
| **Dropout** | 0% → 50% |
| **Noise** | 0 → 0.5 |
| **Train/Test Split** | 50% → 90% |

</td>
</tr>
</table>

<br/>

---

## 🚀 Live Demo

<div align="center">

### [**👉 skemato.com/nn-playground 👈**](https://skemato.com/nn-playground/)

**No installation required** — Works in any modern browser!

</div>

<br/>

---

## 💻 Installation

<details>
<summary><b>🐍 Option 1: Python HTTP Server</b> (Recommended)</summary>

<br/>

```bash
# Clone the repository
git clone https://github.com/youssefhajaj/neural-network-playground.git

# Navigate to directory
cd neural-network-playground

# Start server (Python 3)
python -m http.server 8000

# Or Python 2
python -m SimpleHTTPServer 8000
```

Open **http://localhost:8000** in your browser.

</details>

<details>
<summary><b>📦 Option 2: Node.js HTTP Server</b></summary>

<br/>

```bash
# Clone the repository
git clone https://github.com/youssefhajaj/neural-network-playground.git
cd neural-network-playground

# Install http-server globally (one-time)
npm install -g http-server

# Start the server
http-server -p 8000
```

Open **http://localhost:8000** in your browser.

</details>

<details>
<summary><b>🐘 Option 3: PHP Built-in Server</b></summary>

<br/>

```bash
# Clone the repository
git clone https://github.com/youssefhajaj/neural-network-playground.git
cd neural-network-playground

# Start PHP server
php -S localhost:8000
```

Open **http://localhost:8000** in your browser.

</details>

<details>
<summary><b>🆚 Option 4: VS Code Live Server</b></summary>

<br/>

1. Clone the repository
2. Open folder in VS Code
3. Install **"Live Server"** extension
4. Right-click `index.html` → **"Open with Live Server"**

</details>

<details>
<summary><b>🐳 Option 5: Docker</b></summary>

<br/>

```bash
# Clone the repository
git clone https://github.com/youssefhajaj/neural-network-playground.git
cd neural-network-playground

# Run with Docker
docker run -d -p 8080:80 -v $(pwd):/usr/share/nginx/html:ro nginx:alpine
```

Open **http://localhost:8080** in your browser.

</details>

<br/>

---

## 🛠️ Technologies

<div align="center">

| Technology | Purpose |
|:----------:|:-------:|
| <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white"/> | Structure & Canvas API |
| <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white"/> | Styling & Animations |
| <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black"/> | Core Logic (ES6+) |
| <img src="https://img.shields.io/badge/Canvas-API-orange?style=for-the-badge"/> | Real-time Visualization |
| <img src="https://img.shields.io/badge/KaTeX-0.16.9-green?style=for-the-badge"/> | Math Rendering |

</div>

<br/>

### 🔧 Custom Implementations

> **Everything below was built from scratch — no external libraries!**

| Module | Description | Lines |
|--------|-------------|:-----:|
| `matrix.js` | Complete matrix algebra library | 421 |
| `network.js` | Neural network (forward/backward prop) | 472 |
| `optimizers.js` | SGD, Momentum, RMSprop, Adam | 285 |
| `activations.js` | 9 activation functions + derivatives | 192 |
| `datasets.js` | 6 procedural dataset generators | 260 |
| `visualizer.js` | Canvas-based rendering engine | 608 |
| `app.js` | Application orchestration | 1,351 |
| `utils.js` | Utility functions | 222 |

<br/>

---

## 🏛️ Architecture

```
neural-network-playground/
│
├── 📄 index.html          # Main HTML structure
├── 📄 README.md           # Documentation
├── 📄 LICENSE             # MIT License
│
├── 📁 css/
│   └── 🎨 style.css       # Complete styling (1,720 lines)
│
└── 📁 js/
    ├── 🔢 matrix.js       # Matrix operations
    ├── 🧠 network.js      # Neural network core
    ├── ⚡ optimizers.js   # Optimization algorithms
    ├── 📈 activations.js  # Activation functions
    ├── 🎲 datasets.js     # Dataset generators
    ├── 🎨 visualizer.js   # Canvas rendering
    ├── 🎮 app.js          # Main application
    └── 🔧 utils.js        # Utilities
```

<br/>

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         index.html                               │
│                      (UI Structure)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │   style.css     │           │     app.js      │
    │   (Styling)     │           │  (Orchestrator) │
    └─────────────────┘           └────────┬────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              ▼                            ▼                            ▼
    ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
    │   network.js    │          │  visualizer.js  │          │   datasets.js   │
    │ (Neural Network)│          │ (Canvas Render) │          │  (Data Gen)     │
    └────────┬────────┘          └─────────────────┘          └─────────────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
┌────────┐ ┌────────┐ ┌────────────┐
│matrix  │ │optim-  │ │activations │
│.js     │ │izers.js│ │.js         │
└────────┘ └────────┘ └────────────┘
```

<br/>

---

## 📐 Mathematical Foundations

<details open>
<summary><b>🔄 Forward Propagation</b></summary>

<br/>

For each layer *l* in the network:

$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$$

$$a^{[l]} = g(z^{[l]})$$

Where:
| Symbol | Meaning |
|--------|---------|
| $W^{[l]}$ | Weight matrix for layer *l* |
| $b^{[l]}$ | Bias vector for layer *l* |
| $a^{[l-1]}$ | Activation from previous layer |
| $g(\cdot)$ | Activation function |
| $z^{[l]}$ | Pre-activation (linear transform) |
| $a^{[l]}$ | Post-activation output |

</details>

<details>
<summary><b>⬅️ Backward Propagation</b></summary>

<br/>

**Output Layer Error (MSE):**

$$\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial a^{[L]}} \odot g'(z^{[L]}) = 2(a^{[L]} - y) \odot g'(z^{[L]})$$

**Hidden Layer Error:**

$$\delta^{[l]} = (W^{[l+1]})^T \cdot \delta^{[l+1]} \odot g'(z^{[l]})$$

**Gradients:**

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} \cdot (a^{[l-1]})^T$$

$$\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]}$$

</details>

<details>
<summary><b>⚡ Activation Functions</b></summary>

<br/>

| Function | Formula $f(x)$ | Derivative $f'(x)$ |
|:--------:|:--------------:|:------------------:|
| **Sigmoid** | $\frac{1}{1 + e^{-x}}$ | $f(x)(1 - f(x))$ |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - f(x)^2$ |
| **ReLU** | $\max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ |
| **Leaky ReLU** | $\max(0.01x, x)$ | $\begin{cases} 1 & x > 0 \\ 0.01 & x \leq 0 \end{cases}$ |
| **ELU** | $\begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$ | $\begin{cases} 1 & x > 0 \\ f(x) + \alpha & x \leq 0 \end{cases}$ |
| **Swish** | $x \cdot \sigma(x)$ | $\sigma(x) + x \cdot \sigma(x)(1 - \sigma(x))$ |

</details>

<details>
<summary><b>📉 Loss Function</b></summary>

<br/>

**Mean Squared Error (MSE):**

$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**With L2 Regularization:**

$$\mathcal{L}_{total} = \mathcal{L}_{MSE} + \frac{\lambda}{2} \sum_{l} \|W^{[l]}\|_F^2$$

</details>

<details>
<summary><b>🚀 Optimizers</b></summary>

<br/>

### SGD (Stochastic Gradient Descent)
$$W = W - \eta \cdot \nabla_W \mathcal{L}$$

---

### Momentum
$$v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla_W \mathcal{L}$$
$$W = W - v_t$$

*Where* $\beta = 0.9$

---

### RMSprop
$$s_t = \rho \cdot s_{t-1} + (1 - \rho) \cdot (\nabla_W \mathcal{L})^2$$
$$W = W - \frac{\eta}{\sqrt{s_t + \epsilon}} \cdot \nabla_W \mathcal{L}$$

*Where* $\rho = 0.9$, $\epsilon = 10^{-8}$

---

### Adam (Adaptive Moment Estimation)
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla_W \mathcal{L}$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla_W \mathcal{L})^2$$

**Bias Correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update:**
$$W = W - \frac{\eta \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

*Where* $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

</details>

<details>
<summary><b>🎯 Weight Initialization</b></summary>

<br/>

### Xavier/Glorot (for Sigmoid, Tanh)

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

### He (for ReLU variants)

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

</details>

<details>
<summary><b>🛡️ Regularization</b></summary>

<br/>

### L2 Regularization (Weight Decay)

**Loss modification:**
$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2} \sum_{l} \sum_{i,j} (W_{ij}^{[l]})^2$$

**Gradient modification:**
$$\frac{\partial \mathcal{L}_{reg}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W} + \lambda W$$

---

### Dropout (Inverted)

**During Training:**
1. Generate mask: $M_{ij} \sim \text{Bernoulli}(1-p)$
2. Apply & scale: $a = \frac{a \odot M}{1-p}$

**During Inference:**
- No dropout (scaling already handled)

</details>

<br/>

---

## ⏱️ Algorithm Complexity

### Time Complexity

| Operation | Complexity | Notes |
|-----------|:----------:|-------|
| Forward Pass (per sample) | $O(\sum_{l=1}^{L} n_l \cdot n_{l-1})$ | Matrix mult. per layer |
| Backward Pass (per sample) | $O(\sum_{l=1}^{L} n_l \cdot n_{l-1})$ | Gradient computation |
| Single Epoch | $O(N \cdot \sum_{l} n_l \cdot n_{l-1})$ | N = dataset size |
| Decision Boundary | $O(R^2 \cdot \text{Forward})$ | R = resolution |

### Space Complexity

| Component | Complexity | Notes |
|-----------|:----------:|-------|
| Weights | $O(\sum_{l} n_l \cdot n_{l-1})$ | All weight matrices |
| Biases | $O(\sum_{l} n_l)$ | All bias vectors |
| Adam State | $O(2 \times \text{Weights})$ | m and v terms |

<br/>

### 📊 Example: Network `[2, 8, 8, 1]`

```
Parameters:  (2×8+8) + (8×8+8) + (8×1+1) = 24 + 72 + 9 = 105 total
Forward:     2×8 + 8×8 + 8×1 = 88 multiplications per sample
Memory:      105 × 3 × 8 bytes ≈ 2.5 KB (weights + m + v, float64)
```

<br/>

---

## 📁 Code Structure

<details>
<summary><b>Matrix Operations</b> — <code>matrix.js</code></summary>

```javascript
class Matrix {
    // Creation
    static fromArray(arr)           // Column vector from 1D array
    clone()                         // Deep copy

    // Initialization
    xavierInit(fanIn, fanOut)       // Xavier/Glorot init
    heInit(fanIn)                   // He init (ReLU)
    zeros() / ones()                // Fill with constant

    // Static Operations (return new matrix)
    static multiply(a, b)           // Matrix multiplication
    static add(a, b)                // Element-wise addition
    static subtract(a, b)           // Element-wise subtraction
    static hadamard(a, b)           // Element-wise multiplication
    static transpose(m)             // Transpose

    // Instance Operations (in-place)
    add(m) / subtract(m)            // In-place arithmetic
    scale(scalar)                   // Scalar multiplication
    clip(min, max)                  // Clamp values
    map(fn)                         // Apply function

    // Utilities
    sum() / mean()                  // Aggregations
    min() / max()                   // Extremes
    sumSquared()                    // For L2 norm
    toArray()                       // Flatten to 1D
}
```

</details>

<details>
<summary><b>Neural Network</b> — <code>network.js</code></summary>

```javascript
class NeuralNetwork {
    constructor(config)             // { layers, activation, optimizer, ... }

    // Core
    forward(input, training)        // Forward propagation
    backward(target)                // Backward propagation
    trainBatch(inputs, targets)     // Train on mini-batch
    trainEpoch(data, batchSize)     // Full epoch

    // Prediction
    predict(input)                  // Get output
    calculateAccuracy(X, y)         // Classification accuracy
    calculateLoss(X, y)             // MSE loss

    // Configuration
    setLearningRate(lr)
    setOptimizer(type)
    setActivation(name)
    setRegularization(l2, dropout)

    // State
    reset()                         // Reinitialize
    getParameterCount()             // Total parameters
    export() / import(config)       // Serialization
}
```

</details>

<details>
<summary><b>Optimizers</b> — <code>optimizers.js</code></summary>

```javascript
class Optimizer {
    update(weights, biases, wGrads, bGrads)
    setLearningRate(lr)
}

class SGD extends Optimizer { }
class Momentum extends Optimizer { }
class RMSprop extends Optimizer { }
class Adam extends Optimizer { }

// Factory
Optimizers.create('adam', 0.001)
```

</details>

<br/>

---

## ⚙️ Configuration

### Network Settings

| Parameter | Range | Default |
|-----------|:-----:|:-------:|
| Hidden Layers | 1-6 | 2 |
| Neurons/Layer | 1-16 | 4 |
| Activation | 6 options | `tanh` |
| Optimizer | 4 options | `adam` |

### Training Settings

| Parameter | Range | Default |
|-----------|:-----:|:-------:|
| Learning Rate | 0.001-1.0 | 0.03 |
| Batch Size | 1-64 | 32 |
| Max Epochs | 10-1000 | 150 |
| Speed | 1/3/10/50x | 3x |

### Regularization

| Parameter | Range | Default |
|-----------|:-----:|:-------:|
| L2 Lambda | 0-0.1 | 0 |
| Dropout | 0-50% | 0% |

### Dataset

| Parameter | Range | Default |
|-----------|:-----:|:-------:|
| Type | 6 options | `spiral` |
| Noise | 0-0.5 | 0.15 |
| Train Ratio | 50-90% | 80% |

<br/>

---

## 🐍 Python Export

Export your configuration as a **standalone Python script**:

- ✅ Zero dependencies (except NumPy + Matplotlib)
- ✅ Identical implementation to JavaScript
- ✅ Complete training loop + visualization
- ✅ Reproducible results

```python
"""
Neural Network from Scratch - Exported from NN Playground
"""
import numpy as np
import matplotlib.pyplot as plt

LAYERS = [2, 4, 4, 1]
ACTIVATION = "tanh"
OPTIMIZER = "adam"
LEARNING_RATE = 0.03
# ... full implementation included
```

<br/>

---

## 🌐 Browser Support

| Browser | Version | Status |
|:-------:|:-------:|:------:|
| <img src="https://img.shields.io/badge/Chrome-80+-4285F4?logo=googlechrome&logoColor=white"/> | 80+ | ✅ |
| <img src="https://img.shields.io/badge/Firefox-75+-FF7139?logo=firefox&logoColor=white"/> | 75+ | ✅ |
| <img src="https://img.shields.io/badge/Safari-13+-000000?logo=safari&logoColor=white"/> | 13+ | ✅ |
| <img src="https://img.shields.io/badge/Edge-80+-0078D7?logo=microsoftedge&logoColor=white"/> | 80+ | ✅ |
| <img src="https://img.shields.io/badge/IE_11-❌-red"/> | - | ❌ |

<br/>

---

## ⚡ Performance

### Optimizations Implemented

| Optimization | Description |
|--------------|-------------|
| 🎯 **Selective Rendering** | Decision boundary updates every 3rd frame |
| 📐 **Resolution Options** | Low (50²), Medium (75²), High (100²) |
| ✂️ **Gradient Clipping** | Max norm = 5.0 |
| 🔒 **Value Clipping** | Pre-activation clamped to [-50, 50] |
| ⚡ **Lazy Initialization** | Optimizer state created on demand |
| 🖼️ **Canvas Optimization** | Device pixel ratio handling |

### Numerical Stability

- Sigmoid input clipped to [-500, 500]
- Loss differences clipped to [-10, 10]
- NaN fallback returns 10
- All outputs validated for finiteness

<br/>

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

<table>
<tr>
<td align="center">🐛<br/><b>Bug Reports</b></td>
<td align="center">💡<br/><b>Feature Requests</b></td>
<td align="center">🔧<br/><b>Pull Requests</b></td>
<td align="center">📚<br/><b>Documentation</b></td>
</tr>
</table>

```bash
# Fork & Clone
git clone https://github.com/YOUR_USERNAME/neural-network-playground.git

# Start local server
python -m http.server 8000

# Make changes, test, submit PR
```

<br/>

---

## 📄 License

<div align="center">

This project is licensed under the **MIT License**.

```
MIT License • Copyright (c) 2024 Youssef Hajaj
```

See [LICENSE](LICENSE) for details.

</div>

<br/>

---

<div align="center">

### ⭐ Star this repo if you found it useful!

<br/>

**by [Youssef Hajaj](https://www.linkedin.com/in/youssef-hajaj/)**

</div>
