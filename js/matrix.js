/**
 * Matrix Operations Library
 * High-performance matrix operations for neural network computations
 * Built from scratch - no dependencies
 */

class Matrix {
    constructor(rows, cols, data = null) {
        this.rows = rows;
        this.cols = cols;

        if (data) {
            this.data = data;
        } else {
            // Initialize with zeros using typed arrays for performance
            this.data = [];
            for (let i = 0; i < rows; i++) {
                this.data.push(new Array(cols).fill(0));
            }
        }
    }

    /**
     * Create matrix from 1D array (column vector)
     */
    static fromArray(arr) {
        const m = new Matrix(arr.length, 1);
        for (let i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    /**
     * Convert to 1D array
     */
    toArray() {
        const arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }

    /**
     * Create a deep copy
     */
    clone() {
        const m = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                m.data[i][j] = this.data[i][j];
            }
        }
        return m;
    }

    /**
     * Xavier/Glorot initialization
     * Good for tanh and sigmoid activations
     */
    xavierInit(fanIn, fanOut) {
        const limit = Math.sqrt(6 / (fanIn + fanOut));
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = (Math.random() * 2 - 1) * limit;
            }
        }
        return this;
    }

    /**
     * He initialization
     * Good for ReLU activations
     */
    heInit(fanIn) {
        const std = Math.sqrt(2 / fanIn);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                // Box-Muller transform for normal distribution
                const u1 = Math.random();
                const u2 = Math.random();
                const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                this.data[i][j] = z * std;
            }
        }
        return this;
    }

    /**
     * Randomize with uniform distribution
     */
    randomize(scale = 1) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = (Math.random() * 2 - 1) * scale;
            }
        }
        return this;
    }

    /**
     * Fill with zeros
     */
    zeros() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = 0;
            }
        }
        return this;
    }

    /**
     * Fill with ones
     */
    ones() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = 1;
            }
        }
        return this;
    }

    /**
     * Matrix multiplication: A × B
     */
    static multiply(a, b) {
        if (a.cols !== b.rows) {
            throw new Error(`Matrix multiplication error: ${a.cols} !== ${b.rows}`);
        }

        const result = new Matrix(a.rows, b.cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    /**
     * Element-wise multiplication (Hadamard product)
     */
    static hadamard(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error('Hadamard product requires same dimensions');
        }

        const result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }
        return result;
    }

    /**
     * Element-wise addition
     */
    static add(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error('Addition requires same dimensions');
        }

        const result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }
        return result;
    }

    /**
     * Element-wise subtraction: A - B
     */
    static subtract(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error('Subtraction requires same dimensions');
        }

        const result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    /**
     * Scalar multiplication (static)
     */
    static scale(m, scalar) {
        const result = new Matrix(m.rows, m.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                result.data[i][j] = m.data[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * In-place scalar multiplication
     */
    scale(scalar) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] *= scalar;
            }
        }
        return this;
    }

    /**
     * In-place addition
     */
    add(m) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] += m.data[i][j];
            }
        }
        return this;
    }

    /**
     * In-place subtraction
     */
    subtract(m) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] -= m.data[i][j];
            }
        }
        return this;
    }

    /**
     * Transpose
     */
    static transpose(m) {
        const result = new Matrix(m.cols, m.rows);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                result.data[j][i] = m.data[i][j];
            }
        }
        return result;
    }

    /**
     * Apply function to each element (returns new matrix)
     */
    map(fn) {
        const result = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = fn(this.data[i][j], i, j);
            }
        }
        return result;
    }

    /**
     * Apply function in place
     */
    mapInPlace(fn) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = fn(this.data[i][j], i, j);
            }
        }
        return this;
    }

    /**
     * Sum of all elements
     */
    sum() {
        let total = 0;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                total += this.data[i][j];
            }
        }
        return total;
    }

    /**
     * Sum of squared elements (for L2 regularization)
     */
    sumSquared() {
        let total = 0;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                total += this.data[i][j] * this.data[i][j];
            }
        }
        return total;
    }

    /**
     * Mean of all elements
     */
    mean() {
        return this.sum() / (this.rows * this.cols);
    }

    /**
     * Max value
     */
    max() {
        let maxVal = -Infinity;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                if (this.data[i][j] > maxVal) {
                    maxVal = this.data[i][j];
                }
            }
        }
        return maxVal;
    }

    /**
     * Min value
     */
    min() {
        let minVal = Infinity;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                if (this.data[i][j] < minVal) {
                    minVal = this.data[i][j];
                }
            }
        }
        return minVal;
    }

    /**
     * Get value at position
     */
    get(row, col) {
        return this.data[row][col];
    }

    /**
     * Set value at position
     */
    set(row, col, value) {
        this.data[row][col] = value;
    }

    /**
     * Create identity matrix
     */
    static identity(size) {
        const m = new Matrix(size, size);
        for (let i = 0; i < size; i++) {
            m.data[i][i] = 1;
        }
        return m;
    }

    /**
     * Apply dropout mask (for regularization)
     */
    applyDropout(rate) {
        if (rate <= 0) return this.clone();

        const scale = 1 / (1 - rate);
        const result = new Matrix(this.rows, this.cols);

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                if (Math.random() > rate) {
                    result.data[i][j] = this.data[i][j] * scale;
                } else {
                    result.data[i][j] = 0;
                }
            }
        }

        return result;
    }

    /**
     * Clip values to range
     */
    clip(min, max) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.max(min, Math.min(max, this.data[i][j]));
            }
        }
        return this;
    }

    /**
     * Debug print
     */
    print(name = 'Matrix') {
        console.log(`${name} [${this.rows}×${this.cols}]:`);
        console.table(this.data);
    }
}

// Export
window.Matrix = Matrix;
