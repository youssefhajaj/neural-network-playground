/**
 * Utility Functions
 * Helper functions used throughout the application
 */

const Utils = {
    /**
     * Clamp a value between min and max
     */
    clamp(value, min, max) {
        return Math.min(Math.max(value, min), max);
    },

    /**
     * Linear interpolation
     */
    lerp(a, b, t) {
        return a + (b - a) * t;
    },

    /**
     * Map a value from one range to another
     */
    map(value, inMin, inMax, outMin, outMax) {
        return ((value - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin;
    },

    /**
     * Debounce function
     */
    debounce(fn, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn.apply(this, args), delay);
        };
    },

    /**
     * Throttle function
     */
    throttle(fn, limit) {
        let inThrottle;
        return function (...args) {
            if (!inThrottle) {
                fn.apply(this, args);
                inThrottle = true;
                setTimeout(() => (inThrottle = false), limit);
            }
        };
    },

    /**
     * Format number with specified decimal places
     */
    formatNumber(num, decimals = 4) {
        if (num === null || num === undefined || isNaN(num)) return '—';
        return num.toFixed(decimals);
    },

    /**
     * Format percentage
     */
    formatPercent(num, decimals = 1) {
        if (num === null || num === undefined || isNaN(num)) return '—';
        return (num * 100).toFixed(decimals) + '%';
    },

    /**
     * Generate unique ID
     */
    generateId() {
        return Math.random().toString(36).substr(2, 9);
    },

    /**
     * Deep clone object
     */
    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },

    /**
     * Shuffle array (Fisher-Yates)
     */
    shuffle(array) {
        const result = [...array];
        for (let i = result.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [result[i], result[j]] = [result[j], result[i]];
        }
        return result;
    },

    /**
     * Get color for value (blue to red gradient)
     */
    getValueColor(value, alpha = 1) {
        // Clamp value between -1 and 1
        const v = this.clamp(value, -1, 1);

        if (v >= 0) {
            // Positive: blue (indigo)
            const intensity = v;
            return `rgba(99, 102, 241, ${intensity * alpha})`;
        } else {
            // Negative: red
            const intensity = -v;
            return `rgba(239, 68, 68, ${intensity * alpha})`;
        }
    },

    /**
     * Get gradient color for heatmap
     */
    getHeatmapColor(value) {
        const v = this.clamp(value, -1, 1);
        const normalized = (v + 1) / 2; // 0 to 1

        // Red -> Gray -> Blue
        if (normalized < 0.5) {
            const t = normalized * 2;
            return {
                r: Math.round(239 * (1 - t) + 30 * t),
                g: Math.round(68 * (1 - t) + 41 * t),
                b: Math.round(68 * (1 - t) + 59 * t)
            };
        } else {
            const t = (normalized - 0.5) * 2;
            return {
                r: Math.round(30 * (1 - t) + 99 * t),
                g: Math.round(41 * (1 - t) + 102 * t),
                b: Math.round(59 * (1 - t) + 241 * t)
            };
        }
    },

    /**
     * Create canvas with proper DPI scaling
     */
    setupCanvas(canvas) {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;

        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        return {
            ctx,
            width: rect.width,
            height: rect.height,
            dpr
        };
    },

    /**
     * Show toast notification
     */
    showToast(message, type = 'info', duration = 3000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span>${message}</span>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },

    /**
     * Gaussian random using Box-Muller transform
     */
    gaussianRandom(mean = 0, std = 1) {
        const u1 = Math.random();
        const u2 = Math.random();
        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return z0 * std + mean;
    },

    /**
     * Calculate statistics for an array
     */
    statistics(arr) {
        if (!arr || arr.length === 0) return null;

        const n = arr.length;
        const mean = arr.reduce((a, b) => a + b, 0) / n;
        const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
        const std = Math.sqrt(variance);
        const min = Math.min(...arr);
        const max = Math.max(...arr);

        return { mean, std, variance, min, max, n };
    },

    /**
     * Smooth array using moving average
     */
    movingAverage(arr, windowSize = 5) {
        const result = [];
        for (let i = 0; i < arr.length; i++) {
            const start = Math.max(0, i - windowSize + 1);
            const slice = arr.slice(start, i + 1);
            result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
        }
        return result;
    }
};

// Export
window.Utils = Utils;
