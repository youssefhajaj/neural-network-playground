/**
 * Dataset Generators
 * Creates various 2D classification datasets for training neural networks
 */

const Datasets = {
    /**
     * XOR Pattern
     * Classic non-linearly separable problem
     */
    xor(numPoints, noise = 0.1) {
        const data = [];
        const labels = [];
        const perQuadrant = Math.floor(numPoints / 4);

        const quadrants = [
            { cx: 0.5, cy: 0.5, label: 1 },   // Top-right
            { cx: -0.5, cy: 0.5, label: 0 },  // Top-left
            { cx: -0.5, cy: -0.5, label: 1 }, // Bottom-left
            { cx: 0.5, cy: -0.5, label: 0 }   // Bottom-right
        ];

        for (const q of quadrants) {
            for (let i = 0; i < perQuadrant; i++) {
                const x = q.cx + (Math.random() - 0.5) * 0.8 + (Math.random() - 0.5) * noise;
                const y = q.cy + (Math.random() - 0.5) * 0.8 + (Math.random() - 0.5) * noise;
                data.push([x, y]);
                labels.push([q.label]);
            }
        }

        return Datasets._shuffle({ data, labels });
    },

    /**
     * Circle Pattern
     * Points inside vs outside a circle
     */
    circle(numPoints, noise = 0.1) {
        const data = [];
        const labels = [];
        const radius = 0.5;

        for (let i = 0; i < numPoints; i++) {
            const angle = Math.random() * Math.PI * 2;
            const isInside = Math.random() > 0.5;

            let r;
            if (isInside) {
                r = Math.random() * radius * 0.7;
            } else {
                r = radius + 0.15 + Math.random() * 0.35;
            }

            const x = r * Math.cos(angle) + (Math.random() - 0.5) * noise;
            const y = r * Math.sin(angle) + (Math.random() - 0.5) * noise;

            data.push([x, y]);
            labels.push([isInside ? 1 : 0]);
        }

        return Datasets._shuffle({ data, labels });
    },

    /**
     * Spiral Pattern
     * Two interleaving spirals - hardest to separate
     */
    spiral(numPoints, noise = 0.1) {
        const data = [];
        const labels = [];
        const perClass = Math.floor(numPoints / 2);

        for (let classIdx = 0; classIdx < 2; classIdx++) {
            for (let i = 0; i < perClass; i++) {
                const t = i / perClass;
                const r = t * 0.8;
                const angle = t * Math.PI * 2.5 + classIdx * Math.PI;

                const x = r * Math.cos(angle) + (Math.random() - 0.5) * noise;
                const y = r * Math.sin(angle) + (Math.random() - 0.5) * noise;

                data.push([x, y]);
                labels.push([classIdx]);
            }
        }

        return Datasets._shuffle({ data, labels });
    },

    /**
     * Two Moons Pattern
     * Two interleaving crescents
     */
    moons(numPoints, noise = 0.1) {
        const data = [];
        const labels = [];
        const perClass = Math.floor(numPoints / 2);

        // Top moon
        for (let i = 0; i < perClass; i++) {
            const angle = Math.PI * i / perClass;
            const x = Math.cos(angle) * 0.5 + (Math.random() - 0.5) * noise;
            const y = Math.sin(angle) * 0.5 + (Math.random() - 0.5) * noise;
            data.push([x, y]);
            labels.push([0]);
        }

        // Bottom moon (offset)
        for (let i = 0; i < perClass; i++) {
            const angle = Math.PI + Math.PI * i / perClass;
            const x = 0.25 + Math.cos(angle) * 0.5 + (Math.random() - 0.5) * noise;
            const y = -0.25 + Math.sin(angle) * 0.5 + (Math.random() - 0.5) * noise;
            data.push([x, y]);
            labels.push([1]);
        }

        return Datasets._shuffle({ data, labels });
    },

    /**
     * Gaussian Clusters
     * Two Gaussian blobs
     */
    clusters(numPoints, noise = 0.1) {
        const data = [];
        const labels = [];
        const perClass = Math.floor(numPoints / 2);

        const centers = [
            { x: -0.4, y: -0.4 },
            { x: 0.4, y: 0.4 }
        ];

        for (let classIdx = 0; classIdx < 2; classIdx++) {
            for (let i = 0; i < perClass; i++) {
                // Box-Muller for Gaussian
                const u1 = Math.random();
                const u2 = Math.random();
                const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

                const spread = 0.2 + noise * 0.5;
                const x = centers[classIdx].x + z0 * spread;
                const y = centers[classIdx].y + z1 * spread;

                data.push([x, y]);
                labels.push([classIdx]);
            }
        }

        return Datasets._shuffle({ data, labels });
    },

    /**
     * Concentric Rings
     * Multiple rings around center
     */
    rings(numPoints, noise = 0.1) {
        const data = [];
        const labels = [];
        const perRing = Math.floor(numPoints / 3);

        const rings = [
            { r: 0.2, label: 0 },
            { r: 0.5, label: 1 },
            { r: 0.8, label: 0 }
        ];

        for (const ring of rings) {
            for (let i = 0; i < perRing; i++) {
                const angle = Math.random() * Math.PI * 2;
                const rNoise = ring.r + (Math.random() - 0.5) * 0.1 * (1 + noise);

                const x = rNoise * Math.cos(angle) + (Math.random() - 0.5) * noise * 0.5;
                const y = rNoise * Math.sin(angle) + (Math.random() - 0.5) * noise * 0.5;

                data.push([x, y]);
                labels.push([ring.label]);
            }
        }

        return Datasets._shuffle({ data, labels });
    },

    /**
     * Checkerboard Pattern
     * 2x2 checkerboard
     */
    checkerboard(numPoints, noise = 0.1) {
        const data = [];
        const labels = [];
        const perCell = Math.floor(numPoints / 4);

        const cells = [
            { cx: -0.5, cy: 0.5, label: 0 },
            { cx: 0.5, cy: 0.5, label: 1 },
            { cx: -0.5, cy: -0.5, label: 1 },
            { cx: 0.5, cy: -0.5, label: 0 }
        ];

        for (const cell of cells) {
            for (let i = 0; i < perCell; i++) {
                const x = cell.cx + (Math.random() - 0.5) * 0.9 + (Math.random() - 0.5) * noise;
                const y = cell.cy + (Math.random() - 0.5) * 0.9 + (Math.random() - 0.5) * noise;
                data.push([x, y]);
                labels.push([cell.label]);
            }
        }

        return Datasets._shuffle({ data, labels });
    },

    /**
     * Split dataset into train/test
     */
    split(dataset, trainRatio = 0.8) {
        const n = dataset.data.length;
        const trainSize = Math.floor(n * trainRatio);

        return {
            train: {
                data: dataset.data.slice(0, trainSize),
                labels: dataset.labels.slice(0, trainSize)
            },
            test: {
                data: dataset.data.slice(trainSize),
                labels: dataset.labels.slice(trainSize)
            }
        };
    },

    /**
     * Shuffle dataset (Fisher-Yates)
     */
    _shuffle(dataset) {
        const n = dataset.data.length;
        const indices = Array.from({ length: n }, (_, i) => i);

        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }

        return {
            data: indices.map(i => dataset.data[i]),
            labels: indices.map(i => dataset.labels[i])
        };
    },

    /**
     * Get all dataset names
     */
    getTypes() {
        return ['xor', 'circle', 'spiral', 'moons', 'clusters', 'rings'];
    }
};

// Export
window.Datasets = Datasets;
