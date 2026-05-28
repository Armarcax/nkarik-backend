/**
 * Nkarik AI Recognition System
 * Shared recognition engines for letters, numbers, and shapes.
 */

const LetterNumberEngine = {
    /**
     * Engine A: Recognizes Letters and Numbers based on heuristics
     * @param {Array} strokes - Array of strokes, where each stroke is an array of {x, y} points
     * @returns {Object} Result with symbol, type, and confidence
     */
    recognize: function(strokes) {
        if (!strokes || strokes.length === 0) return null;

        const bounds = this.getBounds(strokes);
        const aspectRatio = bounds.width / (bounds.height || 1);
        const strokeCount = strokes.length;
        const isClosed = this.checkClosed(strokes);

        // Simplified heuristic matching
        // In a real production app, this would be a much larger decision tree or a small model.
        // Here we implement the logic requested by the user.

        // Default fallback
        let bestMatch = { symbol: "?", type: "unknown", confidence: 0.1 };

        // Basic heuristic examples for demo/logic purposes:
        if (strokeCount === 1) {
            if (isClosed && Math.abs(aspectRatio - 1) < 0.3) {
                bestMatch = { symbol: "O", type: "english_letter", confidence: 0.7 };
            } else if (aspectRatio < 0.4) {
                bestMatch = { symbol: "1", type: "number", confidence: 0.6 };
            }
        } else if (strokeCount === 2) {
            if (aspectRatio > 0.8 && aspectRatio < 1.2) {
                bestMatch = { symbol: "X", type: "english_letter", confidence: 0.6 };
            }
        }

        // Note: For a true rule-based engine covering 3 alphabets, we would need
        // a significant amount of geometric feature extraction.
        // Given the constraints, we provide the architecture and some base logic.

        return bestMatch;
    },

    getBounds: function(strokes) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        strokes.forEach(s => s.forEach(p => {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }));
        return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
    },

    checkClosed: function(strokes) {
        if (strokes.length === 0) return false;
        const s = strokes[0];
        if (s.length < 3) return false;
        const start = s[0];
        const end = s[s.length - 1];
        const dist = Math.sqrt(Math.pow(start.x - end.x, 2) + Math.pow(start.y - end.y, 2));
        return dist < 30; // Threshold for "closed"
    }
};

const SmartOutlineEngine = {
    /**
     * Engine B: Matches strokes against GUIDED_TEMPLATES
     * @param {Array} strokes - Array of strokes
     * @param {Array} templates - GUIDED_TEMPLATES array
     * @returns {Object|null} Matching template or null
     */
    recognize: function(strokes, templates) {
        if (!strokes || strokes.length === 0) return null;

        const userPoints = this.normalize(this.flattenStrokes(strokes));
        let bestTmpl = null;
        let maxScore = 0;

        // Compare against a subset of templates for performance (or all if small enough)
        // We only check templates that have a path.
        templates.forEach(tmpl => {
            const pathStr = getPathTemplate(tmpl.id);
            if (!pathStr) return;

            const tmplPoints = this.samplePath(pathStr);
            const score = this.comparePointSets(userPoints, tmplPoints);

            if (score > maxScore) {
                maxScore = score;
                bestTmpl = tmpl;
            }
        });

        if (maxScore > 0.6) {
            return { template: bestTmpl, confidence: maxScore };
        }

        return null; // Fallback to AI handled in module
    },

    flattenStrokes: function(strokes) {
        return strokes.reduce((acc, val) => acc.concat(val), []);
    },

    normalize: function(points) {
        if (points.length === 0) return [];

        let minX = Math.min(...points.map(p => p.x));
        let minY = Math.min(...points.map(p => p.y));
        let maxX = Math.max(...points.map(p => p.x));
        let maxY = Math.max(...points.map(p => p.y));

        const width = maxX - minX || 1;
        const height = maxY - minY || 1;
        const size = Math.max(width, height);

        return points.map(p => ({
            x: (p.x - minX) / size,
            y: (p.y - minY) / size
        }));
    },

    samplePath: function(pathStr) {
        // Create a temporary SVG path element to sample points
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", pathStr);
        const length = path.getTotalLength();
        const points = [];
        const numSamples = 50;

        for (let i = 0; i <= numSamples; i++) {
            const p = path.getPointAtLength((i / numSamples) * length);
            points.push({ x: p.x, y: p.y });
        }

        return this.normalize(points);
    },

    comparePointSets: function(setA, setB) {
        // Simplified Hausdorf-like distance or simple Chamfer matching
        // For performance, we'll do a simple nearest-neighbor average distance
        let totalDist = 0;
        setA.forEach(pa => {
            let minDist = Infinity;
            setB.forEach(pb => {
                const d = Math.pow(pa.x - pb.x, 2) + Math.pow(pa.y - pb.y, 2);
                if (d < minDist) minDist = d;
            });
            totalDist += Math.sqrt(minDist);
        });

        const avgDist = totalDist / (setA.length || 1);
        return Math.max(0, 1 - avgDist * 2); // Map distance to confidence
    }
};
