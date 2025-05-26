import numpy as np

def line_fit(p1, p2):
    """Fit a line given two points (p1, p2)."""
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return None  # Avoid division by zero
    slope = (y2 - y1) / (x2 - x1)
    if slope < 0:
        return None # Avoid negative slope lines
    intercept = y1 - slope * x1
    return (slope, intercept)


def dist_perp(line, point):
    """Compute perpendicular distance from a point to a line."""
    if line is None:
        return float('inf')
    slope, intercept = line
    x, y = point
    return abs(slope * x - y + intercept) / np.sqrt(slope ** 2 + 1)

def ransac_multiple_lines(x, y, delta, num_hypotheses=100, min_inlier_threshold=100, min_samples=100, max_iter=10):
    """Detect multiple lines using RANSAC."""
    data = np.array(list(zip(x, y)))
    detected_lines = []
    iteration = 0

    while len(data) > min_samples and iteration < max_iter:
        indices = np.arange(len(data))
        best_line = None
        best_inlier_indices = np.array([], dtype=int)
        max_inliers = 0

        for _ in range(num_hypotheses):
            i, j = np.random.choice(indices, 2, replace=False)
            s_i, s_j = data[i], data[j]
            line = line_fit(s_i, s_j)
            dists = np.array([dist_perp(line, data[k]) for k in indices])
            inlier_indices = indices[dists <= delta]

            if len(inlier_indices) > max_inliers:
                best_line = line
                best_inlier_indices = inlier_indices
                max_inliers = len(inlier_indices)

        if best_line is not None and max_inliers >= min_inlier_threshold:
            detected_lines.append(best_line)
            mask = np.ones(len(data), dtype=bool)
            mask[best_inlier_indices] = False
            data = data[mask]

        iteration += 1

    return detected_lines
