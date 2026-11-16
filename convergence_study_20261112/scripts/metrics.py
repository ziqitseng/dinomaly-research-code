"""One metric implementation for both methods. Pixel metrics use fixed histograms."""
import numpy as np


def binary_metrics(labels, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(set(labels)) != 2:
        return {'auroc': None, 'ap': None}
    return {'auroc': float(roc_auc_score(labels, scores)),
            'ap': float(average_precision_score(labels, scores))}


def top_score(array, fraction):
    flat = np.asarray(array).reshape(-1)
    count = max(1, int(flat.size * fraction))
    return float(np.partition(flat, flat.size - count)[-count:].mean(dtype=np.float64))


def integrate_pro(fpr, pro, limit=0.3):
    """Integrate the attainable curve, including an interpolated FPR-limit endpoint."""
    # Preserve vertical segments at repeated FPRs; they have zero width.
    fpr, pro = np.asarray(fpr, dtype=np.float64), np.asarray(pro, dtype=np.float64)
    below = fpr <= limit
    xs, ys = fpr[below].tolist(), pro[below].tolist()
    if not xs:
        xs, ys = [0.0], [0.0]
    if xs[-1] < limit:
        right = np.searchsorted(fpr, limit, side='right')
        if right < len(fpr):
            left = right - 1
            value = pro[left] + (pro[right] - pro[left]) * (limit - fpr[left]) / (fpr[right] - fpr[left])
        else:
            value = pro[-1]
        xs.append(limit)
        ys.append(float(value))
    return float(np.trapz(ys, xs) / limit)


class PixelMetrics:
    """Fixed [0,2] bins; exact counts, approximate threshold integration.

    AUPRO averages recall per 8-connected foreground region, integrates FPR<=0.3,
    and includes eligible normal images in the background denominator.
    """
    def __init__(self, bins):
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)
        self.region_recall_sum = np.zeros(bins, dtype=np.float64)
        self.regions = 0
        self.views = 0

    def update(self, mask, scores):
        from scipy.ndimage import label
        mask = np.asarray(mask, dtype=bool)
        scores = np.asarray(scores)
        if mask.shape != scores.shape or not np.isfinite(scores).all():
            raise ValueError('Invalid map/mask')
        if scores.min() < -1e-5 or scores.max() > 2 + 1e-5:
            raise ValueError('Cosine map outside declared score range [0,2]')
        indices = np.minimum((np.clip(scores, 0, 2) * (self.bins / 2)).astype(np.int64), self.bins - 1)
        self.positive += np.bincount(indices[mask], minlength=self.bins)
        self.negative += np.bincount(indices[~mask], minlength=self.bins)
        regions, count = label(mask, structure=np.ones((3, 3)))
        for region in range(1, count + 1):
            values = indices[regions == region]
            hist = np.bincount(values, minlength=self.bins)
            self.region_recall_sum += hist[::-1].cumsum()[::-1] / len(values)
        self.regions += count
        self.views += 1

    def summary(self):
        positives, negatives = self.positive.sum(), self.negative.sum()
        if not positives or not negatives:
            return dict(P_AUROC=None, P_AP=None, P_AUPRO=None, pixel_views=self.views,
                        foreground_pixels=int(positives), background_pixels=int(negatives), regions=self.regions)
        tp = np.r_[0, self.positive[::-1].cumsum()].astype(np.float64)
        fp = np.r_[0, self.negative[::-1].cumsum()].astype(np.float64)
        recall, fpr = tp / positives, fp / negatives
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) != 0)
        pro = np.r_[0, self.region_recall_sum[::-1] / self.regions]
        return dict(P_AUROC=float(np.trapz(recall, fpr)),
                    P_AP=float(np.sum(np.diff(recall) * precision[1:])),
                    P_AUPRO=integrate_pro(fpr, pro), pixel_views=self.views,
                    foreground_pixels=int(positives), background_pixels=int(negatives), regions=self.regions)

    def save(self, path):
        np.savez_compressed(path, positive=self.positive, negative=self.negative,
                            region_recall_sum=self.region_recall_sum, regions=self.regions, views=self.views,
                            bins=self.bins, score_min=0., score_max=2.)
