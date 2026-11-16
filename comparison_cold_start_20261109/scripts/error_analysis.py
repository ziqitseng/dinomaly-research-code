"""Protocol B error accounting, with thresholds fitted exclusively to source OK data."""
import math


def fit_normal_threshold(records, target_fpr):
    if not records or not 0 < target_fpr < 1:
        raise ValueError('Need nonempty true-normal calibration data and a valid FPR')
    for r in records:
        if (r['object_label'] != 0 or r['is_ng_without_mask']
                or r['partition'] != 'normal_calibration' or r['source_split'] != 'train'
                or r.get('quality_flags')):
            raise ValueError('Threshold calibration accepts reviewed source OK/train calibration views only')
    values = sorted(float(r['score']) for r in records)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite calibration score')
    index = math.ceil((1 - target_fpr) * (len(values) - 1))
    return dict(threshold=values[index], calibration_views=len(values), target_fpr=target_fpr,
                quantile='higher empirical quantile', alarm_rule='score > threshold',
                observed_calibration_fpr=sum(value > values[index] for value in values) / len(values))


def error_bucket(record):
    if record['is_ng_without_mask']:
        if record['object_label'] != 1 or record['protocol_b_image_label'] != 0:
            raise ValueError('Invalid Protocol B missing-mask target')
        return 'ng_without_mask'
    return 'true_normal' if record['object_label'] == 0 else 'ng_with_mask'


def analyze_errors(records, thresholds):
    scored, by_category = [], {}
    for r in records:
        threshold = thresholds[r['category']]['threshold']
        row = dict(r, error_bucket=error_bucket(r), threshold=threshold, alarm=r['score'] > threshold)
        row['protocol_b_false_alarm'] = row['protocol_b_image_label'] == 0 and row['alarm']
        scored.append(row)
        by_category.setdefault(r['category'], []).append(row)

    def summarize(values):
        normal = [r for r in values if r['error_bucket'] == 'true_normal']
        missing = [r for r in values if r['error_bucket'] == 'ng_without_mask']
        masked = [r for r in values if r['error_bucket'] == 'ng_with_mask']
        normal_fp, missing_fp = sum(r['alarm'] for r in normal), sum(r['alarm'] for r in missing)
        total_fp = normal_fp + missing_fp
        return dict(true_normal_views=len(normal), true_normal_false_alarms=normal_fp,
                    true_normal_fpr=normal_fp / len(normal) if normal else None,
                    ng_without_mask_views=len(missing), ng_without_mask_alarms=missing_fp,
                    ng_without_mask_alarm_rate=missing_fp / len(missing) if missing else None,
                    protocol_b_false_alarms=total_fp,
                    ng_without_mask_share_of_protocol_b_false_alarms=missing_fp / total_fp if total_fp else None,
                    ng_with_mask_views=len(masked), ng_with_mask_misses=sum(not r['alarm'] for r in masked))
    return scored, dict(overall=summarize(scored),
                        per_category={c: summarize(rs) for c, rs in sorted(by_category.items())},
                        interpretation='NG/no-mask alarms count as Protocol B false alarms, but remain annotation-ambiguous; review before attributing them to model failure. Thresholds use no source NG or test images.')
