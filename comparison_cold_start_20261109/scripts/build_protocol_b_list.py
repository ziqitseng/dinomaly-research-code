"""Setup only: index audited source-test NG/null-mask rows; never decode images."""
import re
from pathlib import Path
from common import config, group_id, rows, write_csv


def main():
    cfg = config()
    target = Path(cfg['protocol_b_null_mask_list'])
    if target.exists():
        raise RuntimeError('Protocol B list already exists; never silently replace it')
    result = []
    for r in rows(Path(cfg['audit_root']) / 'manifest_inventory.csv'):
        if r['anomaly_class'] != 'OK' and not r['actual_mask']:
            if r['split'] != 'test' or r['declared_mask']:
                raise RuntimeError('Unexpected source membership/null-mask semantics')
            result.append(dict(id=r['declared_image'], raw_image=r['actual_image'], category=r['category'],
                group=group_id(r['declared_image']),
                camera=int(re.search(r'_C0?([1-5])_', Path(r['actual_image']).name).group(1)),
                source_split=r['split'], source_class=r['anomaly_class']))
    if len(result) != cfg['expected_ng_without_masks']:
        raise RuntimeError('Audited NG/null-mask count changed')
    result.sort(key=lambda r: r['id'])
    write_csv(target, result, list(result[0]))
    print('Wrote %d Protocol B IDs from audit metadata only; no masks or model outputs generated.' % len(result))


if __name__ == '__main__':
    main()
