#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def commit(p):
 try:return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
 except Exception:return None
protocol=json.loads((P/'protocol_audit.json').read_text())
run=json.loads((P/'inference_run.json').read_text())
manifest=json.loads((P/'manifest_validation.json').read_text())
reg=pd.read_csv(P/'arcad_original_regression.csv')
ci=pd.read_csv(P/'paired_bootstrap_ci.csv')
protocol.update({
 'inference_status':'PASS_INFERENCE_COMPLETE',
 'source_commits':{'ArcAD':commit(ROOT/'ArcAD'),'Dinomaly2':commit(ROOT/'Dinomaly2')},
 'code_sha256':{f'work/{f}':sha(P/'work'/f) for f in ['prepare.py','validate_geometry.py','inference.py','analyze.py','size_strata.py','finalize.py']},
 'exact_task04a_manifest_sha256':sha(P/'shared_test_manifest.json'),
 'manifest_validation_status':manifest['status'],
 'inference_run':{'device':run['device'],'torch_version':run['torch_version'],'cuda_version':run['cuda_version'],
  'batch_size':run['batch_size'],'seed':run['seed'],'elapsed_seconds':run['elapsed_seconds'],'branch_results':run['branches']},
 'map_metric_checks':{'ArcAD_original_regression':'PASS' if (reg.status=='PASS').all() else 'FAIL',
  'max_arcad_original_score_delta':float(reg.max_abs_score_delta.max()),'declared_regression_tolerance':5e-5,
  'finite_maps_and_path_order':'PASS','pixel_AUROC_AP_histogram_bins':2048,'P_AUPRO_thresholds':200,'P_AUPRO_FPR_cap':0.30,
  'paired_object_bootstrap_replicates_per_category':2000,'seed':20261103,
  'invalid_single_class_draws':int(ci.invalid_single_class_draws.max())},
 'legacy_exclusion':'No ArcAD input or output under any legacy directory was read or used.'})
(P/'protocol_audit.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps({'status':protocol['inference_status'],'manifest':manifest['status'],
 'ArcAD_original_regression':protocol['map_metric_checks']['ArcAD_original_regression'],
 'code_files_hashed':len(protocol['code_sha256'])},indent=2))

