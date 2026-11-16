#!/usr/bin/env python3
"""Run the paired ArcAD export and D2 extension export sequentially."""
import subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
exporter=here/'export_predictions.py'
cmds=[
 [sys.executable,str(exporter),'--model','arcad','--batch-size','8'],
 [sys.executable,str(exporter),'--model','d2','--categories','LED_indicator','thyristor','--batch-size','16'],
]
for cmd in cmds:
 print('Running:', ' '.join(cmd),flush=True)
 subprocess.run(cmd,check=True)
