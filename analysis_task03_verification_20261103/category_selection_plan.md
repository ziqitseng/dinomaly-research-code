# Precommitted Task 03 extension selection

The existing diagnostic set is the actual current Task 02 set: lattice_block_plug, laser_diode, power_jack, LED_indicator, aircraft_model_head. The Task 03 brief’s prior-run list names `thyristor` instead of the current `aircraft_model_head`; this mismatch is recorded, and the current saved maps are the basis for this follow-up.

Before any new maps or per-object outcomes are examined, extensions are selected only from the 160-category native summary scores. Selection rules are fixed at I-AUROC <0.90 for weakness and >=0.95 for the strong comparison. Current Task 02 categories are excluded from extension candidates.

- **shared_weakness — `blue_light_switch`:** lowest minimum of both model I-AUROCs among categories where both are <0.90.
- **D2_only_weakness — `rotary_position_sensor`:** lowest D2 I-AUROC where D2<0.90 and ArcAD>=0.90.
- **ArcAD_only_weakness — `lilypad_led`:** lowest ArcAD protoinit I-AUROC where ArcAD<0.90 and D2>=0.90.
- **strong_performance — `thyristor`:** highest minimum I-AUROC among categories where both are >=0.95.

This is a diagnostic, score-ranked selection plan, not a representative sample. No extension inference is launched in Task 03 unless later evidence shows saved numeric maps are insufficient and a separate compute plan is approved.
