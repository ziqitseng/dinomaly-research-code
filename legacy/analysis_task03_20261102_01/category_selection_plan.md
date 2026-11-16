# Frozen extension category plan

{
  "rule_frozen_before_any_new_per_object_outcome_or_inference": true,
  "basis": "Original native-split category I-AUROC only; weak means strict <0.90; category list is diagnostic, not a paired comparison because model native test lists differ.",
  "selection_rules": {
    "shared_weakness": "Both below 0.90; select the lowest mean of the two I-AUROCs.",
    "D2_only_weakness": "D2 below 0.90 and ArcAD at least 0.90; select the most negative D2-ArcAD delta.",
    "ArcAD_only_weakness": "ArcAD below 0.90 and D2 at least 0.90; select the most positive D2-ArcAD delta.",
    "strong_both": "Both at least 0.90; maximize the weaker model I-AUROC."
  },
  "already_analyzed": [
    "lattice_block_plug",
    "laser_diode",
    "power_jack",
    "LED_indicator",
    "thyristor"
  ],
  "prioritized_additional_categories": {
    "shared_weakness": "blue_light_switch",
    "D2_only_weakness": "inverter_connector",
    "ArcAD_only_weakness": "lilypad_led",
    "strong_both": "steering_T_head"
  },
  "inference_decision": "No new categories inferred in this run. Reuse Task 02 numeric maps first; any expansion requires a separate compute/storage review and prespecified held-out analysis."
}
