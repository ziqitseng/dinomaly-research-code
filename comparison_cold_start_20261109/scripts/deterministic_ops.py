"""Equivalent ArcAD DGC selection without CUDA gather backward.

Keep strict deterministic algorithms enabled. Candidate choice is discrete in
the original method: search without autograd, then rebuild only the chosen
normalized candidate with gradients to the anchor. RNG draw shape/order, scoring,
candidate count, prototypes, and branch derivative are unchanged.
"""


def augment_features(learner, anchors):
    import torch
    import torch.nn.functional as F
    candidates_per_anchor = 50  # Matches ArcAD's native BottleneckPrototypeLearner.
    count, width = anchors.shape
    with torch.no_grad():
        expanded = anchors.unsqueeze(1).expand(count, candidates_per_anchor, width)
        noise = torch.randn_like(expanded)
        candidates = F.normalize(expanded + learner.noise_std * noise, dim=2)
        prototypes = F.normalize(learner.prototypes.data, dim=1)
        nearest = torch.mm(candidates.reshape(count * candidates_per_anchor, width), prototypes.t()).max(dim=1)[0]
        selected = nearest.reshape(count, candidates_per_anchor).min(dim=1)[1]
        indices = selected.reshape(count, 1, 1).expand(count, 1, width)
        # Noise does not require gradients, so this gather has no scatter-add backward.
        selected_noise = torch.gather(noise, 1, indices).squeeze(1)
        del noise, candidates, expanded, nearest, selected, indices
    # Only the selected branch contributes to the original gather's derivative.
    return F.normalize(anchors + learner.noise_std * selected_noise, dim=1)
