"""Test-set accuracy for both teachers and their ensemble."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate(teacher_lora, mlp_model, teacher_classifier, test_loader, device):
    """Both teachers look at the exact same image each batch.

    ONE shared dassl test loader feeds the LoRA teacher and the CMKD teacher
    (`mlp_model.teacher_model`'s EMA backbone features through
    `teacher_classifier`, its own EMA head), so the ensemble is a fair average.

    Either side may be absent: `teacher_lora` is None when
    `branch_lora.enabled` is False, and `mlp_model`/`teacher_classifier` are
    None when `branch_mlp.enabled` is False. A missing branch's accuracy comes
    back as None rather than as a fabricated number, and so does the ensemble
    unless BOTH branches are there -- an "ensemble" of one model is just that
    model, and reporting it under a second name would invite reading the two
    numbers as independent evidence. Both being absent is rejected in
    config._validate, so at least one accuracy is always real.
    """
    lora_on = teacher_lora is not None
    mlp_on = mlp_model is not None
    correct_lora, correct_mlp, correct_ens, total = 0, 0, 0, 0
    for batch in test_loader:
        image = batch["img"].to(device)
        label = batch["label"].to(device)

        if lora_on:
            logits_lora, _ = teacher_lora(image)
            prob_lora = F.softmax(logits_lora, dim=-1)
            correct_lora += (prob_lora.argmax(dim=-1) == label).sum().item()

        if mlp_on:
            feat_mlp_teacher = mlp_model.teacher_model.forward_features(image)
            logits_mlp = teacher_classifier(feat_mlp_teacher)
            prob_mlp = F.softmax(logits_mlp, dim=-1)
            correct_mlp += (prob_mlp.argmax(dim=-1) == label).sum().item()

        if lora_on and mlp_on:
            prob_ens = 0.5 * (prob_lora + prob_mlp)
            correct_ens += (prob_ens.argmax(dim=-1) == label).sum().item()

        total += label.size(0)

    denom = max(total, 1)
    return (
        100.0 * correct_lora / denom if lora_on else None,
        100.0 * correct_mlp / denom if mlp_on else None,
        100.0 * correct_ens / denom if lora_on and mlp_on else None,
    )
