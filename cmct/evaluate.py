"""Test-set accuracy for both teachers and their ensemble."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate(teacher_lora, mlp_model, teacher_classifier, test_loader, device):
    """Both teachers look at the exact same image each batch.

    ONE shared dassl test loader feeds the LoRA teacher and the CMKD teacher
    (`mlp_model.teacher_model`'s EMA backbone features through
    `teacher_classifier`, its own EMA head), so the ensemble is a fair average.

    `teacher_lora` may be None when `branch_lora.enabled` is False: there is no
    LoRA teacher and no ensemble to report, so those two accuracies come back
    as None rather than as a fabricated number.
    """
    correct_lora, correct_mlp, correct_ens, total = 0, 0, 0, 0
    for batch in test_loader:
        image = batch["img"].to(device)
        label = batch["label"].to(device)

        if teacher_lora is not None:
            logits_lora, _ = teacher_lora(image)
            prob_lora = F.softmax(logits_lora, dim=-1)

        feat_mlp_teacher = mlp_model.teacher_model.forward_features(image)
        logits_mlp = teacher_classifier(feat_mlp_teacher)
        prob_mlp = F.softmax(logits_mlp, dim=-1)

        correct_mlp += (prob_mlp.argmax(dim=-1) == label).sum().item()
        if teacher_lora is not None:
            prob_ens = 0.5 * (prob_lora + prob_mlp)

            correct_lora += (prob_lora.argmax(dim=-1) == label).sum().item()
            correct_ens += (prob_ens.argmax(dim=-1) == label).sum().item()

        total += label.size(0)

    acc_mlp = 100.0 * correct_mlp / max(total, 1)
    if teacher_lora is None:
        return None, acc_mlp, None
    return (
        100.0 * correct_lora / max(total, 1),
        acc_mlp,
        100.0 * correct_ens / max(total, 1),
    )
