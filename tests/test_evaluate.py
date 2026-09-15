import torch

from cmct.evaluate import evaluate


class _Teacher:
    """Right only on the normalization it was pretrained for.

    `marker` is the pixel value standing in for that normalization, and the
    teacher predicts `pred_class` when it sees it and class 0 when it does not
    -- so handing it the other loader's batch turns every prediction wrong.
    """

    def __init__(self, marker, pred_class):
        self.marker = marker
        self.pred_class = pred_class

    def _logits(self, image):
        seen_own_normalization = image[:, 0, 0, 0] == self.marker
        logits = torch.zeros(image.shape[0], 3)
        logits[seen_own_normalization, self.pred_class] = 10.0
        logits[~seen_own_normalization, 0] = 10.0
        return logits


class _LoraTeacher(_Teacher):
    def __call__(self, image):
        return self._logits(image), None


class _MlpBackbone(_Teacher):
    def forward_features(self, image):
        return self._logits(image)


class _MlpModel:
    def __init__(self, teacher_model):
        self.teacher_model = teacher_model


def _loader(marker, labels):
    return [{"img": torch.full((len(labels), 3, 1, 1), float(marker)),
             "label": torch.tensor(labels)}]


def test_the_mlp_teacher_reads_its_own_normalization_when_one_is_given():
    # One loader on the LoRA branch's normalization (marker 1). Each teacher is
    # right only on its own -- so without `to_mlp_norm` converting the batch to
    # marker 2, acc_mlp would be 0 instead of 100.
    acc_lora, acc_mlp, acc_ens = evaluate(
        _LoraTeacher(1, pred_class=1), _MlpModel(_MlpBackbone(2, pred_class=1)), lambda x: x,
        _loader(1, [1, 1]), torch.device("cpu"), to_mlp_norm=lambda images: images + 1,
    )
    assert (acc_lora, acc_mlp, acc_ens) == (100.0, 100.0, 100.0)


def test_both_teachers_read_the_batch_unchanged_when_no_converter_is_given():
    acc_lora, acc_mlp, _ = evaluate(
        _LoraTeacher(1, pred_class=1), _MlpModel(_MlpBackbone(1, pred_class=1)), lambda x: x,
        _loader(1, [1, 1]), torch.device("cpu"),
    )
    assert (acc_lora, acc_mlp) == (100.0, 100.0)
