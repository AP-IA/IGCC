"""CPU regression checks: python -m unittest discover -s tests -v."""
import ast
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from network.PFE import AdaptiveWeighting, Pairwise_Feature_Enhancer, get_pairs


ROOT = Path(__file__).resolve().parents[1]


def load_cal_loss():
    # Exercise the actual function without importing training-only GPU/data deps.
    tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cal_loss")
    scope = {"torch": torch, "OrderedDict": OrderedDict,
             "utils": SimpleNamespace(cal_accuracy=lambda *_: 0.5)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "train.py", "exec"), scope)
    return scope["cal_loss"]


class PairMiningTests(unittest.TestCase):
    def test_nearest_pairs_exclude_self_and_wrong_class(self):
        # The old in-place mask mutation picked self for every negative here.
        features = torch.tensor([[0., 0.], [1., 0.], [3., 0.], [4., 0.]])
        labels = torch.tensor([0, 0, 1, 1])
        positive, negative, _, _ = get_pairs(features, labels)
        self.assertEqual(positive[:, 1].tolist(), [1, 0, 3, 2])
        self.assertEqual(negative[:, 1].tolist(), [2, 2, 1, 1])
        self.assertTrue(torch.all(labels[negative[:, 0]] != labels[negative[:, 1]]))

    def test_zero_distance_other_class_is_a_valid_negative(self):
        positive, negative, _, _ = get_pairs(torch.zeros(4, 2), torch.tensor([0, 0, 1, 1]))
        self.assertTrue(torch.all(positive[:, 0] != positive[:, 1]))
        self.assertTrue(torch.all(negative[:, 0] != negative[:, 1]))

    def test_missing_positive_only_excludes_ineligible_anchors(self):
        positive, negative, _, _ = get_pairs(torch.arange(6.).view(3, 2), torch.tensor([0, 0, 1]))
        self.assertEqual(positive[:, 0].tolist(), [0, 1])
        self.assertEqual(negative[:, 1].tolist(), [2, 2])

    def test_no_pairs_for_single_class_or_all_singletons(self):
        for labels in (torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]), torch.tensor([0])):
            with self.subTest(labels=labels.tolist()):
                pairs = get_pairs(torch.randn(len(labels), 3), labels)
                for tensor in pairs:
                    self.assertEqual(tuple(tensor.shape), (0, 2))

    def test_enhancer_backward_with_and_without_valid_pairs(self):
        for labels in (torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 0, 0])):
            with self.subTest(labels=labels.tolist()):
                features = torch.randn(4, 3, requires_grad=True)
                a, b = Pairwise_Feature_Enhancer(
                    features, features.detach(), labels,
                    AdaptiveWeighting(3), nn.Linear(3, 2), nn.Softmax(dim=1))
                self.assertEqual(a.shape, b.shape)
                (a.sum() + b.sum()).backward()
                self.assertTrue(torch.isfinite(features.grad).all())


class TrainingLossTests(unittest.TestCase):
    def test_dynamic_loss_added_once_and_empty_ranking_is_finite(self):
        cal_loss = load_cal_loss()
        for no_pfe in (True, False):
            with self.subTest(no_pfe=no_pfe):
                logits = torch.ones(2, 2, requires_grad=True)

                def model(inputs, targets=None):
                    if no_pfe:
                        return logits, logits, []
                    empty = logits[:0].sum(dim=1)
                    return logits, empty, empty, logits, []

                criteria = (lambda *_: logits.sum() * 0 + 3,
                            nn.MarginRankingLoss(margin=0.05),
                            lambda *_: logits.sum() * 0 + 5)
                loss, details, _ = cal_loss(
                    model, torch.zeros(2, 2), torch.tensor([0, 1]),
                    criteria, nullcontext(), "cpu", SimpleNamespace(NoPFE=no_pfe))
                self.assertEqual(loss.item(), 11.)  # 2 * CE(3) + dynamic(5)
                self.assertEqual(details["dynamic_loss"], 5.)
                if not no_pfe:
                    self.assertEqual(details["PFE_loss"], 0.)
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
