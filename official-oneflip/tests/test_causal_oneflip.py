import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_oneflip import causal_margin_loss, SPECS, apply_trigger, calibration_subset, eligible_flip, normalize_pixels, restore_pixels


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return torch.full((3, 32, 32), float(index)), index % 10


class CausalOneFlipTests(unittest.TestCase):
    def test_eligible_flip_is_exactly_one_bit(self):
        self.assertEqual(eligible_flip(0.75), 1.5)
        self.assertIsNone(eligible_flip(-0.75))

    def test_trigger_blends_in_raw_pixel_space(self):
        spec = SPECS["CIFAR10"]
        raw = torch.full((1, 3, 32, 32), 0.2)
        images = normalize_pixels(raw, spec)
        changed, changed_raw = apply_trigger(images, torch.ones((32, 32)), torch.full((3, 32, 32), 0.8), spec)
        self.assertTrue(torch.allclose(changed_raw, torch.full_like(changed_raw, 0.8)))
        self.assertTrue(torch.allclose(restore_pixels(changed, spec), changed_raw, atol=1e-6))

    def test_calibration_split_is_deterministic(self):
        first_images, first_labels = calibration_subset(TinyDataset(), 10, 7)
        second_images, second_labels = calibration_subset(TinyDataset(), 10, 7)
        self.assertTrue(torch.equal(first_images, second_images))
        self.assertTrue(torch.equal(first_labels, second_labels))

    def test_margin_loss_prefers_a_causal_interval(self):
        clean = torch.tensor([[0.0, 2.0]])
        flipped_good = torch.tensor([[3.0, 2.0]])
        flipped_bad = torch.tensor([[1.0, 2.0]])
        self.assertLess(
            causal_margin_loss(clean, flipped_good, 0, 0.5, 0.5),
            causal_margin_loss(clean, flipped_bad, 0, 0.5, 0.5),
        )


if __name__ == "__main__":
    unittest.main()
