from __future__ import annotations

import unittest

import numpy as np

from opengazelink_pc.shared_eye_appearance import (
    BASE_HEIGHT,
    BASE_WIDTH,
    BaseEyeImage,
    CNN_MODEL_HEIGHT,
    CNN_MODEL_WIDTH,
    augment_cnn_eye_input,
    cnn_eye_input,
    deserialize_base_eye,
    serialize_base_eye,
)


class CnnEyeInputTest(unittest.TestCase):
    def setUp(self) -> None:
        yy, xx = np.mgrid[0:CNN_MODEL_HEIGHT, 0:CNN_MODEL_WIDTH]
        self.gray = (45.0 + 2.1 * xx + 1.7 * yy + 18.0 * np.sin(xx / 4.0)).astype(np.float32)
        radius = (
            (xx - (CNN_MODEL_WIDTH - 1) * 0.5) / (CNN_MODEL_WIDTH * 0.45)
        ) ** 2 + (
            (yy - (CNN_MODEL_HEIGHT - 1) * 0.5) / (CNN_MODEL_HEIGHT * 0.36)
        ) ** 2
        self.alpha = np.clip((1.08 - radius) * 7.0, 0.0, 1.0).astype(np.float32)

    def test_exposes_normalized_intensity_and_soft_mask_channels(self) -> None:
        values = cnn_eye_input(self.gray, self.alpha)
        self.assertEqual((2, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH), values.shape)
        np.testing.assert_allclose(values[1], self.alpha, atol=1e-6)
        self.assertTrue(np.all(values[0][self.alpha == 0.0] == 0.0))
        self.assertTrue(np.isfinite(values).all())

    def test_is_invariant_to_affine_brightness_change_without_clipping(self) -> None:
        original = cnn_eye_input(self.gray, self.alpha)
        relit = cnn_eye_input(self.gray * 1.25 + 17.0, self.alpha)
        np.testing.assert_allclose(original, relit, atol=2e-5)

    def test_cnn_augmentation_modes_preserve_model_input_contract(self) -> None:
        for mode in ("geometry", "photometric"):
            with self.subTest(mode=mode):
                values = augment_cnn_eye_input(
                    self.gray, self.alpha, np.random.default_rng(73), mode=mode,
                )
                self.assertEqual((2, CNN_MODEL_HEIGHT, CNN_MODEL_WIDTH), values.shape)
                self.assertTrue(np.isfinite(values).all())
                self.assertGreater(float(np.std(values[0])), 0.01)
                self.assertGreaterEqual(float(np.min(values[1])), 0.0)
                self.assertLessEqual(float(np.max(values[1])), 1.0)

    def test_unknown_cnn_augmentation_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown CNN augmentation mode"):
            augment_cnn_eye_input(
                self.gray, self.alpha, np.random.default_rng(73), mode="strong",
            )

    def test_base_eye_serialization_records_its_dimensions(self) -> None:
        base = BaseEyeImage(
            gray_base=self.gray,
            corners=np.asarray([[8.0, 18.0], [55.0, 18.0]], dtype=np.float64),
            contour=np.asarray([[8.0, 18.0], [32.0, 10.0], [55.0, 18.0]], dtype=np.float64),
        )
        payload = serialize_base_eye(base)
        self.assertEqual([BASE_WIDTH, BASE_HEIGHT], payload["size"])
        restored = deserialize_base_eye(payload)
        self.assertEqual((BASE_HEIGHT, BASE_WIDTH), restored.gray_base.shape)
        np.testing.assert_allclose(restored.corners, base.corners)


if __name__ == "__main__":
    unittest.main()
