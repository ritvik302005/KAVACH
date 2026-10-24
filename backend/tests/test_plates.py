import unittest

import numpy as np

from app.plates import PlateReader, format_plate, normalize_plate, vote_plate


class TestNormalizePlate(unittest.TestCase):
    def test_clean_plate_is_unchanged(self):
        self.assertEqual(normalize_plate('UP14EP7743'), 'UP14EP7743')
        self.assertEqual(normalize_plate('up 14 ep 7743'), 'UP14EP7743')

    def test_digit_in_a_letter_slot_is_repaired(self):
        # B read as 8: the series slot cannot hold a digit.
        self.assertEqual(normalize_plate('UP238T4920'), 'UP23BT4920')

    def test_letter_in_a_digit_slot_is_repaired(self):
        self.assertEqual(normalize_plate('UP23BTZ920'), 'UP23BT2920')

    def test_noise_before_the_plate_is_dropped(self):
        self.assertEqual(normalize_plate('FUP23BT4920'), 'UP23BT4920')
        self.assertEqual(normalize_plate('IND UP14EP7743 X'), 'UP14EP7743')

    def test_text_that_is_not_a_plate_is_rejected(self):
        for text in ('', 'MARUTI SUZUKI', 'WAGON R', 'ZZ99AA1234', '12345'):
            self.assertIsNone(normalize_plate(text), text)

    def test_two_line_plate_joined(self):
        self.assertEqual(normalize_plate('UP14' + 'EP7743'), 'UP14EP7743')

    def test_format(self):
        self.assertEqual(format_plate('UP23BT4920'), 'UP 23 BT 4920')
        self.assertEqual(format_plate('DL01C4471'), 'DL 01 C 4471')


class TestVotePlate(unittest.TestCase):
    def test_one_bad_read_is_outvoted(self):
        reads = [('UP23BT4920', 0.8), ('UP23BT2920', 0.7), ('UP23BT4920', 0.85)]
        plate, conf, agreement = vote_plate(reads)
        self.assertEqual(plate, 'UP23BT4920')
        self.assertGreater(agreement, 0.9)

    def test_identical_reads_give_full_agreement(self):
        _, _, agreement = vote_plate([('UP14EP7743', 0.9), ('UP14EP7743', 0.8)])
        self.assertEqual(agreement, 1.0)

    def test_nothing_to_vote_on(self):
        self.assertIsNone(vote_plate([]))
        self.assertIsNone(vote_plate([(None, 0.5)]))

    def test_only_the_most_common_length_votes(self):
        plate, _, _ = vote_plate([('UP23BT4920', 0.8), ('UP23BT4920', 0.8), ('UP23B4920', 0.9)])
        self.assertEqual(plate, 'UP23BT4920')


def _has_ocr():
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_ocr(), 'rapidocr-onnxruntime is not installed')
class TestRealOcr(unittest.TestCase):
    """Runs the real OCR engine on a drawn plate (not a photo, so only a smoke test)."""

    def test_reads_a_drawn_plate(self):
        import cv2
        img = np.full((140, 520, 3), 235, np.uint8)
        cv2.putText(img, 'UP14EP7743', (20, 95), cv2.FONT_HERSHEY_DUPLEX, 2.2, (20, 20, 20), 5)
        result = PlateReader().read_crop(img)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 'UP14EP7743')


class TestPlateReaderQueue(unittest.TestCase):
    def test_unavailable_reader_answers_with_none(self):
        reader = PlateReader()
        reader.available = False
        got = []
        reader.submit(np.zeros((10, 10, 3), np.uint8), lambda plate, conf: got.append((plate, conf)))
        self.assertEqual(got, [(None, 0.0)])

    def test_empty_crop_reads_nothing(self):
        reader = PlateReader()
        reader.available = False
        self.assertIsNone(reader.read_crop(np.zeros((0, 0, 3), np.uint8)))


if __name__ == '__main__':
    unittest.main()
