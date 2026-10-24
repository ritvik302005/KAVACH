import unittest

import numpy as np

from app import config
from app.vehicles import VehicleRegistry, Watchlist, classify_color

CAR = 2
FRAME = np.full((720, 1280, 3), 200, dtype=np.uint8)
NATIVE = np.full((1080, 1920, 3), 200, dtype=np.uint8)
BOX = (100, 100, 400, 300)     # 300 px wide: close enough to read


class FakeReader:
    """Collects submitted crops; the test answers them by hand."""
    available = True

    def __init__(self):
        self.jobs = []

    def submit(self, crop, callback):
        self.jobs.append((crop, callback))

    def answer(self, plate, conf=0.85):
        _, callback = self.jobs.pop(0)
        callback(plate, conf)


def registry(watch=()):
    reader = FakeReader()
    return VehicleRegistry(reader=reader, watchlist=Watchlist(watch, max_misread=1)), reader


def observe(reg, track_id, box, frame_count, native=NATIVE):
    return reg.observe(track_id, CAR, FRAME, box, (0.0, 0.0), frame_count, [], native)


def read(reg, reader, plates, track_id=1, box=BOX, start=1):
    """Show the vehicle again and answer each read with the next plate."""
    frame = start
    for plate in plates:
        rec = observe(reg, track_id, box, frame)
        reader.answer(plate)
        reg.poll()
        frame += config.PLATE_READ_EVERY_FRAMES
    return rec, frame


class TestPlateReading(unittest.TestCase):
    def test_no_plate_until_reads_agree(self):
        reg, reader = registry()
        rec, frame = read(reg, reader, ['UP23BT4920'])
        self.assertIsNone(rec['profile'])        # one read is not enough
        rec, _ = read(reg, reader, ['UP23BT4920'], start=frame)
        self.assertEqual(rec['profile']['plate'], 'UP 23 BT 4920')

    def test_a_wrong_read_is_outvoted(self):
        reg, reader = registry()
        rec, _ = read(reg, reader, ['UP23BT4920', 'UP23BT2920', 'UP23BT4920'])
        self.assertEqual(rec['profile']['plate'], 'UP 23 BT 4920')

    def test_crop_comes_from_the_original_frame(self):
        reg, reader = registry()
        observe(reg, 1, BOX, 1)
        crop, _ = reader.jobs[0]
        # 1.5x the working frame; only the lower part of the box is cropped.
        self.assertEqual(crop.shape[1], int(400 * 1.5) - int(100 * 1.5))
        self.assertLess(crop.shape[0], int(300 * 1.5) - int(100 * 1.5))

    def test_far_vehicles_are_not_read(self):
        reg, reader = registry()
        observe(reg, 1, (100, 100, 200, 160), 1)     # 100 px wide
        self.assertEqual(reader.jobs, [])

    def test_no_native_frame_means_no_read(self):
        reg, reader = registry()
        observe(reg, 1, BOX, 1, native=None)
        self.assertEqual(reader.jobs, [])

    def test_one_read_in_flight_per_vehicle_and_spaced_out(self):
        reg, reader = registry()
        observe(reg, 1, BOX, 1)
        observe(reg, 1, BOX, 2)                      # still waiting for the first
        self.assertEqual(len(reader.jobs), 1)
        reader.answer(None)
        reg.poll()
        observe(reg, 1, BOX, 3)                      # too soon after the last read
        self.assertEqual(len(reader.jobs), 0)
        observe(reg, 1, BOX, 1 + config.PLATE_READ_EVERY_FRAMES)
        self.assertEqual(len(reader.jobs), 1)

    def test_gives_up_after_max_attempts(self):
        reg, reader = registry()
        frame = 1
        for _ in range(config.PLATE_MAX_ATTEMPTS + 3):
            observe(reg, 1, BOX, frame)
            if reader.jobs:
                reader.answer(None)
                reg.poll()
            frame += config.PLATE_READ_EVERY_FRAMES
        self.assertEqual(reg.records[1]['attempts'], config.PLATE_MAX_ATTEMPTS)
        self.assertIsNone(reg.records[1]['profile'])

    def test_unavailable_reader_leaves_plates_unread(self):
        reg, reader = registry()
        reader.available = False
        observe(reg, 1, BOX, 1)
        self.assertEqual(reader.jobs, [])
        self.assertIsNone(reg.summaries(1)[0]['plate'])

    def test_details_are_stable_for_a_plate(self):
        a, ra = registry()
        b, rb = registry()
        rec_a, _ = read(a, ra, ['UP14EP7743'] * 2)
        rec_b, _ = read(b, rb, ['UP14EP7743'] * 2, track_id=9)
        self.assertEqual(rec_a['profile']['status'], rec_b['profile']['status'])
        self.assertEqual(rec_a['profile']['rto'], 'UP-14 Ghaziabad')

    def test_vehicle_keeps_its_reads_when_the_tracker_reassigns_its_id(self):
        reg, reader = registry()
        rec, frame = read(reg, reader, ['UP14EP7743'] * 2)
        again = observe(reg, 57, (110, 104, 410, 304), frame + 10)
        self.assertEqual(again['profile']['plate'], 'UP 14 EP 7743')
        self.assertEqual(len(reg.summaries(frame + 10)), 1)

    def test_same_plate_under_a_new_track_is_one_vehicle_and_alerts_once(self):
        reg, reader = registry([{'plate': 'UP23BT4920', 'reason': 'demo'}])
        read(reg, reader, ['UP23BT4920'] * 2)
        self.assertEqual(len(reg.poll()), 0)
        # Much later, far away: too late to be re-associated, so a new record.
        read(reg, reader, ['UP23BT4920'] * 2, track_id=500, box=(700, 300, 1000, 500), start=5000)
        self.assertEqual(len(reg.summaries(5100)), 1)
        self.assertEqual(reg.records[500]['profile']['level'], 'alert')
        self.assertEqual(reg.poll(), [])

    def test_vehicle_merges_when_a_later_vote_corrects_its_plate(self):
        reg, reader = registry()
        read(reg, reader, ['UP23BT4920'] * 2)
        # The new record first settles on a wrong digit, then votes it right.
        rec, frame = read(reg, reader, ['UP23BT1920'] * 2, track_id=500,
                          box=(700, 300, 1000, 500), start=5000)
        self.assertEqual(len(reg.summaries(5100)), 2)
        read(reg, reader, ['UP23BT4920'] * 3, track_id=500, box=(700, 300, 1000, 500), start=frame)
        self.assertEqual(len(reg.summaries(frame + 100)), 1)

    def test_a_different_vehicle_elsewhere_gets_its_own_record(self):
        reg, reader = registry()
        read(reg, reader, ['UP14EP7743'] * 2)
        observe(reg, 2, (900, 400, 1200, 600), 100)
        self.assertEqual(len(reg.summaries(100)), 2)

    def test_late_answer_for_an_evicted_vehicle_is_ignored(self):
        reg, reader = registry()
        observe(reg, 1, BOX, 1)
        reg.records.clear()
        reader.answer('UP14EP7743')
        self.assertEqual(reg.poll(), [])


class TestWatchlist(unittest.TestCase):
    def test_matching_plate_raises_one_alert(self):
        reg, reader = registry([{'plate': 'UP23BT4920', 'reason': 'demo'}])
        rec, frame = read(reg, reader, ['UP23BT4920'])
        observe(reg, 1, BOX, frame)
        reader.answer('UP23BT4920')
        self.assertEqual(reg.poll(), [rec])
        self.assertEqual(rec['profile']['level'], 'alert')
        self.assertEqual(rec['profile']['reason'], 'demo')
        # Further reads of the same vehicle do not alert again.
        observe(reg, 1, BOX, frame + config.PLATE_READ_EVERY_FRAMES)
        reader.answer('UP23BT4920')
        self.assertEqual(reg.poll(), [])

    def test_one_misread_character_still_matches(self):
        watch = Watchlist([{'plate': 'UP23BT4920'}], max_misread=1)
        self.assertIsNotNone(watch.match('UP23BT4320'))
        self.assertIsNone(watch.match('UP23BT4390'))
        self.assertIsNone(watch.match('UP14EP7743'))

    def test_other_plates_are_not_flagged(self):
        reg, reader = registry([{'plate': 'UP23BT4920'}])
        rec, _ = read(reg, reader, ['UP14EP7743'] * 2)
        self.assertNotEqual(rec['profile']['level'], 'alert')

    def test_missing_or_broken_file_means_empty_list(self):
        self.assertIsNone(Watchlist.load('no/such/file.json').match('UP23BT4920'))


class TestColour(unittest.TestCase):
    def test_colour_names(self):
        white = np.full((20, 20, 3), 235, dtype=np.uint8)
        red = np.zeros((20, 20, 3), dtype=np.uint8); red[:, :, 2] = 200
        self.assertEqual(classify_color(white), 'WHITE')
        self.assertEqual(classify_color(red), 'RED')


if __name__ == '__main__':
    unittest.main()
