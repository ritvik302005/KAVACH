"""Vehicle registry for the demo console.

Measured from the video: detection, tracking, vehicle class, colour, speed,
heading, and the number plate (read by OCR from the original-resolution frame,
several reads voted together, see plates.py).

Simulated: the registration status (registered / insurance expired / PUC
expired). There is no VAHAN lookup behind it; it is generated from the plate
text so a vehicle keeps the same status. The RTO code comes from the plate
itself. The watchlist is real matching against a demo list in
data/watchlist.json (config.WATCHLIST_FILE).
"""

import json
import random
import time
import zlib
from collections import Counter, OrderedDict, deque

import cv2
import numpy as np

from app import config
from app.plates import format_plate, get_reader, vote_plate

CLASS_TYPES = {2: 'CAR', 3: 'TWO-WHEELER', 5: 'BUS', 7: 'TRUCK'}

# Registering offices we can name; any other plate shows just its state-RTO code.
RTO_OFFICES = {
    ('DL', 1): 'Delhi North', ('DL', 3): 'Delhi Sheikh Sarai', ('DL', 4): 'Delhi Janakpuri',
    ('DL', 7): 'Delhi Mayur Vihar', ('DL', 8): 'Delhi Wazirpur', ('DL', 12): 'Delhi Vasant Vihar',
    ('UP', 14): 'Ghaziabad', ('UP', 16): 'Gautam Buddh Nagar', ('UP', 32): 'Lucknow',
    ('HR', 26): 'Gurugram', ('HR', 51): 'Faridabad', ('HR', 55): 'Gurugram (Comm.)',
    ('RJ', 14): 'Jaipur', ('PB', 10): 'Ludhiana',
}

# (status, level, weight): simulated registration status.
STATUSES = [
    ('REGISTERED', 'ok', 84),
    ('INSURANCE EXPIRED', 'warn', 9),
    ('PUC EXPIRED', 'warn', 7),
]
WATCHLIST_STATUS = 'WATCHLIST MATCH (DEMO)'

# Real-world height used to turn pixel speed into ground speed.
TYPE_HEIGHT_M = {'CAR': 1.5, 'TWO-WHEELER': 1.6, 'BUS': 3.2, 'TRUCK': 2.8}

COLOR_SWATCH = {
    'WHITE': '#e8e8e8', 'SILVER': '#a9adb3', 'GREY': '#6b6f75', 'BLACK': '#1c1c1c',
    'RED': '#c62828', 'MAROON': '#6d1b1b', 'ORANGE': '#e0702a', 'YELLOW': '#e3c02b',
    'GREEN': '#2e7d32', 'BLUE': '#1f5fbf', 'BROWN': '#6d4c33',
}


def classify_color(crop):
    """Name the dominant body colour of a BGR crop."""
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = (float(np.median(hsv[:, i])) for i in range(3))
    if v < 55:
        return 'BLACK'
    # Camera white balance tints grey paint and road slightly blue, so the
    # neutral band is wider for blue hues than for the rest.
    if s < 60 or (85 <= h < 130 and s < 95):
        return 'WHITE' if v > 175 else 'SILVER' if v > 115 else 'GREY'
    if h < 8 or h >= 170:
        return 'RED' if v > 110 else 'MAROON'
    if h < 20:
        return 'ORANGE' if v > 140 else 'BROWN'
    if h < 34:
        return 'YELLOW'
    if h < 85:
        return 'GREEN'
    if h < 130:
        return 'BLUE'
    return 'MAROON'


def heading(vx, vy):
    """Compass-style heading from on-screen motion."""
    if abs(vx) < 0.3 and abs(vy) < 0.3:
        return 'HALTED'
    if abs(vx) >= abs(vy):
        return 'EAST' if vx > 0 else 'WEST'
    return 'APPROACHING' if vy > 0 else 'RECEDING'


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


class Watchlist:
    """Plates to flag, loaded from a JSON file. Matching tolerates a small OCR
    misread: same length and at most `max_misread` characters different."""

    def __init__(self, entries=(), max_misread=0):
        self.entries = [(e['plate'].replace(' ', '').upper(), e.get('reason') or 'on the watchlist')
                        for e in entries]
        self.max_misread = max_misread

    @classmethod
    def load(cls, path, max_misread=0):
        try:
            with open(path, encoding='utf-8') as f:
                return cls(json.load(f).get('entries', []), max_misread)
        except FileNotFoundError:
            return cls(max_misread=max_misread)
        except (OSError, ValueError, AttributeError, KeyError) as exc:
            print(f"Watchlist not loaded from {path}: {type(exc).__name__}: {exc}")
            return cls(max_misread=max_misread)

    def match(self, plate):
        """Reason string if `plate` is on the list, else None."""
        for listed, reason in self.entries:
            if len(listed) == len(plate) and _hamming(listed, plate) <= self.max_misread:
                return reason
        return None


class VehicleRegistry:
    def __init__(self, reader=None, watchlist=None):
        self.records = OrderedDict()   # record id -> record, oldest first
        self.alias = {}                # tracker id -> record id it was re-associated to
        self.fps = 30.0
        # `reader` needs .available and .submit(crop, callback); None = the shared reader.
        self.reader = reader if reader is not None else get_reader()
        self.watchlist = watchlist if watchlist is not None else Watchlist.load(
            config.WATCHLIST_FILE, config.WATCHLIST_MAX_MISREAD)
        self._inbox = deque()          # (record id, plate, confidence) from the OCR thread
        self._flagged = []             # records newly matched to the watchlist

    def _reassociate(self, vtype, box, frame_count):
        """A record recently lost at this spot, or None.

        In dense traffic the tracker drops a vehicle behind another and returns
        it under a new id. Without this the same car would be read again from
        scratch and could show up twice in the log.
        """
        x1, y1, x2, y2 = box
        cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
        best, best_d = None, None
        for rec in self.records.values():
            gap = frame_count - rec['last_frame']
            if rec['type'] != vtype or not 0 < gap <= config.VEHICLE_REASSOC_FRAMES:
                continue
            px1, py1, px2, py2 = rec['box']
            d = np.hypot((px1 + px2) / 2 - cx, (py1 + py2) / 2 - cy)
            if d < 0.6 * size and (best_d is None or d < best_d):
                best, best_d = rec, d
        return best

    # ── Plate reading ────────────────────────────────────────────────────────
    def _deliver(self, rec_id, plate, confidence):
        """Called on the OCR thread; the main thread picks it up in poll()."""
        self._inbox.append((rec_id, plate, confidence))

    def _maybe_read_plate(self, rec, frame, native, box, frame_count):
        if (self.reader is None or not self.reader.available or native is None
                or rec['inflight'] or rec['plate_done']
                or rec['attempts'] >= config.PLATE_MAX_ATTEMPTS
                or frame_count - rec['last_submit'] < config.PLATE_READ_EVERY_FRAMES):
            return
        x1, y1, x2, y2 = box
        if x2 - x1 < config.PLATE_MIN_BOX_WIDTH:
            return
        # The plate is read from the original frame: at the 1280x720 working
        # size the characters are too small and OCR garbles them.
        k = native.shape[1] / frame.shape[1]
        top = y1 + config.PLATE_CROP_TOP * (y2 - y1)
        crop = native[max(0, int(top * k)):int(y2 * k), max(0, int(x1 * k)):int(x2 * k)]
        if crop.size == 0:
            return
        rec['inflight'] = True
        rec['attempts'] += 1
        rec['last_submit'] = frame_count
        rec_id = rec['id']
        self.reader.submit(crop.copy(), lambda plate, conf: self._deliver(rec_id, plate, conf))

    def poll(self):
        """Apply finished plate reads. Returns vehicles newly matched to the
        watchlist since the last call."""
        while self._inbox:
            rec_id, plate, confidence = self._inbox.popleft()
            rec = self.records.get(rec_id)
            if rec is None:
                continue
            rec['inflight'] = False
            if plate:
                rec['reads'].append((plate, confidence))
                self._update_plate(rec)
        flagged, self._flagged = self._flagged, []
        return flagged

    def _update_plate(self, rec):
        plate, confidence, agreement = vote_plate(rec['reads'])
        n = len(rec['reads'])
        created = rec['profile'] is None
        if created:
            # One read can be wrong; wait for a second that agrees (or a third).
            if n < 3 and not (n >= 2 and agreement >= config.PLATE_MIN_AGREEMENT):
                return
            rec['profile'] = self._profile(plate)
        profile = rec['profile']
        changed = created or profile['plate_raw'] != plate
        profile['plate'], profile['plate_raw'] = format_plate(plate), plate
        profile['confidence'] = confidence
        if changed:
            self._merge_same_plate(rec, plate)
        reason = self.watchlist.match(plate)
        if reason and not rec['alerted']:
            profile.update(status=WATCHLIST_STATUS, level='alert', reason=reason)
            rec['alerted'] = True
            self._flagged.append(rec)
        if n >= config.PLATE_MAX_READS:
            rec['plate_done'] = True

    def _merge_same_plate(self, rec, plate):
        """The tracker can lose a vehicle for good (it stops behind traffic, then
        returns under a new id too late to be re-associated). Once the plate
        shows it is the same vehicle, keep one row and do not alert twice."""
        for other_id, other in list(self.records.items()):
            if other is rec or not other['profile'] or other['profile'].get('plate_raw') != plate:
                continue
            rec['first_seen'] = other['first_seen']
            rec['alerted'] = rec['alerted'] or other['alerted']
            if other['alerted']:
                rec['profile'].update(status=other['profile']['status'],
                                      level=other['profile']['level'],
                                      reason=other['profile']['reason'])
            del self.records[other_id]
            for tracker_id, target in self.alias.items():
                if target == other_id:
                    self.alias[tracker_id] = rec['id']
            self.alias[other_id] = rec['id']

    def _profile(self, plate):
        """Simulated registration details, stable for a given plate."""
        rng = random.Random(zlib.crc32(plate.encode()))
        state, number = plate[:2], int(plate[2:4])
        office = RTO_OFFICES.get((state, number))
        status = rng.choices(STATUSES, weights=[s[2] for s in STATUSES])[0]
        return {
            'rto': f"{state}-{number:02d}" + (f" {office}" if office else ''),
            'status': status[0],
            'level': status[1],
            'reason': None,
        }

    def observe(self, track_id, cls, frame, box, velocity, frame_count, zones, native=None):
        """Update one vehicle from a detection and queue a plate read if due.

        `frame` is the working-size frame the box is in; `native` is the same
        frame at the camera's own resolution, used for reading the plate.
        """
        seen_type = CLASS_TYPES.get(cls, 'CAR')
        rec = self.records.get(self.alias.get(track_id, track_id))
        vtype = rec['type'] if rec is not None else seen_type
        if rec is None:
            rec = self._reassociate(vtype, box, frame_count)
            if rec is not None:
                self.alias[track_id] = rec['id']
            else:
                rec = {
                    'id': track_id, 'type': vtype, 'types': Counter(), 'sightings': 0,
                    'colors': Counter(),
                    'color': None, 'speed': 0.0, 'first_seen': time.strftime('%H:%M:%S'),
                    'profile': None, 'alerted': False,
                    'reads': [], 'attempts': 0, 'inflight': False,
                    'last_submit': -config.PLATE_READ_EVERY_FRAMES, 'plate_done': False,
                }
                self.records[track_id] = rec
                while len(self.records) > config.VEHICLE_LOG_MAX:
                    self.records.popitem(last=False)
                if len(self.alias) > 4 * config.VEHICLE_LOG_MAX:
                    self.alias = {k: v for k, v in self.alias.items() if v in self.records}

        rec['sightings'] += 1
        # YOLO flips class on the same vehicle between frames (truck/bus/car);
        # vote until the plate is read, then the type is fixed with the make.
        if rec['profile'] is None:
            rec['types'][seen_type] += 1
            rec['type'] = vtype = rec['types'].most_common(1)[0][0]
        rec['last_frame'] = frame_count
        rec['box'] = box
        rec['zones'] = zones

        x1, y1, x2, y2 = box
        if rec['sightings'] <= config.COLOR_SAMPLE_SIGHTINGS:
            w, h = x2 - x1, y2 - y1
            crop = frame[max(0, y1 + h // 4):max(0, y2 - h // 3),
                         max(0, x1 + w // 5):max(0, x2 - w // 5)]
            name = classify_color(crop)
            if name:
                rec['colors'][name] += 1
                rec['color'] = rec['colors'].most_common(1)[0][0]

        vx, vy = velocity
        box_h = max(1, y2 - y1)
        metres_per_px = TYPE_HEIGHT_M[vtype] / box_h
        kmh = float(np.hypot(vx, vy)) * self.fps * metres_per_px * 3.6
        rec['speed'] = rec['speed'] * 0.7 + min(kmh, 140.0) * 0.3
        rec['heading'] = heading(vx, vy)

        self._maybe_read_plate(rec, frame, native, box, frame_count)
        return rec

    def summaries(self, frame_count):
        """Newest first, for the console's vehicle panel."""
        out = []
        for rec in reversed(self.records.values()):
            p = rec['profile'] or {}
            speed = rec['speed']
            out.append({
                'id': rec['id'],
                'type': rec['type'],
                'plate': p.get('plate'),
                'confidence': p.get('confidence'),
                'plate_reads': len(rec['reads']),
                'rto': p.get('rto'),
                'status': p.get('status'),
                'level': p.get('level'),
                'color': rec['color'],
                'color_hex': COLOR_SWATCH.get(rec['color']),
                'speed_kmh': 0 if speed < 3 else round(speed),
                'heading': rec.get('heading'),
                'zones': rec.get('zones', []),
                'first_seen': rec['first_seen'],
                'in_view': frame_count - rec.get('last_frame', 0) <= config.VEHICLE_IN_VIEW_FRAMES,
            })
        return out
