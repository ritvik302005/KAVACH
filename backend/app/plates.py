"""Number plate reading: OCR on a vehicle crop, Indian plate format, and voting.

One OCR read of one frame is unreliable (B read as 8, an extra letter in front,
a digit dropped). Three things make the result usable:

1. `normalize_plate` finds the part of the text shaped like an Indian plate
   (state code, RTO number, series letters, four digits) and fixes look-alike
   characters by position: a letter slot cannot hold an 8, a digit slot cannot
   hold a B.
2. `vote_plate` combines several reads of the same vehicle, character by
   character, so one bad frame is outvoted.
3. `PlateReader` runs the OCR on its own thread, so detection and the video
   never wait for it.

The OCR engine (RapidOCR, ONNX, CPU) is imported on first use. If it is not
installed the reader reports itself unavailable and no plates are read.
"""

import re
import threading
from collections import Counter, deque

import cv2
import numpy as np

from app import config

STATE_CODES = frozenset(
    'AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ '
    'NL OD OR PB PY RJ SK TN TR TS UK UP WB'.split())

# What an OCR commonly returns where the plate has the other kind of character.
TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}
TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2',
            'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}

_CLEAN = re.compile(r'[^A-Z0-9]')


def _fit(window, series_len):
    """Force `window` into SS NN LLL NNNN by position. Returns (text, fixes)
    or None when a position cannot be repaired."""
    kinds = 'LL' + 'DD' + 'L' * series_len + 'DDDD'
    out, fixes = [], 0
    for ch, kind in zip(window, kinds):
        if kind == 'L':
            if ch.isdigit():
                ch = TO_LETTER.get(ch)
                fixes += 1
        else:
            if ch.isalpha():
                ch = TO_DIGIT.get(ch)
                fixes += 1
        if ch is None:
            return None
        out.append(ch)
    return ''.join(out), fixes


def normalize_plate(text):
    """Best Indian-format plate inside `text` as 'UP23BT4920', or None.

    Looks at every window of the right length, repairs look-alike characters by
    position, keeps only windows that start with a real state code, and prefers
    the one needing the fewest repairs.
    """
    s = _CLEAN.sub('', (text or '').upper())
    best = None
    for series_len in (1, 2, 3):
        n = 4 + series_len + 4
        for start in range(0, len(s) - n + 1):
            fitted = _fit(s[start:start + n], series_len)
            if fitted is None or fitted[0][:2] not in STATE_CODES:
                continue
            key = (fitted[1], start)
            if best is None or key < best[0]:
                best = (key, fitted[0])
    return best[1] if best else None


def format_plate(plate):
    """'UP23BT4920' -> 'UP 23 BT 4920'."""
    return f"{plate[:2]} {plate[2:4]} {plate[4:-4]} {plate[-4:]}"


def vote_plate(reads):
    """Combine reads [(plate, ocr_confidence), ...] into one plate.

    Returns (plate, confidence, agreement) or None. Only reads of the most
    common length vote (a different length means a different series), each
    character position goes to the weighted majority, and `agreement` is the
    share of the vote the winners held.
    """
    reads = [(p, c) for p, c in reads if p]
    if not reads:
        return None
    length = Counter(len(p) for p, _ in reads).most_common(1)[0][0]
    group = [(p, c) for p, c in reads if len(p) == length]
    plate, share = [], []
    for i in range(length):
        weight = Counter()
        for p, c in group:
            weight[p[i]] += c
        ch, w = weight.most_common(1)[0]
        plate.append(ch)
        share.append(w / sum(weight.values()))
    agreement = sum(share) / len(share)
    mean_conf = sum(c for _, c in group) / len(group)
    return ''.join(plate), round(mean_conf * agreement, 2), round(agreement, 2)


def _candidates(boxes):
    """Texts worth trying: each text box alone, and all boxes joined in reading
    order (two-wheeler plates are often split over two lines)."""
    # Each box is four [x, y] corners, top-left first.
    items = [(b[0][1], b[0][0], t, float(c)) for b, t, c in boxes]
    texts = [(t, c) for _, _, t, c in items]
    if len(items) > 1:
        ordered = sorted(items, key=lambda i: (round(i[0] / 25), i[1]))
        texts.append((''.join(i[2] for i in ordered),
                      sum(i[3] for i in ordered) / len(ordered)))
    return texts


class PlateReader:
    """OCR on a background thread. `submit` never blocks the caller."""

    def __init__(self, queue_size=6, target_width=500):
        self._queue = deque()
        self._queue_size = queue_size
        self._target_width = target_width
        self._cond = threading.Condition()
        self._engine = None
        self._thread = None
        self.available = True

    def read_crop(self, crop):
        """OCR one BGR crop. Returns (plate, confidence) or None."""
        engine = self._load()
        if engine is None or crop is None or crop.size == 0:
            return None
        scale = max(1.0, self._target_width / crop.shape[1])
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        boxes, _ = engine(crop)
        best = None
        for text, conf in _candidates(boxes or []):
            plate = normalize_plate(text)
            if plate and (best is None or conf > best[1]):
                best = (plate, conf)
        return best

    def submit(self, crop, callback):
        """Queue a crop. `callback(plate_or_None, confidence)` runs on the OCR
        thread. When the queue is full the oldest job is answered with None."""
        dropped = None
        with self._cond:
            if not self.available:
                dropped = callback
            else:
                if len(self._queue) >= self._queue_size:
                    dropped = self._queue.popleft()[1]
                self._queue.append((crop, callback))
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, daemon=True)
                    self._thread.start()
                self._cond.notify()
        if dropped is not None:
            dropped(None, 0.0)

    @staticmethod
    def _limit_threads():
        """RapidOCR gives onnxruntime every core. On a CPU-only machine that
        starves the YOLO detector (measured: detection ran 2.5x slower while
        plates were being read), so cap the OCR's threads."""
        import rapidocr_onnxruntime.utils as ocr_utils
        make_options = ocr_utils.SessionOptions

        def limited():
            options = make_options()
            options.intra_op_num_threads = config.PLATE_OCR_THREADS
            options.inter_op_num_threads = 1
            return options

        ocr_utils.SessionOptions = limited

    def _load(self):
        if self._engine is None and self.available:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._limit_threads()
                # Defaults scale every crop up to a 736 px short side and run a
                # text-rotation classifier; plates are small and upright, and
                # that is most of the cost. (det_model_path=None keeps the
                # bundled model; the library requires the key.)
                self._engine = RapidOCR(use_angle_cls=False, det_model_path=None,
                                        det_limit_side_len=640, det_limit_type='max')
            except Exception as exc:
                print(f"Plate reading unavailable ({type(exc).__name__}: {exc}). "
                      "Install it with: pip install rapidocr-onnxruntime")
                self.available = False
        return self._engine

    def _run(self):
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                crop, callback = self._queue.popleft()
            try:
                result = self.read_crop(crop)
            except Exception as exc:   # a bad crop must not kill the reader
                print(f"Plate read error: {type(exc).__name__}: {exc}")
                result = None
            if not self.available:
                # The engine failed to load; answer everything still waiting.
                with self._cond:
                    pending, self._queue = list(self._queue), deque()
                for _, cb in pending:
                    cb(None, 0.0)
            callback(*(result or (None, 0.0)))


_reader = None
_reader_lock = threading.Lock()


def get_reader():
    """The one shared reader (it owns the OCR model, which is large)."""
    global _reader
    with _reader_lock:
        if _reader is None:
            _reader = PlateReader()
        return _reader
