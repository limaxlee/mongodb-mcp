import random

from common.constants import DriftWindowMode
from mongodb_mcp.drift import RecordExtractor
from tests.drift.synthetic import make_row, cls_prediction, det_prediction, START


def extract_one(row, window=DriftWindowMode.CURRENT, task=None):
    """Feeds one row into a fresh extractor and returns the extractor with the record it produced, if any"""
    extractor = RecordExtractor(task=task)
    extractor.add_record(row, window)
    return extractor, (extractor.records[0] if extractor.records else None)


class TestGetConfidence:
    def test_list_takes_maximum(self):
        extractor = RecordExtractor()
        assert extractor.get_confidence([0.05, 0.95]) == 0.95
        assert extractor.get_confidence([0.8]) == 0.8

    def test_scalar_is_kept(self):
        extractor = RecordExtractor()
        assert extractor.get_confidence(0.7) == 0.7
        assert extractor.get_confidence(1) == 1.0

    def test_missing_or_invalid_is_none(self):
        extractor = RecordExtractor()
        assert extractor.get_confidence([]) is None
        assert extractor.get_confidence(None) is None
        assert extractor.get_confidence("high") is None
        assert extractor.get_confidence([True]) is None
        assert extractor.get_confidence(True) is None
        assert extractor.get_confidence([0.4, "x", None]) == 0.4


class TestClassificationExtraction:
    def test_record_fields(self):
        prediction = cls_prediction(random.Random(1), 0.9, 0.0)
        prediction["confidence"] = [0.82, 0.18]
        extractor, record = extract_one(make_row(1, START, "cls", prediction))

        assert record is not None
        assert record.window == DriftWindowMode.CURRENT
        assert record.prediction == "Good"
        assert record.max_confidence == 0.82
        assert record.below_threshold is False
        assert record.near_threshold is True
        assert record.image_spec == [400, 400, 3]
        assert record.created_at == START
        assert record.gbm == "SEV" and record.equipment_id == "Metal_Inspector_01" and record.backend == "ts"
        assert extractor.get_task() == "cls"
        assert extractor.classes == ["Good", "NG"]

    def test_window_tag(self):
        row = make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0))
        _, record = extract_one(row, DriftWindowMode.REFERENCE)
        assert record.window == DriftWindowMode.REFERENCE

    def test_null_threshold(self):
        row = make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0, threshold=None))
        _, record = extract_one(row)
        assert record.threshold is None
        assert record.below_threshold is None
        assert record.near_threshold is None

    def test_missing_confidence_is_counted_and_the_record_kept(self):
        prediction = cls_prediction(random.Random(1), 0.9, 0.0)
        prediction["confidence"] = []
        extractor, record = extract_one(make_row(1, START, "cls", prediction))

        assert record is not None
        assert record.max_confidence is None
        assert record.below_threshold is None and record.near_threshold is None
        assert extractor.quality.missing_confidence_count == 1
        assert extractor.quality.parse_error_count == 0



class TestDetectionExtraction:
    def test_list_confidence_uses_maximum(self):
        prediction = det_prediction(random.Random(2), 0.9, 0.0, boxes_mean=2.0, no_box_rate=0.0, list_confidence=True)
        _, record = extract_one(make_row(1, START, "det", prediction))

        assert record.box_count == len(prediction["detections"])
        for box, detection in zip(record.boxes, prediction["detections"]):
            assert box.max_confidence == max(detection["confidence"])

    def test_geometry_and_normalisation(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["detections"] = [{
            "bboxId": 0, "prediction": "NG", "confidence": 0.7, "threshold": 0.8,
            "bbox": [100.0, 100.0, 200.0, 150.0]
        }]
        _, record = extract_one(make_row(1, START, "det", prediction))
        box = record.boxes[0]

        assert (box.width, box.height, box.area, box.aspect) == (100.0, 50.0, 5000.0, 2.0)
        assert (box.cx, box.cy) == (150.0, 125.0)
        assert box.normalized_width == 0.25 and box.normalized_height == 0.125
        assert box.normalized_area == 5000 / 160000
        assert box.normalized_cx == 0.375 and box.normalized_cy == 0.3125
        assert box.below_threshold is True
        assert record.threshold == 0.8
        assert record.box_count_by_class == {"Good": 0, "NG": 1}
        assert record.below_threshold_count == 1
        assert record.min_confidence == 0.7 and record.mean_confidence == 0.7

    def test_file_index_beyond_data_spec(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["fileIndex"] = 4
        extractor, record = extract_one(make_row(1, START, "det", prediction))

        assert record.image_spec is None
        assert all(box.normalized_area is None for box in record.boxes)
        assert extractor.quality.missing_image_spec_count == 1

    def test_no_boxes(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=1.0)
        _, record = extract_one(make_row(1, START, "det", prediction))
        assert record.has_no_boxes and record.mean_confidence is None and record.threshold is None

    def test_missing_box_confidence_is_counted(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["detections"][0]["confidence"] = None
        extractor, record = extract_one(make_row(1, START, "det", prediction))
        assert record.boxes[0].max_confidence is None and record.boxes[0].below_threshold is None
        assert extractor.quality.missing_confidence_count == 1

    def test_malformed_box_is_skipped(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["detections"].append({"prediction": "NG", "confidence": 0.9, "bbox": [1, 2]})
        extractor, record = extract_one(make_row(1, START, "det", prediction))
        assert record.box_count == len(prediction["detections"]) - 1
        assert extractor.quality.parse_error_count == 1


class TestExtractorBookkeeping:
    def test_model_repeated_in_one_document(self):
        rng = random.Random(4)
        extractor = RecordExtractor()
        extractor.add_record(make_row(1, START, "cls", cls_prediction(rng, 0.9, 0.0), entry_index=0))
        extractor.add_record(make_row(1, START, "cls", cls_prediction(rng, 0.9, 0.0), entry_index=2))

        assert len(extractor.records) == 2
        assert extractor.quality.matched_document_count == 1
        assert extractor.quality.matched_entry_count == 2

    def test_duplicate_prediction_across_windows_is_ignored(self):
        # A document stamped exactly on the boundary between the windows is returned by both queries
        row = make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0))
        extractor = RecordExtractor()
        extractor.add_record(row, DriftWindowMode.REFERENCE)
        extractor.add_record(dict(row), DriftWindowMode.CURRENT)

        assert len(extractor.records) == 1
        assert extractor.records[0].window == DriftWindowMode.REFERENCE
        assert extractor.quality.matched_document_count == 1
        assert extractor.quality.matched_entry_count == 1

    def test_malformed_row_is_counted_not_raised(self):
        extractor = RecordExtractor()
        row = make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0))
        row["createdAt"] = "not a date"
        extractor.add_record(row)
        assert extractor.quality.parse_error_count == 1
        assert extractor.quality.parse_error_examples == [row["_id"]]
        assert extractor.records == []

    def test_requested_task_filters_rows(self):
        extractor = RecordExtractor(task="det")
        extractor.add_record(make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0)))
        assert extractor.records == []
        assert extractor.quality.matched_document_count == 0
        assert extractor.get_task() == "det"

    def test_first_task_seen_becomes_the_task(self):
        rng = random.Random(1)
        extractor = RecordExtractor()
        assert extractor.get_task() is None

        extractor.add_record(make_row(1, START, "det", det_prediction(rng, 0.9, 0.0)))
        assert extractor.get_task() == "det"

        # A later row of another task is ignored, it neither becomes a record nor counts as matched
        extractor.add_record(make_row(2, START, "cls", cls_prediction(rng, 0.9, 0.0)))
        assert extractor.get_task() == "det"
        assert len(extractor.records) == 1
        assert extractor.quality.matched_document_count == 1
