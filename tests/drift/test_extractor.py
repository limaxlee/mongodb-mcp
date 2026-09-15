import random
import pytest

from common.constants import DriftWindow
from mongodb_mcp.drift import RecordExtractor, UnsupportedTaskError
from tests.drift.synthetic import make_row, cls_prediction, det_prediction, START


class TestNormalizeConfidence:
    def test_list_takes_maximum(self):
        assert RecordExtractor.normalize_confidence([0.05, 0.95]) == 0.95
        assert RecordExtractor.normalize_confidence([0.8]) == 0.8

    def test_scalar(self):
        assert RecordExtractor.normalize_confidence(0.7) == 0.7
        assert RecordExtractor.normalize_confidence(1) == 1.0

    def test_missing(self):
        assert RecordExtractor.normalize_confidence([]) is None
        assert RecordExtractor.normalize_confidence(None) is None
        assert RecordExtractor.normalize_confidence("high") is None
        assert RecordExtractor.normalize_confidence([True]) is None
        assert RecordExtractor.normalize_confidence(True) is None


class TestClassificationExtraction:
    def test_record_fields(self):
        prediction = cls_prediction(random.Random(1), 0.9, 0.0)
        prediction["confidence"] = [0.82, 0.18]
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "cls", prediction))

        assert record is not None
        assert record.window == DriftWindow.CURRENT
        assert record.prediction == "Good" and record.decision == "Good"
        assert record.max_confidence == 0.82
        assert record.below_threshold is False
        assert record.near_threshold is True
        assert record.decision_differs is False
        assert record.patch_width == 66.0 and record.patch_height == 40.0
        assert record.image_spec == [400, 400, 3]
        assert record.created_at == START
        assert record.gbm == "SEV" and record.equipment_id == "Metal_Inspector_01" and record.backend == "ts"
        assert extractor.tasks_seen == {"cls"}
        assert extractor.classes_seen == ["Good", "NG"]

    def test_window_tag(self):
        record = RecordExtractor().add(
            make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0)), DriftWindow.REFERENCE
        )
        assert record.window == DriftWindow.REFERENCE

    def test_null_threshold(self):
        record = RecordExtractor().add(
            make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0, threshold=None))
        )
        assert record.threshold is None
        assert record.below_threshold is None
        assert record.near_threshold is None

    def test_missing_confidence_is_counted(self):
        prediction = cls_prediction(random.Random(1), 0.9, 0.0)
        prediction["confidence"] = []
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "cls", prediction))
        assert record.max_confidence is None
        assert record.below_threshold is None
        assert extractor.quality.missing_confidence_count == 1

    def test_decision_differs(self):
        prediction = cls_prediction(random.Random(1), 0.9, 0.0)
        prediction["decision"] = "NG"
        record = RecordExtractor().add(make_row(1, START, "cls", prediction))
        assert record.prediction == "Good" and record.decision_differs is True


class TestDetectionExtraction:
    def test_list_confidence_uses_maximum(self):
        prediction = det_prediction(random.Random(2), 0.9, 0.0, boxes_mean=2.0, no_box_rate=0.0, list_confidence=True)
        record = RecordExtractor().add(make_row(1, START, "det", prediction))

        assert record.box_count == len(prediction["detections"])
        for box, detection in zip(record.boxes, prediction["detections"]):
            assert box.max_confidence == max(detection["confidence"])

    def test_geometry_and_normalisation(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["detections"] = [{
            "bboxId": 0, "prediction": "NG", "confidence": 0.7, "threshold": 0.8,
            "bbox": [100.0, 100.0, 200.0, 150.0]
        }]
        record = RecordExtractor().add(make_row(1, START, "det", prediction))
        box = record.boxes[0]

        assert (box.width, box.height, box.area, box.aspect) == (100.0, 50.0, 5000.0, 2.0)
        assert (box.cx, box.cy) == (150.0, 125.0)
        assert box.normalized_width == 0.25 and box.normalized_height == 0.125
        assert box.normalized_area == pytest.approx(5000 / 160000)
        assert box.normalized_cx == 0.375 and box.normalized_cy == 0.3125
        assert box.below_threshold is True
        assert record.threshold == 0.8
        assert record.box_count_by_class == {"Good": 0, "NG": 1}
        assert record.below_threshold_count == 1
        assert record.min_confidence == 0.7 and record.mean_confidence == 0.7

    def test_file_index_beyond_data_spec(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["fileIndex"] = 4
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "det", prediction))

        assert record.image_spec is None
        assert all(box.normalized_area is None for box in record.boxes)
        assert extractor.quality.missing_image_spec_count == 1

    def test_no_boxes(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=1.0)
        record = RecordExtractor().add(make_row(1, START, "det", prediction))
        assert record.has_no_boxes and record.mean_confidence is None and record.threshold is None

    def test_malformed_box_is_skipped(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["detections"].append({"prediction": "NG", "confidence": 0.9, "bbox": [1, 2]})
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "det", prediction))
        assert record.box_count == len(prediction["detections"]) - 1
        assert extractor.quality.parse_error_count == 1


class TestExtractorBookkeeping:
    def test_model_repeated_in_one_document(self):
        rng = random.Random(4)
        extractor = RecordExtractor()
        extractor.add(make_row(1, START, "cls", cls_prediction(rng, 0.9, 0.0), entry_index=0))
        extractor.add(make_row(1, START, "cls", cls_prediction(rng, 0.9, 0.0), entry_index=2))

        assert len(extractor.records) == 2
        assert extractor.quality.matched_document_count == 1
        assert extractor.quality.matched_entry_count == 2

    def test_malformed_row_is_counted_not_raised(self):
        extractor = RecordExtractor()
        row = make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0))
        row["createdAt"] = "not a date"
        assert extractor.add(row) is None
        assert extractor.quality.parse_error_count == 1
        assert extractor.quality.parse_error_examples == [row["_id"]]
        assert extractor.records == []

    def test_segmentation_rejected(self):
        row = make_row(1, START, "seg", {"predictionId": 1, "mask": "m.json"})
        with pytest.raises(UnsupportedTaskError, match="not supported"):
            RecordExtractor().add(row)

    def test_requested_task_filters_rows(self):
        extractor = RecordExtractor(task="det")
        assert extractor.add(make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0))) is None
        assert extractor.records == []
        assert extractor.quality.matched_document_count == 0

    def test_unknown_class_and_schema_version_warn_once(self):
        prediction = cls_prediction(random.Random(1), 0.9, 0.0)
        prediction["prediction"] = "Scratch"
        row = make_row(1, START, "cls", prediction)
        row["schemaVersion"] = "2.0"
        extractor = RecordExtractor()
        extractor.add(row)
        extractor.add(dict(row, _id="2"))
        assert sum("not in the model classes" in message for message in extractor.warnings) == 1
        assert sum("schema version" in message for message in extractor.warnings) == 1

    def test_resolve_task(self):
        rng = random.Random(1)
        extractor = RecordExtractor()
        assert extractor.resolve_task() == "cls"

        extractor.add(make_row(1, START, "det", det_prediction(rng, 0.9, 0.0)))
        assert extractor.resolve_task() == "det"
        assert extractor.resolve_task("cls") == "cls"

        extractor.add(make_row(2, START, "cls", cls_prediction(rng, 0.9, 0.0)))
        with pytest.raises(ValueError, match="multiple tasks"):
            extractor.resolve_task()
        with pytest.raises(ValueError, match="not supported"):
            extractor.resolve_task("seg")
