import math
import random
import pytest
from datetime import datetime, timezone

from mongodb_mcp.drift.extract import RecordExtractor, UnsupportedTaskError, normalize_confidence
from tests.drift.synthetic import make_row, cls_prediction, det_prediction, START


class TestNormalizeConfidence:
    def test_list_takes_maximum(self):
        conf_max, margin, entropy = normalize_confidence([0.05, 0.95])
        assert conf_max == 0.95
        assert margin == pytest.approx(0.9)
        assert entropy == pytest.approx(-(0.95 * math.log(0.95) + 0.05 * math.log(0.05)))

    def test_scalar(self):
        assert normalize_confidence(0.7) == (0.7, None, None)
        assert normalize_confidence(1) == (1.0, None, None)

    def test_missing(self):
        assert normalize_confidence([]) == (None, None, None)
        assert normalize_confidence(None) == (None, None, None)
        assert normalize_confidence("high") == (None, None, None)
        assert normalize_confidence([True]) == (None, None, None)

    def test_single_element_list(self):
        assert normalize_confidence([0.8]) == (0.8, None, pytest.approx(0.0))


class TestClassificationExtraction:
    def test_record_fields(self):
        rng = random.Random(1)
        prediction = cls_prediction(rng, 0.9, 0.0)
        prediction["confidence"] = [0.82, 0.18]
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "cls", prediction))

        assert record is not None
        assert record.prediction == "Good"
        assert record.conf_max == 0.82
        assert record.below_threshold is False
        assert record.near_threshold is True
        assert record.patch_w == 66.0 and record.patch_h == 40.0
        assert record.image_spec == (400, 400, 3)
        assert record.local_created_at.hour == 9
        assert record.local_created_at.utcoffset().total_seconds() == 9 * 3600
        assert extractor.tasks_seen == {"cls"}
        assert extractor.classes_seen == ["Good", "NG"]

    def test_null_threshold(self):
        rng = random.Random(1)
        record = RecordExtractor().add(make_row(1, START, "cls", cls_prediction(rng, 0.9, 0.0, threshold=None)))
        assert record.below_threshold is None
        assert record.near_threshold is None

    def test_missing_confidence_is_counted(self):
        rng = random.Random(1)
        prediction = cls_prediction(rng, 0.9, 0.0)
        prediction["confidence"] = []
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "cls", prediction))
        assert record.conf_max is None
        assert extractor.quality.n_missing_confidence == 1

    def test_feedback_label_and_comment(self):
        rng = random.Random(1)
        labelled = cls_prediction(rng, 0.9, 0.0, feedback=" ng ")
        commented = cls_prediction(rng, 0.9, 0.0, feedback="looks blurry")
        extractor = RecordExtractor()
        first = extractor.add(make_row(1, START, "cls", labelled))
        second = extractor.add(make_row(2, START, "cls", commented))

        assert first.feedback_label == "NG" and first.feedback_mismatch is True and first.feedback_other is False
        assert second.feedback_label is None and second.feedback_mismatch is None and second.feedback_other is True

    def test_latest_feedback_wins(self):
        rng = random.Random(1)
        prediction = cls_prediction(rng, 0.9, 0.0)
        prediction["feedbacks"] = [
            {"feedback": "NG", "registeredAt": datetime(2026, 9, 2, tzinfo=timezone.utc)},
            {"feedback": "Good", "registeredAt": datetime(2026, 9, 3, tzinfo=timezone.utc)}
        ]
        record = RecordExtractor().add(make_row(1, START, "cls", prediction))
        assert record.feedback_label == "Good" and record.feedback_mismatch is False


class TestDetectionExtraction:
    def test_list_confidence_uses_maximum(self):
        rng = random.Random(2)
        prediction = det_prediction(rng, 0.9, 0.0, boxes_mean=2.0, no_box_rate=0.0, list_confidence=True)
        record = RecordExtractor().add(make_row(1, START, "det", prediction))

        assert record.n_boxes == len(prediction["detections"])
        for box, detection in zip(record.boxes, prediction["detections"]):
            assert box.conf_max == max(detection["confidence"])

    def test_geometry_and_normalisation(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["detections"] = [{
            "bboxId": 0, "prediction": "NG", "confidence": 0.7, "threshold": 0.8,
            "bbox": [100.0, 100.0, 200.0, 150.0], "feedbacks": []
        }]
        record = RecordExtractor().add(make_row(1, START, "det", prediction))
        box = record.boxes[0]

        assert (box.w, box.h, box.area, box.aspect) == (100.0, 50.0, 5000.0, 2.0)
        assert (box.cx, box.cy) == (150.0, 125.0)
        assert box.w_norm == 0.25 and box.h_norm == 0.125
        assert box.area_norm == pytest.approx(5000 / 160000)
        assert box.cx_norm == 0.375 and box.cy_norm == 0.3125
        assert box.below_threshold is True
        assert record.threshold == 0.8
        assert record.n_boxes_by_class == {"Good": 0, "NG": 1}
        assert record.n_below_threshold == 1
        assert record.min_conf == 0.7

    def test_file_index_beyond_data_spec(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["fileIndex"] = 4
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "det", prediction))

        assert record.image_spec is None
        assert all(box.area_norm is None for box in record.boxes)
        assert extractor.quality.n_missing_image_spec == 1

    def test_no_boxes(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=1.0)
        record = RecordExtractor().add(make_row(1, START, "det", prediction))
        assert record.has_no_boxes and record.mean_conf is None and record.threshold is None

    def test_malformed_box_is_skipped(self):
        prediction = det_prediction(random.Random(3), 0.9, 0.0, no_box_rate=0.0)
        prediction["detections"].append({"prediction": "NG", "confidence": 0.9, "bbox": [1, 2]})
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "det", prediction))
        assert record.n_boxes == len(prediction["detections"]) - 1
        assert extractor.quality.n_parse_errors == 1


class TestExtractorBookkeeping:
    def test_model_repeated_in_one_document(self):
        rng = random.Random(4)
        extractor = RecordExtractor()
        extractor.add(make_row(1, START, "cls", cls_prediction(rng, 0.9, 0.0), entry_index=0))
        extractor.add(make_row(1, START, "cls", cls_prediction(rng, 0.9, 0.0), entry_index=2))

        assert len(extractor.records) == 2
        quality = extractor.quality.as_dict(2, None)
        assert quality["n_docs_matched"] == 1
        assert quality["n_entries_matched"] == 2

    def test_malformed_row_is_counted_not_raised(self):
        extractor = RecordExtractor()
        row = make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0))
        row["createdAt"] = "not a date"
        assert extractor.add(row) is None
        assert extractor.quality.n_parse_errors == 1
        assert extractor.quality.parse_error_examples == [row["_id"]]

    def test_segmentation_rejected(self):
        row = make_row(1, START, "seg", {"predictionId": 1, "mask": "m.json"})
        with pytest.raises(UnsupportedTaskError):
            RecordExtractor().add(row)

    def test_requested_task_filters_rows(self):
        extractor = RecordExtractor(task="det")
        assert extractor.add(make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0))) is None
        assert extractor.records == []

    def test_unknown_timezone_falls_back_to_utc(self):
        extractor = RecordExtractor()
        record = extractor.add(make_row(1, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0),
                                        timezone_name="Mars/Olympus"))
        assert record.local_created_at == START
        assert any("Unknown timezone" in message for message in extractor.warnings)

        extractor.add(make_row(2, START, "cls", cls_prediction(random.Random(1), 0.9, 0.0), timezone_name=None))
        assert any("no local timezone" in message for message in extractor.warnings)

    def test_unknown_class_and_schema_version_warn(self):
        prediction = cls_prediction(random.Random(1), 0.9, 0.0)
        prediction["prediction"] = "Scratch"
        row = make_row(1, START, "cls", prediction)
        row["schemaVersion"] = "2.0"
        extractor = RecordExtractor()
        extractor.add(row)
        assert any("not in the model classes" in message for message in extractor.warnings)
        assert any("schema version" in message for message in extractor.warnings)
