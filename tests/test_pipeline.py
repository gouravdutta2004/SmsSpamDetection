import unittest
import os
import csv
import json
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import clean_text


class TestSpamPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_path = os.path.join(cls.base_dir, "data", "SMSSpamCollection")
        cls.pipeline_path = os.path.join(cls.base_dir, "models", "pipeline.joblib")
        cls.metrics_path = os.path.join(cls.base_dir, "models", "metrics.json")

    def test_clean_text_basic(self):
        self.assertEqual(clean_text("Hello, WORLD!!! 123"), "hello world 123")
        self.assertEqual(clean_text("  multiple   spaces  "), "multiple spaces")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(123), "")

    def test_dataset_loading_full_5574_records(self):
        self.assertTrue(os.path.exists(self.data_path), "Dataset file SMSSpamCollection does not exist.")
        df = pd.read_csv(self.data_path, sep="\t", names=["label", "message"], quoting=csv.QUOTE_NONE)
        self.assertEqual(len(df), 5574, f"Expected 5574 samples, got {len(df)}")
        self.assertEqual((df["label"] == "spam").sum(), 747, "Expected 747 spam messages")
        self.assertEqual((df["label"] == "ham").sum(), 4827, "Expected 4827 ham messages")

    def test_pipeline_artifact_and_inference(self):
        self.assertTrue(os.path.exists(self.pipeline_path), "Trained pipeline.joblib does not exist.")
        pipeline = joblib.load(self.pipeline_path)

        spam_sample = clean_text("WINNER!! You have won a $1,000 Walmart gift card! Call 800-123-4567 NOW")
        ham_sample = clean_text("Hey mate, are we still meeting for dinner at 7pm tonight?")

        preds = pipeline.predict([spam_sample, ham_sample])
        self.assertEqual(preds[0], 1, "Expected spam prediction for prize winning text")
        self.assertEqual(preds[1], 0, "Expected ham prediction for dinner meeting text")

        probs = pipeline.predict_proba([spam_sample, ham_sample])
        self.assertEqual(probs.shape, (2, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        self.assertAlmostEqual(probs[0].sum(), 1.0, places=5)
        self.assertAlmostEqual(probs[1].sum(), 1.0, places=5)
        self.assertGreater(probs[0, 1], probs[1, 1], "Spam probability should be higher for spam message")

    def test_metrics_json_schema(self):
        self.assertTrue(os.path.exists(self.metrics_path), "metrics.json does not exist.")
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertIn("best_model", metrics)
        self.assertIn("models", metrics)
        self.assertIn("test_metrics", metrics)
        self.assertEqual(metrics["total_samples"], 5574)
        self.assertEqual(metrics["spam_count"], 747)
        self.assertEqual(metrics["ham_count"], 4827)

        for model_name, model_metrics in metrics["models"].items():
            self.assertIn("cv_accuracy", model_metrics)
            self.assertIn("cv_precision", model_metrics)
            self.assertIn("cv_recall", model_metrics)
            self.assertIn("cv_f1_score", model_metrics)


if __name__ == "__main__":
    unittest.main()
