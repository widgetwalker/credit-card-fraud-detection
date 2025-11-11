import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from demo_complete import (
    DataValidator,
    MockFraudDetector,
    demonstrate_data_loading,
    demonstrate_data_preprocessing,
    demonstrate_fraud_detection
)

class TestDataValidator:

    def test_validator_initialization(self):

        validator = DataValidator()
        assert isinstance(validator, DataValidator)

    def test_validate_transaction_valid(self):

        validator = DataValidator()

        valid_transaction = {}
        for i in range(1, 29):
            valid_transaction[f'V{i}'] = 0.5
        valid_transaction['Amount'] = 100.0
        valid_transaction['Time'] = 12345

        is_valid, message = validator.validate_transaction(valid_transaction)
        assert is_valid == True
        assert "Valid transaction" in message

    def test_validate_transaction_missing_features(self):

        validator = DataValidator()

        invalid_transaction = {
            'V1': 0.5,
            'V2': -0.3,
            'Amount': 100.0

        }

        is_valid, message = validator.validate_transaction(invalid_transaction)
        assert is_valid == False
        assert "Missing features" in message

    def test_validate_transaction_invalid_values(self):

        validator = DataValidator()

        invalid_transaction = {}
        for i in range(1, 29):
            invalid_transaction[f'V{i}'] = None
        invalid_transaction['Amount'] = 100.0
        invalid_transaction['Time'] = 12345

        is_valid, message = validator.validate_transaction(invalid_transaction)
        assert is_valid == False
        assert "Invalid value" in message

    def test_validate_batch(self):

        validator = DataValidator()

        transactions = []
        for j in range(3):
            transaction = {}
            for i in range(1, 29):
                transaction[f'V{i}'] = 0.5 + j * 0.1
            transaction['Amount'] = 100.0 + j * 10
            transaction['Time'] = 12345 + j
            transactions.append(transaction)

        results = validator.validate_batch(transactions)

        assert isinstance(results, list)
        assert len(results) == 3

        for result in results:
            assert 'index' in result
            assert 'valid' in result
            assert 'message' in result
            assert result['valid'] == True

class TestMockFraudDetector:

    def test_detector_initialization(self):

        detector = MockFraudDetector()
        assert isinstance(detector, MockFraudDetector)
        assert detector.is_trained == True
        assert detector.model_name == "Fraud Detection Ensemble"
        assert detector.threshold == 0.5

    def test_predict_fraud_probability_normal_transaction(self):

        detector = MockFraudDetector()

        normal_transaction = {}
        for i in range(1, 29):
            normal_transaction[f'V{i}'] = 0.1
        normal_transaction['Amount'] = 50.0
        normal_transaction['Time'] = 12345

        fraud_prob = detector.predict_fraud_probability(normal_transaction)

        assert isinstance(fraud_prob, float)
        assert 0.0 <= fraud_prob <= 1.0

        assert fraud_prob < 0.5

    def test_predict_fraud_probability_suspicious_transaction(self):

        detector = MockFraudDetector()

        suspicious_transaction = {}
        for i in range(1, 29):
            suspicious_transaction[f'V{i}'] = 2.0 if i <= 5 else 0.1
        suspicious_transaction['Amount'] = 1500.0
        suspicious_transaction['Time'] = 12345

        fraud_prob = detector.predict_fraud_probability(suspicious_transaction)

        assert isinstance(fraud_prob, float)
        assert 0.0 <= fraud_prob <= 1.0

        assert fraud_prob > 0.3

    def test_classify_transaction(self):

        detector = MockFraudDetector()

        high_prob = 0.8
        classification = detector.classify_transaction(high_prob)
        assert classification == 1

        low_prob = 0.2
        classification = detector.classify_transaction(low_prob)
        assert classification == 0

        threshold_prob = 0.5
        classification = detector.classify_transaction(threshold_prob)
        assert classification == 1

    def test_explain_prediction(self):

        detector = MockFraudDetector()

        high_risk_transaction = {}
        for i in range(1, 29):
            high_risk_transaction[f'V{i}'] = 3.0 if i == 1 else 0.1
        high_risk_transaction['Amount'] = 1000.0
        high_risk_transaction['Time'] = 12345

        fraud_prob = 0.8
        explanations = detector.explain_prediction(high_risk_transaction, fraud_prob)

        assert isinstance(explanations, list)

        assert len(explanations) > 0

        amount_explanation = any("High transaction amount" in exp for exp in explanations)
        assert amount_explanation == True

class TestDataProcessing:

    def test_demonstrate_data_loading(self):

        try:
            df = demonstrate_data_loading(test_mode=True)

            assert isinstance(df, pd.DataFrame)

            expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']
            assert all(col in df.columns for col in expected_columns)
        except Exception as e:

            pytest.fail(f"Unexpected exception: {e}")

    def test_demonstrate_data_preprocessing(self):

        sample_data = pd.DataFrame({
            'V1': [0.1, 0.2, 0.3],
            'V2': [-0.1, -0.2, -0.3],
            'V3': [0.05, 0.1, 0.15],
            'Amount': [100.0, 200.0, 300.0],
            'Class': [0, 1, 0]
        })

        try:
            X_processed, y = demonstrate_data_preprocessing(sample_data)
            assert isinstance(X_processed, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert len(X_processed) == len(sample_data)
        except Exception as e:

            pytest.fail(f"Preprocessing failed: {e}")

    def test_demonstrate_fraud_detection(self):

        sample_data = pd.DataFrame({
            'V1': [0.1, -1.5, 0.3],
            'V2': [-0.1, -2.0, -0.3],
            'V3': [0.05, 1.5, 0.15],
            'V4': [0.02, 0.8, 0.1],
            'V5': [-0.05, 0.3, -0.02],
            'V6': [0.03, -0.4, 0.2],
            'V7': [-0.02, 0.7, -0.1],
            'V8': [0.04, 0.6, 0.05],
            'V9': [-0.03, -0.5, 0.08],
            'V10': [0.01, 0.9, -0.15],
            'V11': [-0.04, 0.4, 0.12],
            'V12': [0.02, -0.8, -0.05],
            'V13': [-0.01, 0.5, 0.07],
            'V14': [0.05, -0.6, -0.08],
            'V15': [-0.02, 0.8, 0.03],
            'V16': [0.03, -0.7, 0.09],
            'V17': [-0.05, 0.6, -0.04],
            'V18': [0.01, -0.9, 0.06],
            'V19': [-0.03, 0.7, -0.07],
            'V20': [0.04, -0.5, 0.02],
            'V21': [-0.01, 0.8, -0.06],
            'V22': [0.02, -0.4, 0.1],
            'V23': [-0.04, 0.5, -0.03],
            'V24': [0.05, -0.6, 0.08],
            'V25': [-0.02, 0.9, -0.09],
            'V26': [0.03, -0.7, 0.04],
            'V27': [-0.05, 0.6, -0.02],
            'V28': [0.01, -0.8, 0.05],
            'Amount': [100.0, 1500.0, 300.0],
            'Class': [0, 1, 0]
        })

        sample_y = pd.Series([0, 1, 0])

        try:
            fraud_predictions = demonstrate_fraud_detection(sample_data, sample_y)
            assert isinstance(fraud_predictions, list)
            assert len(fraud_predictions) > 0

            for pred in fraud_predictions:
                assert 'transaction_id' in pred
                assert 'fraud_probability' in pred
                assert 'is_fraud' in pred
                assert 'amount' in pred
                assert 'explanations' in pred
        except Exception as e:

            pytest.fail(f"Fraud detection failed: {e}")

class TestEdgeCases:

    def test_validator_empty_transaction(self):

        validator = DataValidator()

        empty_transaction = {}
        is_valid, message = validator.validate_transaction(empty_transaction)
        assert is_valid == False
        assert "Missing features" in message

    def test_detector_empty_transaction(self):

        detector = MockFraudDetector()

        empty_transaction = {}

        try:
            fraud_prob = detector.predict_fraud_probability(empty_transaction)

            assert 0.0 <= fraud_prob <= 1.0
        except Exception as e:

            assert isinstance(e, (KeyError, ValueError))

    def test_detector_extreme_values(self):

        detector = MockFraudDetector()

        extreme_transaction = {}
        for i in range(1, 29):
            extreme_transaction[f'V{i}'] = 1000.0 if i <= 5 else 0.1
        extreme_transaction['Amount'] = 1000000.0
        extreme_transaction['Time'] = 999999

        fraud_prob = detector.predict_fraud_probability(extreme_transaction)

        assert isinstance(fraud_prob, float)
        assert 0.0 <= fraud_prob <= 1.0

        assert fraud_prob > 0.5

    def test_detector_zero_values(self):

        detector = MockFraudDetector()

        zero_transaction = {}
        for i in range(1, 29):
            zero_transaction[f'V{i}'] = 0.0
        zero_transaction['Amount'] = 0.0
        zero_transaction['Time'] = 0

        fraud_prob = detector.predict_fraud_probability(zero_transaction)

        assert isinstance(fraud_prob, float)
        assert 0.0 <= fraud_prob <= 1.0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])