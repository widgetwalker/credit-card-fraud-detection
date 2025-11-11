import pytest
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from demo_complete import (
    DataValidator,
    MockFraudDetector,
    demonstrate_data_loading,
    demonstrate_data_preprocessing,
    demonstrate_transaction_validation,
    demonstrate_fraud_detection,
    demonstrate_error_handling,
    demonstrate_performance_metrics,
    demonstrate_batch_processing
)

class TestEndToEndWorkflow:

    def test_complete_fraud_detection_workflow(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        test_transaction = {}
        for i in range(1, 29):
            test_transaction[f'V{i}'] = 0.1 if i <= 5 else 0.01
        test_transaction['Amount'] = 100.0
        test_transaction['Time'] = 12345

        is_valid, validation_message = validator.validate_transaction(test_transaction)
        assert is_valid == True, f"Transaction should be valid: {validation_message}"

        fraud_probability = detector.predict_fraud_probability(test_transaction)
        assert 0.0 <= fraud_probability <= 1.0, "Fraud probability should be between 0 and 1"

        classification = detector.classify_transaction(fraud_probability)
        assert classification in [0, 1], "Classification should be 0 (legitimate) or 1 (fraud)"

        explanations = detector.explain_prediction(test_transaction, fraud_probability)
        assert isinstance(explanations, list), "Explanations should be a list"

    def test_data_pipeline_integration(self):

        df = demonstrate_data_loading(test_mode=True)

        assert df is not None
        assert len(df) > 0

        X_processed, y = demonstrate_data_preprocessing(df)
        assert isinstance(X_processed, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X_processed) == len(y)

        fraud_predictions = demonstrate_fraud_detection(X_processed, y)
        assert isinstance(fraud_predictions, list)
        assert len(fraud_predictions) > 0

    def test_system_components_integration(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        transactions = []
        for j in range(5):
            transaction = {}
            for i in range(1, 29):
                transaction[f'V{i}'] = 0.5 + j * 0.1
            transaction['Amount'] = 100.0 + j * 50
            transaction['Time'] = 12345 + j
            transactions.append(transaction)

        validation_results = validator.validate_batch(transactions)
        assert len(validation_results) == len(transactions)

        valid_transactions = [
            transactions[i] for i, result in enumerate(validation_results)
            if result['valid']
        ]

        for transaction in valid_transactions:
            fraud_prob = detector.predict_fraud_probability(transaction)
            classification = detector.classify_transaction(fraud_prob)
            explanations = detector.explain_prediction(transaction, fraud_prob)

            assert isinstance(fraud_prob, float)
            assert classification in [0, 1]
            assert isinstance(explanations, list)

class TestSystemPerformance:

    def test_single_transaction_performance(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        test_transaction = {}
        for i in range(1, 29):
            test_transaction[f'V{i}'] = 0.1
        test_transaction['Amount'] = 100.0
        test_transaction['Time'] = 12345

        start_time = time.time()
        is_valid, _ = validator.validate_transaction(test_transaction)
        validation_time = time.time() - start_time

        start_time = time.time()
        fraud_prob = detector.predict_fraud_probability(test_transaction)
        detection_time = time.time() - start_time

        assert validation_time < 0.1, f"Validation too slow: {validation_time}s"
        assert detection_time < 0.1, f"Detection too slow: {detection_time}s"
        assert is_valid == True
        assert 0.0 <= fraud_prob <= 1.0

    def test_batch_processing_performance(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        batch_size = 100
        transactions = []
        for j in range(batch_size):
            transaction = {}
            for i in range(1, 29):
                transaction[f'V{i}'] = 0.1 + (j % 10) * 0.01
            transaction['Amount'] = 100.0 + (j % 5) * 50
            transaction['Time'] = 12345 + j
            transactions.append(transaction)

        start_time = time.time()
        validation_results = validator.validate_batch(transactions)
        batch_validation_time = time.time() - start_time

        start_time = time.time()
        fraud_probabilities = []
        for transaction in transactions:
            fraud_prob = detector.predict_fraud_probability(transaction)
            fraud_probabilities.append(fraud_prob)
        batch_detection_time = time.time() - start_time

        assert len(validation_results) == batch_size
        assert len(fraud_probabilities) == batch_size
        assert batch_validation_time < 1.0, f"Batch validation too slow: {batch_validation_time}s"
        assert batch_detection_time < 2.0, f"Batch detection too slow: {batch_detection_time}s"

    def test_concurrent_transaction_processing(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        transactions = []
        for j in range(10):
            transaction = {}
            for i in range(1, 29):
                transaction[f'V{i}'] = 0.1 + j * 0.05
            transaction['Amount'] = 100.0 + j * 20
            transaction['Time'] = 12345 + j
            transactions.append(transaction)

        results = []
        for transaction in transactions:
            is_valid, _ = validator.validate_transaction(transaction)
            if is_valid:
                fraud_prob = detector.predict_fraud_probability(transaction)
                classification = detector.classify_transaction(fraud_prob)
                results.append({
                    'valid': is_valid,
                    'fraud_probability': fraud_prob,
                    'classification': classification
                })

        assert len(results) == len(transactions)

        for result in results:
            assert result['valid'] == True
            assert 0.0 <= result['fraud_probability'] <= 1.0
            assert result['classification'] in [0, 1]

class TestErrorHandling:

    def test_invalid_transaction_handling(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        empty_transaction = {}
        is_valid, message = validator.validate_transaction(empty_transaction)
        assert is_valid == False
        assert "Missing features" in message

        none_transaction = {}
        for i in range(1, 29):
            none_transaction[f'V{i}'] = None
        none_transaction['Amount'] = None
        none_transaction['Time'] = None

        is_valid, message = validator.validate_transaction(none_transaction)
        assert is_valid == False
        assert "Invalid value" in message

    def test_missing_fields_handling(self):

        validator = DataValidator()

        incomplete_transaction = {
            'V1': 0.1,
            'V2': -0.1,
            'V3': 0.05,
            'Amount': 100.0,
            'Time': 12345

        }

        is_valid, message = validator.validate_transaction(incomplete_transaction)
        assert is_valid == False
        assert "Missing features" in message

    def test_empty_batch_handling(self):

        validator = DataValidator()

        empty_batch = []
        results = validator.validate_batch(empty_batch)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_malformed_data_handling(self):

        validator = DataValidator()

        malformed_transaction = {
            'V1': "invalid_string",
            'V2': [1, 2, 3],
            'V3': {"nested": "dict"},
            'Amount': "not_a_number",
            'Time': "not_a_time"
        }

        is_valid, message = validator.validate_transaction(malformed_transaction)
        assert is_valid == False
        assert ("Invalid value" in message) or ("Missing features" in message)

class TestRealWorldScenarios:

    def test_high_value_transaction_scenario(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        high_value_transaction = {}
        for i in range(1, 29):
            high_value_transaction[f'V{i}'] = 0.5 if i <= 3 else 0.1
        high_value_transaction['Amount'] = 5000.0
        high_value_transaction['Time'] = 12345

        is_valid, _ = validator.validate_transaction(high_value_transaction)
        assert is_valid == True

        fraud_prob = detector.predict_fraud_probability(high_value_transaction)
        classification = detector.classify_transaction(fraud_prob)
        explanations = detector.explain_prediction(high_value_transaction, fraud_prob)

        assert fraud_prob > 0.3
        assert isinstance(explanations, list)

        amount_mentioned = any("High transaction amount" in exp for exp in explanations)
        assert amount_mentioned == True

    def test_multiple_small_transactions_scenario(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        small_transactions = []
        for j in range(10):
            transaction = {}
            for i in range(1, 29):
                transaction[f'V{i}'] = 0.05
            transaction['Amount'] = 10.0 + j * 2
            transaction['Time'] = 12345 + j * 60
            small_transactions.append(transaction)

        validation_results = validator.validate_batch(small_transactions)
        assert len(validation_results) == len(small_transactions)

        valid_count = sum(1 for result in validation_results if result['valid'])
        assert valid_count == len(small_transactions)

        fraud_results = []
        for transaction in small_transactions:
            fraud_prob = detector.predict_fraud_probability(transaction)
            classification = detector.classify_transaction(fraud_prob)
            fraud_results.append({
                'fraud_probability': fraud_prob,
                'classification': classification
            })

        assert len(fraud_results) == len(small_transactions)
        for result in fraud_results:
            assert 0.0 <= result['fraud_probability'] <= 1.0
            assert result['classification'] in [0, 1]

    def test_rapid_transaction_sequence_scenario(self):

        validator = DataValidator()
        detector = MockFraudDetector()

        rapid_transactions = []
        for j in range(5):
            transaction = {}
            for i in range(1, 29):
                transaction[f'V{i}'] = 0.8 if i <= 2 else 0.1
            transaction['Amount'] = 200.0
            transaction['Time'] = 12345 + j * 5
            rapid_transactions.append(transaction)

        validation_results = validator.validate_batch(rapid_transactions)
        assert len(validation_results) == len(rapid_transactions)

        fraud_probabilities = []
        for transaction in rapid_transactions:
            fraud_prob = detector.predict_fraud_probability(transaction)
            fraud_probabilities.append(fraud_prob)

        assert len(fraud_probabilities) == len(rapid_transactions)
        for prob in fraud_probabilities:
            assert 0.0 <= prob <= 1.0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])