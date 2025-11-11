import argparse
import logging
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import joblib

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.preprocessing import DataPreprocessor
from src.models.fraud_models import FraudDetectionModel
from src.utils.config import ConfigManager
from src.utils.error_handling import setup_logging

def setup_argument_parser():

    parser = argparse.ArgumentParser(description='Run fraud detection predictions')

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to input data (CSV file)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model file'
    )

    parser.add_argument(
        '--preprocessor-path',
        type=str,
        required=True,
        help='Path to preprocessor file'
    )

    parser.add_argument(
        '--config-path',
        type=str,
        default='config/model_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default='predictions_output.csv',
        help='Path to save predictions'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold'
    )

    parser.add_argument(
        '--return-probabilities',
        action='store_true',
        help='Return prediction probabilities'
    )

    parser.add_argument(
        '--return-features',
        action='store_true',
        help='Return processed features'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for prediction'
    )

    parser.add_argument(
        '--api-endpoint',
        type=str,
        help='API endpoint for prediction (alternative to local model)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for authentication'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser

def load_model_and_preprocessor(model_path, preprocessor_path):

    logging.info("Loading model and preprocessor")

    try:

        model = FraudDetectionModel.load_model(model_path)
        logging.info(f"Model loaded from {model_path}")

        preprocessor = DataPreprocessor.load_pipeline(preprocessor_path)
        logging.info(f"Preprocessor loaded from {preprocessor_path}")

        return model, preprocessor

    except Exception as e:
        logging.error(f"Failed to load model or preprocessor: {e}")
        raise

def load_input_data(data_path):

    logging.info(f"Loading input data from {data_path}")

    try:
        data = pd.read_csv(data_path)
        logging.info(f"Loaded {len(data)} samples with {len(data.columns)} features")
        return data

    except Exception as e:
        logging.error(f"Failed to load input data: {e}")
        raise

def preprocess_input_data(data, preprocessor, config):

    logging.info("Preprocessing input data")

    try:

        target_column = config.data.target_column
        has_target = target_column in data.columns

        if has_target:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            X = data
            y = None

        X_processed = preprocessor.transform(X)

        logging.info(f"Preprocessing completed. Shape: {X_processed.shape}")
        return X_processed, y, has_target

    except Exception as e:
        logging.error(f"Failed to preprocess input data: {e}")
        raise

def predict_with_model(model, X, threshold, return_probabilities=True):

    logging.info("Making predictions with local model")

    try:

        y_proba = model.predict_proba(X)

        y_pred = (y_proba[:, 1] >= threshold).astype(int)

        results = {
            'predictions': y_pred,
            'fraud_probability': y_proba[:, 1]
        }

        if return_probabilities:
            results['probabilities'] = y_proba

        logging.info(f"Predictions completed for {len(y_pred)} samples")
        return results

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise

def predict_with_api(data, api_endpoint, api_key=None, threshold=0.5,
                    return_probabilities=True, batch_size=1000):

    logging.info("Making predictions using API")

    try:
        import requests

        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        if isinstance(data, pd.DataFrame):
            data_list = data.to_dict('records')
        else:
            data_list = data.tolist() if hasattr(data, 'tolist') else data

        all_predictions = []
        total_samples = len(data_list)

        for i in range(0, total_samples, batch_size):
            batch = data_list[i:i + batch_size]

            payload = {
                'transactions': batch,
                'threshold': threshold,
                'return_probabilities': return_probabilities
            }

            response = requests.post(
                f"{api_endpoint}/predict/batch",
                json=payload,
                headers=headers,
                timeout=300
            )

            if response.status_code != 200:
                raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")

            batch_results = response.json()
            all_predictions.extend(batch_results['predictions'])

            logging.info(f"Processed batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1}")

        logging.info(f"API predictions completed for {len(all_predictions)} samples")
        return all_predictions

    except Exception as e:
        logging.error(f"API prediction failed: {e}")
        raise

def save_predictions(results, output_path, input_data, processed_features=None,
                    return_features=False, has_target=False):

    logging.info(f"Saving predictions to {output_path}")

    try:

        output_df = input_data.copy()

        if isinstance(results, dict):

            output_df['prediction'] = results['predictions']
            output_df['fraud_probability'] = results['fraud_probability']

            if 'probabilities' in results:
                output_df['probability_class_0'] = results['probabilities'][:, 0]
                output_df['probability_class_1'] = results['probabilities'][:, 1]
        else:

            predictions = results
            output_df['prediction'] = [p['prediction'] for p in predictions]
            output_df['fraud_probability'] = [p['fraud_probability'] for p in predictions]

            if 'probabilities' in predictions[0]:
                output_df['probability_class_0'] = [p['probabilities'][0] for p in predictions]
                output_df['probability_class_1'] = [p['probabilities'][1] for p in predictions]

        if return_features and processed_features is not None:
            feature_df = pd.DataFrame(
                processed_features,
                columns=[f'feature_{i}' for i in range(processed_features.shape[1])]
            )
            output_df = pd.concat([output_df, feature_df], axis=1)

        output_df.to_csv(output_path, index=False)

        summary_stats = {
            'total_transactions': len(output_df),
            'fraud_predictions': int(output_df['prediction'].sum()),
            'fraud_rate': float(output_df['prediction'].mean()),
            'avg_fraud_probability': float(output_df['fraud_probability'].mean()),
            'min_fraud_probability': float(output_df['fraud_probability'].min()),
            'max_fraud_probability': float(output_df['fraud_probability'].max())
        }

        summary_path = str(Path(output_path).with_suffix('')) + '_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)

        logging.info(f"Predictions saved to {output_path}")
        logging.info(f"Summary statistics saved to {summary_path}")

        logging.info("Prediction Summary:")
        logging.info(f"  Total transactions: {summary_stats['total_transactions']}")
        logging.info(f"  Fraud predictions: {summary_stats['fraud_predictions']}")
        logging.info(f"  Fraud rate: {summary_stats['fraud_rate']:.2%}")
        logging.info(f"  Avg fraud probability: {summary_stats['avg_fraud_probability']:.4f}")

        return summary_stats

    except Exception as e:
        logging.error(f"Failed to save predictions: {e}")
        raise

def evaluate_predictions(y_true, y_pred, y_proba):

    logging.info("Evaluating predictions against ground truth")

    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1_score': float(f1_score(y_true, y_pred)),
            'auc_score': float(roc_auc_score(y_true, y_proba))
        }

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics['confusion_matrix'] = {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }

        logging.info("Evaluation completed:")
        logging.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logging.info(f"  AUC Score: {metrics['auc_score']:.4f}")

        return metrics

    except Exception as e:
        logging.error(f"Failed to evaluate predictions: {e}")
        return None

def main():

    parser = setup_argument_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    logging.info("Starting fraud detection predictions")

    try:

        config_manager = ConfigManager()
        config_manager.load_from_file(args.config_path)
        config = config_manager.get_config()

        input_data = load_input_data(args.data_path)

        if args.api_endpoint:

            results = predict_with_api(
                input_data, args.api_endpoint, args.api_key,
                args.threshold, args.return_probabilities, args.batch_size
            )
            processed_features = None
        else:

            model, preprocessor = load_model_and_preprocessor(
                args.model_path, args.preprocessor_path
            )

            processed_features, y_true, has_target = preprocess_input_data(
                input_data, preprocessor, config
            )

            results = predict_with_model(
                model, processed_features, args.threshold, args.return_probabilities
            )

        summary_stats = save_predictions(
            results, args.output_path, input_data, processed_features,
            args.return_features, has_target if 'has_target' in locals() else False
        )

        if 'y_true' in locals() and y_true is not None:
            evaluation_metrics = evaluate_predictions(
                y_true, results['predictions'], results['fraud_probability']
            )

            if evaluation_metrics:

                eval_path = str(Path(args.output_path).with_suffix('')) + '_evaluation.json'
                with open(eval_path, 'w') as f:
                    json.dump(evaluation_metrics, f, indent=2)

                logging.info(f"Evaluation results saved to {eval_path}")

        logging.info("Predictions completed successfully!")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()