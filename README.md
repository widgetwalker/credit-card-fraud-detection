# Credit Card Fraud  Detection System

A comprehensive machine learning system for detecting credit card fraud using advanced algorithms and real-time prediction capabilities.

## 🚀 Features

- **Multiple ML Models**: Random Forest, XGBoost, Logistic Regression, LightGBM, Gradient Boosting, and Neural Networks
- **Real-time API**: FastAPI-based REST API for real-time fraud predictions
- **Comprehensive Evaluation**: Advanced metrics including fraud-specific measures and cost analysis
- **Robust Error Handling**: Centralized error management and logging
- **Automated Testing**: Unit, integration, and performance tests
- **Production Deployment**: Docker and Kubernetes deployment configurations
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Batch Processing**: Support for both real-time and batch predictions

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- kubectl (for Kubernetes deployment)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/widgetwalker/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

## 🚀 Quick Start

### 1. Prepare your data

Ensure your data is in CSV format with the following structure:
```csv
Time,V1,V2,V3,...,V28,Amount,Class
0,-1.3598071336738,-0.0727811733098497,2.53634673796948,...,0.133558376740292,-0.0210530535080104,149.62,0
```

### 2. Train models

```bash
python scripts/train_model.py \
    --data-path data/creditcard.csv \
    --models random_forest xgboost logistic_regression \
    --use-ensemble \
    --cross-validation \
    --output-dir models/trained_models
```

### 3. Evaluate models

```bash
python scripts/evaluate_model.py \
    --data-path data/test_data.csv \
    --model-path models/trained_models/random_forest_*.pkl \
    --preprocessor-path models/trained_models/preprocessor_*.pkl \
    --generate-plots \
    --threshold-optimization
```

### 4. Run predictions

```bash
python scripts/predict.py \
    --data-path data/new_transactions.csv \
    --model-path models/trained_models/best_model.pkl \
    --preprocessor-path models/trained_models/preprocessor.pkl \
    --output-path predictions/output.csv
```

### 5. Start the API server

```bash
python -m src.api.main
```

Or using Docker:
```bash
docker-compose up -d
```

## 📁 Project Structure

```
credit-card-fraud-detection/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py          # Data preprocessing and feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   └── fraud_models.py         # ML model implementations
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              # Evaluation metrics and model comparison
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI application
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       └── error_handling.py       # Error handling utilities
├── scripts/
│   ├── train_model.py              # Model training script
│   ├── evaluate_model.py           # Model evaluation script
│   ├── predict.py                  # Prediction script
│   └── deploy.py                   # Deployment script
├── tests/
│   ├── conftest.py                 # Test configuration
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── performance/                # Performance tests
├── config/
│   ├── model_config.yaml           # Model configuration
│   └── deployment_config.yaml      # Deployment configuration
├── deployment/
│   ├── docker/                     # Docker deployment files
│   │   ├── Dockerfile
│   │   ├── docker-compose.yml
│   │   └── nginx.conf
│   └── kubernetes/                 # Kubernetes deployment files
│       ├── deployment.yaml
│       ├── service.yaml
│       ├── ingress.yaml
│       └── hpa.yaml
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
```

## 💻 Usage

### Data Preprocessing

The system includes comprehensive data preprocessing capabilities:

```python
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    handle_missing='median',
    scale_features=True,
    encode_categorical=True,
    feature_selection='mutual_info',
    k_best=20
)

X_processed = preprocessor.fit_transform(X, y)
```

### Model Training

Train multiple models with a single command:

```python
from src.models.fraud_models import ModelFactory

# Create and train a model
model = ModelFactory.create_model('xgboost')
model.train(X_train, y_train)

# Evaluate the model
results = model.evaluate(X_test, y_test)
print(f"AUC: {results['auc_score']}")
```

### Model Evaluation

Comprehensive evaluation with fraud-specific metrics:

```python
from src.evaluation.metrics import FraudDetectionMetrics

evaluator = FraudDetectionMetrics(cost_matrix=[1, 50, 10, 0])
metrics = evaluator.evaluate_basic_metrics(y_true, y_pred_proba)
fraud_metrics = evaluator.evaluate_fraud_specific_metrics(y_true, y_pred_proba)
```

### API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "Time": 0,
      "V1": -1.3598071336738,
      "V2": -0.0727811733098497,
      ...
    },
    "threshold": 0.5
  }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"Time": 0, "V1": -1.36, ...},
      {"Time": 1, "V1": 0.5, ...}
    ],
    "threshold": 0.5
  }'
```

## 📚 API Documentation

The API provides the following endpoints:

### Health Check
- **GET** `/health` - Check if the service is healthy

### Model Information
- **GET** `/model/info` - Get model information and parameters

### Predictions
- **POST** `/predict` - Single transaction prediction
- **POST** `/predict/batch` - Batch prediction for multiple transactions

### Model Management
- **POST** `/model/reload` - Reload the model (admin endpoint)

Full API documentation is available at `http://localhost:8000/docs` when the server is running.

## 🎯 Model Training

### Supported Models

- **Random Forest**: Ensemble method with good interpretability
- **XGBoost**: Gradient boosting with excellent performance
- **Logistic Regression**: Simple baseline model
- **LightGBM**: Fast gradient boosting
- **Gradient Boosting**: Scikit-learn implementation
- **Neural Network**: Deep learning approach

### Training Process

1. **Data Loading**: Load and validate training data
2. **Preprocessing**: Handle missing values, scaling, feature selection
3. **Model Training**: Train multiple models with cross-validation
4. **Evaluation**: Comprehensive evaluation with multiple metrics
5. **Ensemble Creation**: Combine best models for improved performance
6. **Model Saving**: Save trained models and preprocessing pipeline

### Training Script Options

```bash
python scripts/train_model.py --help
```

Key options:
- `--models`: Models to train (multiple choices)
- `--use-ensemble`: Create ensemble model
- `--cross-validation`: Perform cross-validation
- `--test-size`: Test set size (default: 0.2)
- `--output-dir`: Directory to save models

## 📊 Evaluation

### Evaluation Metrics

#### Basic Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Confusion Matrix

#### Fraud-Specific Metrics
- **Fraud Recall**: Recall for fraud cases
- **False Alarm Rate**: False positive rate
- **Cost-Weighted Error**: Business cost analysis
- **Precision-Recall Balance**: Optimized threshold finding

#### Advanced Analysis
- **Threshold Optimization**: Find optimal threshold
- **Cost Analysis**: Business impact analysis
- **Model Comparison**: Compare multiple models
- **Cross-Validation**: Robust performance estimation

### Evaluation Script Options

```bash
python scripts/evaluate_model.py --help
```

Key options:
- `--generate-plots`: Create evaluation plots
- `--threshold-optimization`: Find optimal threshold
- `--cost-matrix`: Custom cost matrix for analysis

## 🚀 Deployment

### Docker Deployment

1. **Build and start services**
   ```bash
   docker-compose up -d
   ```

2. **Check service status**
   ```bash
   docker-compose ps
   ```

3. **View logs**
   ```bash
   docker-compose logs -f
   ```

### Kubernetes Deployment

1. **Deploy to Kubernetes**
   ```bash
   python scripts/deploy.py --environment production --deployment-type kubernetes
   ```

2. **Check deployment status**
   ```bash
   kubectl get pods -n fraud-detection
   ```

3. **Access the service**
   ```bash
   kubectl get svc -n fraud-detection
   ```

### Deployment Script Options

```bash
python scripts/deploy.py --help
```

Key options:
- `--environment`: Deployment environment
- `--deployment-type`: Docker or Kubernetes
- `--skip-tests`: Skip pre-deployment tests
- `--dry-run`: Perform dry run

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run unit tests only
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run performance tests
python -m pytest tests/performance/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test system performance and scalability

## ⚙️ Configuration

### Model Configuration

Edit `config/model_config.yaml` to customize:
- Model parameters
- Preprocessing options
- Feature engineering settings
- Evaluation metrics

### Deployment Configuration

Edit `config/deployment_config.yaml` to customize:
- Environment settings
- Security configurations
- Monitoring options
- Resource allocations

## 🔧 Development

### Setting up Development Environment

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run code formatting**
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

4. **Run linting**
   ```bash
   flake8 src/ tests/
   mypy src/
   ```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance

### Benchmarks

Typical performance on standard hardware:
- **Training Time**: 2-5 minutes for 100K samples
- **Prediction Speed**: ~1000 predictions/second
- **Memory Usage**: ~500MB for ensemble model
- **API Latency**: <100ms for single predictions

### Optimization Tips

1. **Model Selection**: Use XGBoost or LightGBM for best performance
2. **Feature Engineering**: Apply domain-specific feature engineering
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Hardware**: Use GPU for neural network training
5. **Caching**: Enable Redis caching for frequently accessed data

## 🔒 Security

### Security Features

- **API Authentication**: Token-based authentication
- **Rate Limiting**: Prevent API abuse
- **Input Validation**: Comprehensive input validation
- **Error Sanitization**: Secure error messages
- **HTTPS**: SSL/TLS encryption
- **CORS**: Cross-origin resource sharing protection

### Best Practices

- Never commit sensitive data (API keys, passwords)
- Use environment variables for configuration
- Regularly update dependencies
- Monitor for security vulnerabilities
- Implement proper logging and monitoring

## 📊 Monitoring

### Metrics Available

- **Application Metrics**: Request count, latency, errors
- **Model Metrics**: Prediction confidence, model drift
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Fraud detection rate, false positive rate

### Dashboards

Access Grafana dashboards at `http://localhost:3000` (admin/admin)

### Alerts

Configure alerts for:
- High error rates
- Model performance degradation
- System resource exhaustion
- Unusual prediction patterns

## 🤝 Support

### Getting Help

- Check the [FAQ](#frequently-asked-questions)
- Review the [documentation](#documentation)
- Open an [issue](https://github.com/your-repo/issues)
- Contact the development team

### Reporting Issues

When reporting issues, please include:
- System information (OS, Python version)
- Error messages and stack traces
- Steps to reproduce the issue
- Expected vs actual behavior

## 📚 Documentation

### Additional Resources

- [API Documentation](http://localhost:8000/docs)
- [Model Documentation](docs/models.md)
- [Deployment Guide](docs/deployment.md)
- [Architecture Overview](docs/architecture.md)

### Frequently Asked Questions

**Q: What data format is required?**
A: CSV format with numerical features and a binary target column.

**Q: How do I handle missing values?**
A: The preprocessing pipeline automatically handles missing values based on configuration.

**Q: Can I use custom models?**
A: Yes, extend the base `FraudDetectionModel` class to add custom models.

**Q: How do I deploy to cloud providers?**
A: Use the Kubernetes deployment files and adapt them to your cloud provider.

**Q: What are the minimum system requirements?**
A: 2GB RAM, 2 CPU cores, 10GB disk space for basic deployment.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn team for the excellent ML library
- FastAPI team for the modern web framework
- The open-source community for various tools and libraries

## 📈 Roadmap

### Upcoming Features

- [ ] Real-time streaming predictions
- [ ] Advanced model interpretability
- [ ] AutoML capabilities
- [ ] Multi-model serving
- [ ] Advanced anomaly detection
- [ ] Graph-based fraud detection

### Performance Improvements

- [ ] Model quantization
- [ ] GPU acceleration
- [ ] Distributed training
- [ ] Edge deployment support

---

**Made with ❤️ by the Fraud Detection Team**
