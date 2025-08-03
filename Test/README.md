# CIAF Test Environment Setup

## Overview
This test environment provides a comprehensive web interface for testing three AI models integrated with the CIAF compliance framework:

1. **Job Classifier AI** - Fair hiring decisions with bias monitoring
2. **CT Scan Medical AI** - FDA-compliant medical imaging analysis
3. **Credit Scoring AI** - Fair lending compliant credit decisions

## Directory Structure
```
Test/
├── models/                     # AI model implementations
│   ├── job_classifier_model.py  # Fair hiring AI with EEOC compliance
│   ├── ct_scan_model.py         # Medical AI with FDA/HIPAA compliance
│   └── credit_scoring_model.py  # Credit AI with Fair Lending compliance
└── web/                        # Web interface
    ├── app.py                  # Flask application
    └── templates/              # HTML templates
        ├── index.html          # Main dashboard
        ├── test_job_classifier.html
        ├── test_ct_scan.html
        └── test_credit_scoring.html
```

## Setup Instructions

### 1. Install Dependencies
```bash
cd Test/web
pip install flask pandas numpy scikit-learn
```

### 2. Start the Web Server
```bash
python app.py
```

### 3. Access the Interface
Open your browser to: `http://localhost:5000`

## Model Features

### Job Classifier Model
- **Compliance**: EEOC, Title VII, ADA compliance
- **Features**: Resume analysis, bias detection, fairness metrics
- **Testing**: Interactive candidate input forms with real-time bias monitoring

### CT Scan Medical Model
- **Compliance**: FDA AI/ML guidance, HIPAA, medical device standards
- **Features**: Malignancy detection, clinical validation, patient privacy
- **Testing**: Medical imaging parameter inputs with safety warnings

### Credit Scoring Model
- **Compliance**: Fair Lending, FCRA, ECOA, GDPR
- **Features**: Credit risk assessment, adverse action notices, disparate impact testing
- **Testing**: Financial application forms with regulatory compliance indicators

## API Endpoints

### Model Management
- `GET /api/models/status` - Get status of all models
- `POST /api/models/{model_name}/load` - Load a model
- `POST /api/models/{model_name}/train` - Train a model
- `GET /api/models/{model_name}/info` - Get model information

### Predictions
- `POST /api/predict/job_classifier` - Job hiring prediction
- `POST /api/predict/ct_scan` - CT scan analysis
- `POST /api/predict/credit_scoring` - Credit decision

### Example API Usage
```python
import requests

# Load model
response = requests.post('http://localhost:5000/api/models/job_classifier/load')

# Make prediction
data = {
    "resume_keywords": 5,
    "years_experience": 3,
    "education_level": 1,
    "skills_match": 0.8,
    "location_preference": 1
}
response = requests.post('http://localhost:5000/api/predict/job_classifier', json=data)
result = response.json()
```

## Compliance Features

### Fair Hiring (Job Classifier)
- ✅ EEOC compliance monitoring
- ✅ Bias detection and mitigation
- ✅ Demographic parity analysis
- ✅ Automated fairness reporting

### Medical AI (CT Scan)
- ✅ FDA AI/ML guidance compliance
- ✅ HIPAA patient privacy protection
- ✅ Clinical validation metrics
- ✅ Medical device safety standards

### Financial AI (Credit Scoring)
- ✅ Fair Credit Reporting Act (FCRA)
- ✅ Equal Credit Opportunity Act (ECOA)
- ✅ Automatic adverse action notices
- ✅ 80% rule disparate impact testing

## Testing Scenarios

### Job Classifier Testing
1. Load the job classifier model
2. Input candidate information (skills, experience, education)
3. Review hiring recommendation and bias metrics
4. Analyze fairness across demographic groups

### CT Scan Testing
1. Load the medical AI model
2. Input patient imaging parameters
3. Review malignancy prediction and confidence
4. Verify FDA compliance and safety warnings

### Credit Scoring Testing
1. Load the credit scoring model
2. Input financial application data
3. Review credit decision and approval probability
4. Check for adverse action notices and fair lending compliance

## Troubleshooting

### Common Issues
1. **Port already in use**: Change port in `app.py` or kill existing process
2. **Module import errors**: Ensure CIAF is in Python path
3. **Model loading fails**: Check model file paths and dependencies

### Debug Mode
Set `debug=True` in `app.py` for detailed error messages:
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

## Next Steps

1. **Production Deployment**: Configure for production with proper WSGI server
2. **Database Integration**: Add persistent storage for audit trails
3. **Authentication**: Implement user authentication and role-based access
4. **Monitoring**: Add comprehensive logging and monitoring
5. **API Documentation**: Generate OpenAPI/Swagger documentation

## Support

For issues or questions about the CIAF test environment, refer to:
- Main CIAF documentation in the root directory
- Individual model documentation in the models/ directory
- Flask application logs for debugging information
