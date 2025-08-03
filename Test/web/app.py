#!/usr/bin/env python3
"""
CIAF Model Testing Web Application
Flask web interface for testing AI models with CIAF compliance monitoring
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Import models
try:
    from job_classifier_model import JobClassifierModel
    from ct_scan_model import CTScanModel
    from credit_scoring_model import CreditScoringModel
except ImportError as e:
    print(f"Warning: Could not import models: {e}")

app = Flask(__name__)
app.secret_key = 'ciaf_testing_secret_key_2024'

# Global model instances
models = {
    'job_classifier': None,
    'ct_scan': None,
    'credit_scoring': None
}

# Model status tracking
model_status = {
    'job_classifier': {'loaded': False, 'trained': False, 'error': None},
    'ct_scan': {'loaded': False, 'trained': False, 'error': None},
    'credit_scoring': {'loaded': False, 'trained': False, 'error': None}
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', model_status=model_status)

@app.route('/api/models/status')
def api_models_status():
    """API endpoint for model status"""
    return jsonify(model_status)

@app.route('/api/models/load/<model_name>')
def api_load_model(model_name):
    """Load a specific model"""
    try:
        if model_name == 'job_classifier':
            models['job_classifier'] = JobClassifierModel()
            model_status['job_classifier']['loaded'] = True
            model_status['job_classifier']['error'] = None
            
        elif model_name == 'ct_scan':
            models['ct_scan'] = CTScanModel()
            model_status['ct_scan']['loaded'] = True
            model_status['ct_scan']['error'] = None
            
        elif model_name == 'credit_scoring':
            models['credit_scoring'] = CreditScoringModel()
            model_status['credit_scoring']['loaded'] = True
            model_status['credit_scoring']['error'] = None
            
        else:
            return jsonify({'error': f'Unknown model: {model_name}'}), 400
        
        return jsonify({'success': True, 'message': f'{model_name} loaded successfully'})
        
    except Exception as e:
        error_msg = str(e)
        model_status[model_name]['error'] = error_msg
        return jsonify({'error': error_msg}), 500

@app.route('/api/models/train/<model_name>')
def api_train_model(model_name):
    """Train a specific model"""
    try:
        if model_name not in models or models[model_name] is None:
            return jsonify({'error': f'{model_name} not loaded'}), 400
        
        model = models[model_name]
        
        if model_name == 'job_classifier':
            # Generate data and train
            data = model.generate_sample_data(n_samples=1000)
            X, y, data_processed = model.preprocess_data(data)
            X_test, y_test, y_pred = model.train_model(X, y)
            model.evaluate_bias(X_test, y_test, y_pred, data_processed)
            model.calculate_compliance_score()
            
        elif model_name == 'ct_scan':
            # Generate medical data and train
            data = model.generate_sample_ct_data(n_samples=800)
            X, y, data_processed = model.preprocess_medical_data(data)
            X_test, y_test, y_pred, y_prob = model.train_medical_model(X, y)
            model.validate_fda_requirements(X_test, y_test, y_pred)
            
        elif model_name == 'credit_scoring':
            # Generate credit data and train
            data = model.generate_sample_credit_data(n_samples=1200)
            X, y, data_processed = model.preprocess_credit_data(data)
            X_test, y_test, y_pred, y_prob = model.train_credit_model(X, y)
            model.evaluate_fair_lending_compliance(X_test, y_test, y_pred, data_processed)
        
        model_status[model_name]['trained'] = True
        model_status[model_name]['error'] = None
        
        return jsonify({'success': True, 'message': f'{model_name} trained successfully'})
        
    except Exception as e:
        error_msg = str(e)
        model_status[model_name]['error'] = error_msg
        return jsonify({'error': error_msg}), 500

@app.route('/api/models/info/<model_name>')
def api_model_info(model_name):
    """Get model information"""
    try:
        if model_name not in models or models[model_name] is None:
            return jsonify({'error': f'{model_name} not loaded'}), 400
        
        model = models[model_name]
        
        if model_name == 'job_classifier':
            info = model.get_model_info()
        elif model_name == 'ct_scan':
            info = model.get_medical_model_info()
        elif model_name == 'credit_scoring':
            info = model.get_credit_model_info()
        else:
            return jsonify({'error': f'Unknown model: {model_name}'}), 400
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test/<model_name>')
def test_model_page(model_name):
    """Model testing page"""
    if model_name not in ['job_classifier', 'ct_scan', 'credit_scoring']:
        return redirect(url_for('index'))
    
    return render_template(f'test_{model_name}.html', model_name=model_name)

@app.route('/api/predict/<model_name>', methods=['POST'])
def api_predict(model_name):
    """Make predictions with a model"""
    try:
        print(f"Debug: Prediction request for {model_name}")
        print(f"Debug: Model loaded: {models[model_name] is not None if model_name in models else False}")
        print(f"Debug: Model trained: {model_status[model_name]['trained'] if model_name in model_status else False}")
        
        # Auto-load model if not loaded
        if model_name not in models or models[model_name] is None:
            print(f"Debug: Auto-loading {model_name}")
            if model_name == 'job_classifier':
                models[model_name] = JobClassifierModel()
                model_status[model_name]['loaded'] = True
            elif model_name == 'ct_scan':
                models[model_name] = CTScanModel()
                model_status[model_name]['loaded'] = True
            elif model_name == 'credit_scoring':
                models[model_name] = CreditScoringModel()
                model_status[model_name]['loaded'] = True
            else:
                error_msg = f'Unknown model: {model_name}'
                print(f"Debug: Error - {error_msg}")
                return jsonify({'error': error_msg}), 400
        
        # Auto-train model if not trained
        if not model_status[model_name]['trained']:
            print(f"Debug: Auto-training {model_name}")
            model = models[model_name]
            
            if model_name == 'job_classifier':
                data = model.generate_sample_data(n_samples=1000)
                X, y, data_processed = model.preprocess_data(data)
                X_test, y_test, y_pred = model.train_model(X, y)
                model.evaluate_bias(X_test, y_test, y_pred, data_processed)
                model.calculate_compliance_score()
                
            elif model_name == 'ct_scan':
                data = model.generate_sample_ct_data(n_samples=800)
                X, y, data_processed = model.preprocess_medical_data(data)
                X_test, y_test, y_pred, y_prob = model.train_medical_model(X, y)
                model.validate_fda_requirements(X_test, y_test, y_pred)
                
            elif model_name == 'credit_scoring':
                data = model.generate_sample_credit_data(n_samples=1200)
                X, y, data_processed = model.preprocess_credit_data(data)
                X_test, y_test, y_pred, y_prob = model.train_credit_model(X, y)
                model.evaluate_fair_lending_compliance(X_test, y_test, y_pred, data_processed)
            
            model_status[model_name]['trained'] = True
            model_status[model_name]['error'] = None
            print(f"Debug: {model_name} trained successfully")
        
        model = models[model_name]
        data = request.json
        print(f"Debug: Received data: {data}")
        
        if model_name == 'job_classifier':
            # Prepare job classifier input
            input_data = np.array([[
                float(data['education_score']),
                float(data['experience_years']),
                float(data['skill_score']),
                float(data['age']),
                int(data['gender_encoded'])  # 0: F, 1: M, 2: NB
            ]])
            
            input_scaled = model.scaler.transform(input_data)
            predictions, uncertainties = model.predict_with_uncertainty(input_scaled)
            
            result = {
                'model_type': 'Job Classifier',
                'hire_probability': float(predictions[0][1]),
                'uncertainty': float(uncertainties[0]),
                'decision': 'HIRE' if predictions[0][1] > 0.5 else 'NO HIRE',
                'compliance_score': model.compliance_score,
                'bias_score': model.bias_score
            }
            
        elif model_name == 'ct_scan':
            # Prepare CT scan input
            input_data = np.array([[
                float(data['hounsfield_mean']),
                float(data['hounsfield_std']),
                float(data['nodule_size_mm']),
                float(data['nodule_density']),
                float(data['edge_sharpness']),
                int(data['smoking_history']),
                int(data['upper_lobe']),
                int(data['peripheral'])
            ] + [0] * 8])  # Placeholder for encoded features
            
            # Pad to match training dimensions
            if input_data.shape[1] < 16:
                padding = np.zeros((1, 16 - input_data.shape[1]))
                input_data = np.hstack([input_data, padding])
            
            input_scaled = model.scaler.transform(input_data)
            predictions, confidences, uncertainties = model.predict_with_medical_confidence(input_scaled)
            
            malignancy_prob = float(predictions[0][1])
            risk_level = "HIGH" if malignancy_prob > 0.7 else "MEDIUM" if malignancy_prob > 0.3 else "LOW"
            
            result = {
                'model_type': 'CT Scan Classifier',
                'malignancy_probability': malignancy_prob,
                'risk_level': risk_level,
                'confidence': float(confidences[0]),
                'uncertainty': float(uncertainties[0]),
                'fda_clearance': model.fda_clearance,
                'hipaa_compliance': model.hipaa_compliance,
                'clinical_validation': model.clinical_validation
            }
            
        elif model_name == 'credit_scoring':
            # Prepare credit scoring input
            input_data = np.array([[
                float(data['annual_income']),
                float(data['credit_history_length']),
                float(data['existing_debt']),
                float(data['debt_to_income_ratio']),
                int(data['num_credit_accounts']),
                float(data['payment_history_score']),
                float(data['credit_utilization']),
                float(data['age']),
                int(data['employment_type_encoded']),
                int(data['state_encoded'])
            ]])
            
            input_scaled = model.scaler.transform(input_data)
            predictions, confidences, feature_importance = model.predict_with_explainability(input_scaled)
            
            approval_prob = float(predictions[0][1])
            decision = "APPROVED" if approval_prob > 0.5 else "DENIED"
            
            # Generate adverse action notice
            feature_names = [
                'annual_income', 'credit_history_length', 'existing_debt', 
                'debt_to_income_ratio', 'num_credit_accounts', 'payment_history_score',
                'credit_utilization', 'age', 'employment_type', 'state'
            ]
            adverse_action = model.generate_adverse_action_notice(
                approval_prob, feature_importance, feature_names
            )
            
            result = {
                'model_type': 'Credit Scoring',
                'approval_probability': approval_prob,
                'decision': decision,
                'confidence': float(confidences[0]),
                'adverse_action': adverse_action,
                'fair_lending_compliance': model.fair_lending_compliance,
                'gdpr_compliance': model.gdpr_compliance,
                'bias_score': model.bias_score
            }
        
        return jsonify(result)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Debug: Exception in prediction: {str(e)}")
        print(f"Debug: Traceback: {error_trace}")
        return jsonify({'error': str(e), 'trace': error_trace}), 500

@app.route('/compliance')
def compliance_dashboard():
    """Compliance monitoring dashboard"""
    return render_template('compliance_dashboard.html')

@app.route('/api/compliance/summary')
def api_compliance_summary():
    """Get compliance summary for all models"""
    try:
        summary = {}
        
        for model_name, model in models.items():
            if model is not None and model_status[model_name]['trained']:
                if model_name == 'job_classifier':
                    info = model.get_model_info()
                elif model_name == 'ct_scan':
                    info = model.get_medical_model_info()
                elif model_name == 'credit_scoring':
                    info = model.get_credit_model_info()
                
                summary[model_name] = {
                    'compliance_score': info.get('compliance_score', 0),
                    'model_type': info.get('model_type', 'Unknown'),
                    'is_trained': info.get('is_trained', False),
                    'regulatory_frameworks': info.get('regulatory_frameworks', [])
                }
            else:
                summary[model_name] = {
                    'compliance_score': 0,
                    'model_type': 'Not Loaded',
                    'is_trained': False,
                    'regulatory_frameworks': []
                }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pipeline/trace/<model_name>')
def api_pipeline_trace(model_name):
    """Get detailed pipeline tracing data for a specific model"""
    try:
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404
        
        model = models[model_name]
        if not model:
            return jsonify({'error': f'Model {model_name} not loaded'}), 400
        
        # Generate sample pipeline trace data based on model type
        trace_data = generate_pipeline_trace(model_name, model)
        
        return jsonify(trace_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_pipeline_trace(model_name, model):
    """Generate pipeline tracing data with CIAF metadata"""
    timestamp = datetime.now().isoformat()
    
    if model_name == 'job_classifier':
        return {
            'model_type': 'job_classifier',
            'pipeline_id': f'pipeline_{model_name}_{int(datetime.now().timestamp())}',
            'stages': [
                {
                    'stage_id': 'input',
                    'stage_name': 'Data Input',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'data_source': 'candidate_application',
                        'input_fields': ['education_score', 'experience_years', 'skill_score', 'age', 'gender'],
                        'record_count': 1,
                        'data_hash': 'sha256:a1b2c3d4e5f6789abc123def456ghi789',
                        'compliance_checks': {
                            'gdpr_consent': True,
                            'data_minimization': True,
                            'purpose_limitation': True
                        },
                        'data_quality': {
                            'completeness': 1.0,
                            'validity': 0.95,
                            'consistency': 0.98,
                            'timeliness': 0.92
                        }
                    }
                },
                {
                    'stage_id': 'preprocessing',
                    'stage_name': 'Data Preprocessing',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'transformations_applied': ['standard_scaling', 'label_encoding', 'feature_selection'],
                        'feature_engineering': {
                            'original_features': 5,
                            'engineered_features': 8,
                            'scaling_method': 'StandardScaler',
                            'encoding_method': 'LabelEncoder'
                        },
                        'bias_detection': {
                            'protected_attributes': ['gender', 'age'],
                            'initial_bias_score': 0.92,
                            'mitigation_applied': True
                        },
                        'data_lineage': {
                            'source_stage': 'input',
                            'transformation_log': ['normalize_education', 'encode_categorical', 'scale_numerical']
                        }
                    }
                },
                {
                    'stage_id': 'model_training',
                    'stage_name': 'Model Training',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'model_type': 'RandomForestClassifier',
                        'hyperparameters': {
                            'n_estimators': 100,
                            'max_depth': 10,
                            'min_samples_split': 2,
                            'class_weight': 'balanced'
                        },
                        'training_data': {
                            'training_samples': 1600,
                            'validation_samples': 400,
                            'test_samples': 200,
                            'class_distribution': {'hire': 0.65, 'no_hire': 0.35}
                        },
                        'model_artifacts': {
                            'model_hash': 'sha256:f1e2d3c4b5a6789def012ghi345jkl678',
                            'weights_checksum': 'md5:abc123def456ghi789',
                            'serialization_format': 'pickle'
                        },
                        'compliance_validation': {
                            'eeoc_bias_testing': 'passed',
                            'fairness_constraints': 'enforced',
                            'explainability_requirements': 'met'
                        }
                    }
                },
                {
                    'stage_id': 'inference',
                    'stage_name': 'Model Inference',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'prediction_result': {
                            'hire_probability': 0.78,
                            'decision': 'HIRE',
                            'confidence_score': 0.85,
                            'decision_threshold': 0.5
                        },
                        'uncertainty_quantification': {
                            'epistemic_uncertainty': 0.12,
                            'aleatoric_uncertainty': 0.08,
                            'total_uncertainty': 0.15,
                            'confidence_interval': [0.71, 0.85]
                        },
                        'bias_monitoring': {
                            'real_time_bias_check': 'passed',
                            'fairness_metrics': {
                                'demographic_parity': 0.94,
                                'equalized_odds': 0.91,
                                'statistical_parity': 0.93
                            },
                            'protected_group_impact': 'neutral'
                        },
                        'explainability': {
                            'feature_importance': {
                                'education_score': 0.35,
                                'experience_years': 0.28,
                                'skill_score': 0.25,
                                'other_factors': 0.12
                            },
                            'shap_values_available': True,
                            'lime_explanation_generated': True
                        }
                    }
                },
                {
                    'stage_id': 'audit_trail',
                    'stage_name': 'Audit Trail Generation',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'audit_record_id': f'audit_{model_name}_{int(datetime.now().timestamp())}',
                        'compliance_frameworks': ['EEOC', 'GDPR', 'Fair_Hiring_Act', 'SOX'],
                        'audit_events': [
                            'data_ingestion_logged',
                            'preprocessing_documented',
                            'bias_evaluation_completed',
                            'prediction_recorded',
                            'compliance_verified'
                        ],
                        'cryptographic_integrity': {
                            'previous_hash': 'sha256:abc123def456ghi789jkl012mno345pqr',
                            'current_hash': 'sha256:def456ghi789jkl012mno345pqr678stu',
                            'merkle_tree_root': 'sha256:789jkl012mno345pqr678stu901vwx234',
                            'chain_verified': True
                        },
                        'regulatory_compliance': {
                            'bias_testing_results': 'compliant',
                            'transparency_documentation': 'complete',
                            'auditability_verified': True,
                            'data_governance': 'enforced'
                        },
                        'stakeholder_impact': {
                            'affected_parties': ['candidates', 'hiring_managers', 'hr_department'],
                            'impact_assessment': 'low_risk',
                            'mitigation_measures': ['human_oversight', 'appeal_process']
                        }
                    }
                }
            ],
            'compliance_metrics': {
                'overall_compliance_score': 0.94,
                'bias_score': 0.92,
                'fairness_score': 0.91,
                'transparency_score': 0.96,
                'auditability_score': 0.89,
                'data_governance_score': 0.93
            }
        }
    
    elif model_name == 'ct_scan':
        return {
            'model_type': 'ct_scan',
            'pipeline_id': f'pipeline_{model_name}_{int(datetime.now().timestamp())}',
            'stages': [
                {
                    'stage_id': 'medical_input',
                    'stage_name': 'Medical Data Input',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'data_source': 'ct_scan_dicom',
                        'patient_id': 'PATIENT_***_ANONYMIZED',
                        'imaging_parameters': {
                            'slice_thickness': 1.25,
                            'pixel_spacing': [0.625, 0.625],
                            'hounsfield_range': [-1024, 3071],
                            'nodule_measurements': {
                                'diameter_mm': 8.5,
                                'volume_mm3': 321.4,
                                'location': 'upper_right_lobe'
                            }
                        },
                        'hipaa_compliance': {
                            'phi_removed': True,
                            'anonymization_method': 'safe_harbor_de_identification',
                            'consent_obtained': True,
                            'minimum_necessary_standard': True
                        },
                        'data_integrity': {
                            'dicom_validation': 'passed',
                            'checksum_verified': True,
                            'corruption_check': 'clean'
                        }
                    }
                },
                {
                    'stage_id': 'medical_preprocessing',
                    'stage_name': 'Medical Image Preprocessing',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'medical_transformations': [
                            'hounsfield_normalization',
                            'lung_segmentation',
                            'nodule_detection',
                            'feature_extraction'
                        ],
                        'clinical_context': {
                            'patient_demographics': 'anonymized',
                            'smoking_history': 'encoded',
                            'family_history': 'encoded',
                            'radiologist_annotations': 'processed'
                        },
                        'quality_assurance': {
                            'image_quality_score': 0.94,
                            'artifact_detection': 'none_found',
                            'clinical_adequacy': 'sufficient'
                        },
                        'fda_preprocessing': {
                            'method_validation': 'FDA_510k_compliant',
                            'predicate_device_comparison': 'substantial_equivalence',
                            'clinical_validation': 'verified'
                        }
                    }
                },
                {
                    'stage_id': 'medical_training',
                    'stage_name': 'Medical AI Training',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'model_architecture': 'GradientBoostingClassifier',
                        'clinical_validation': {
                            'validation_cohort_size': 800,
                            'clinical_sites': 3,
                            'radiologist_agreement_kappa': 0.89,
                            'inter_observer_variability': 0.12
                        },
                        'fda_compliance': {
                            'device_classification': 'Class_II_Medical_Device',
                            'predicate_device': 'K193456',
                            'substantial_equivalence': 'demonstrated',
                            'clinical_study_protocol': 'approved'
                        },
                        'performance_validation': {
                            'sensitivity': 0.91,
                            'specificity': 0.88,
                            'positive_predictive_value': 0.85,
                            'negative_predictive_value': 0.93,
                            'auc_roc': 0.92
                        },
                        'safety_validation': {
                            'false_positive_rate': 0.12,
                            'false_negative_rate': 0.09,
                            'clinical_impact_assessment': 'low_risk'
                        }
                    }
                },
                {
                    'stage_id': 'medical_inference',
                    'stage_name': 'Medical Diagnosis',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'diagnostic_result': {
                            'malignancy_probability': 0.23,
                            'classification': 'BENIGN',
                            'confidence_interval': [0.18, 0.28],
                            'diagnostic_confidence': 'high'
                        },
                        'clinical_decision_support': {
                            'recommendation': 'routine_follow_up_12_months',
                            'urgency_level': 'low',
                            'additional_imaging_required': False,
                            'radiologist_review_recommended': True
                        },
                        'safety_monitoring': {
                            'adverse_event_potential': 'none',
                            'clinical_oversight_required': True,
                            'patient_notification_protocol': 'standard'
                        },
                        'radiologist_integration': {
                            'ai_as_second_reader': True,
                            'final_diagnosis_authority': 'radiologist',
                            'disagreement_resolution': 'clinical_review'
                        }
                    }
                },
                {
                    'stage_id': 'medical_audit',
                    'stage_name': 'Medical Compliance Audit',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'regulatory_audit': {
                            'fda_audit_trail': 'complete',
                            'hipaa_logging': 'compliant',
                            'clinical_documentation': 'verified',
                            'quality_management_system': 'iso_13485_certified'
                        },
                        'patient_safety': {
                            'adverse_event_monitoring': 'active',
                            'post_market_surveillance': 'enabled',
                            'clinical_performance_monitoring': 'ongoing',
                            'safety_reporting': 'current'
                        },
                        'clinical_governance': {
                            'medical_oversight': 'board_certified_radiologist',
                            'clinical_validation_ongoing': True,
                            'performance_degradation_monitoring': 'active'
                        }
                    }
                }
            ],
            'compliance_metrics': {
                'overall_compliance_score': 0.96,
                'fda_compliance': 0.95,
                'hipaa_compliance': 0.98,
                'clinical_accuracy': 0.91,
                'safety_score': 0.94,
                'patient_privacy_score': 0.97
            }
        }
    
    elif model_name == 'credit_scoring':
        return {
            'model_type': 'credit_scoring',
            'pipeline_id': f'pipeline_{model_name}_{int(datetime.now().timestamp())}',
            'stages': [
                {
                    'stage_id': 'financial_input',
                    'stage_name': 'Financial Data Input',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'data_source': 'credit_application_system',
                        'applicant_data': {
                            'annual_income': 65000,
                            'credit_history_months': 96,
                            'debt_to_income_ratio': 0.23,
                            'employment_status': 'full_time_permanent'
                        },
                        'data_protection': {
                            'gdpr_consent_obtained': True,
                            'ccpa_compliance_verified': True,
                            'encryption_standard': 'AES_256_GCM',
                            'data_retention_policy': 'enforced'
                        },
                        'fair_lending_compliance': {
                            'protected_class_identification': True,
                            'disparate_impact_monitoring': 'enabled',
                            'adverse_action_tracking': 'active'
                        }
                    }
                },
                {
                    'stage_id': 'financial_preprocessing',
                    'stage_name': 'Financial Data Processing',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'data_transformations': [
                            'income_normalization',
                            'credit_utilization_calculation',
                            'risk_factor_engineering',
                            'debt_consolidation_analysis'
                        ],
                        'fair_lending_preprocessing': {
                            'protected_attributes': ['age', 'gender', 'ethnicity', 'marital_status'],
                            'bias_mitigation_techniques': ['reweighting', 'fairness_constraints'],
                            'demographic_parity_monitoring': 'initialized'
                        },
                        'regulatory_preprocessing': {
                            'fcra_compliance_check': 'verified',
                            'ecoa_requirements': 'met',
                            'fair_credit_reporting': 'compliant'
                        },
                        'risk_assessment': {
                            'credit_bureau_data_integration': 'complete',
                            'alternative_data_sources': 'validated',
                            'risk_segmentation': 'applied'
                        }
                    }
                },
                {
                    'stage_id': 'credit_training',
                    'stage_name': 'Credit Model Training',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'model_architecture': 'GradientBoostingClassifier',
                        'training_compliance': {
                            'fair_lending_validation': 'passed',
                            'adverse_impact_ratio': 0.87,
                            'model_governance_documented': True,
                            'risk_management_framework': 'implemented'
                        },
                        'performance_metrics': {
                            'cross_validation_auc': 0.84,
                            'holdout_validation_auc': 0.82,
                            'temporal_stability': 0.91,
                            'population_stability_index': 0.15
                        },
                        'regulatory_training': {
                            'sr_11_7_compliance': 'model_risk_management_compliant',
                            'back_testing_results': 'satisfactory',
                            'stress_testing_results': 'passed',
                            'model_validation': 'independent_validation_completed'
                        }
                    }
                },
                {
                    'stage_id': 'credit_inference',
                    'stage_name': 'Credit Decision',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'credit_decision': {
                            'approval_probability': 0.71,
                            'final_decision': 'APPROVED',
                            'credit_limit_approved': 15000,
                            'interest_rate_offered': 14.5,
                            'terms': '24_months'
                        },
                        'fair_lending_monitoring': {
                            'disparate_impact_analysis': 'passed',
                            'adverse_action_required': False,
                            'demographic_parity_score': 0.88,
                            'equalized_odds_ratio': 0.92
                        },
                        'explainability_requirements': {
                            'primary_positive_factors': [
                                'stable_employment_history',
                                'positive_payment_history',
                                'low_debt_to_income_ratio'
                            ],
                            'primary_risk_factors': [
                                'limited_credit_history_length'
                            ],
                            'fcra_adverse_action_reasons': 'not_applicable',
                            'model_interpretability_score': 0.89
                        }
                    }
                },
                {
                    'stage_id': 'regulatory_audit',
                    'stage_name': 'Financial Regulatory Audit',
                    'timestamp': timestamp,
                    'status': 'completed',
                    'metadata': {
                        'regulatory_compliance': {
                            'fair_lending_audit': 'compliant',
                            'fcra_documentation': 'complete',
                            'ecoa_compliance': 'verified',
                            'cfpb_examination_ready': True
                        },
                        'consumer_protection': {
                            'adverse_action_notice': 'not_required',
                            'credit_report_access_provided': True,
                            'dispute_resolution_available': True,
                            'consumer_rights_protected': True
                        },
                        'risk_management': {
                            'ongoing_model_monitoring': 'active',
                            'performance_degradation_detection': 'none_detected',
                            'regulatory_reporting': 'current',
                            'capital_adequacy_impact': 'assessed'
                        }
                    }
                }
            ],
            'compliance_metrics': {
                'overall_compliance_score': 0.93,
                'fair_lending_score': 0.88,
                'gdpr_compliance': 0.96,
                'model_accuracy': 0.84,
                'bias_mitigation_score': 0.87,
                'consumer_protection_score': 0.91
            }
        }
    
    # Default return for unknown models
    return {
        'model_type': model_name,
        'error': 'Pipeline trace not available for this model type'
    }

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.route('/test_credit_scoring')
def test_credit_scoring():
    return render_template('test_credit_scoring.html')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    print("🚀 Starting CIAF Model Testing Web Application")
    print("📊 Available models: Job Classifier, CT Scan, Credit Scoring")
    print("🌐 Open http://localhost:5000 to access the dashboard")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
