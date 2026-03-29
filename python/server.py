from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import time
import json

app = Flask(__name__)
CORS(app)

# --- CORE FAIRNESS UTILITIES ---

def calculate_fairness_metrics(y_true, y_pred, sensitive_features):
    """
    Calculates fundamental fairness metrics (SPD, DI, EOD, AOD).
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': sensitive_features
    })
    
    groups = df['group'].unique()
    group_metrics = {}
    
    for group in groups:
        group_df = df[df['group'] == group]
        
        tp = ((group_df['y_true'] == 1) & (group_df['y_pred'] == 1)).sum()
        fp = ((group_df['y_true'] == 0) & (group_df['y_pred'] == 1)).sum()
        tn = ((group_df['y_true'] == 0) & (group_df['y_pred'] == 0)).sum()
        fn = ((group_df['y_true'] == 1) & (group_df['y_pred'] == 0)).sum()
        
        selection_rate = group_df['y_pred'].mean()
        accuracy = (tp + tn) / len(group_df) if len(group_df) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        group_metrics[group] = {
            'selection_rate': float(selection_rate),
            'accuracy': float(accuracy),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'count': int(len(group_df))
        }
    
    # Privileged vs Unprivileged
    sorted_groups = sorted(group_metrics.items(), key=lambda x: x[1]['selection_rate'], reverse=True)
    privileged_group = sorted_groups[0][0]
    unprivileged_group = sorted_groups[-1][0]
    
    priv = group_metrics[privileged_group]
    unpriv = group_metrics[unprivileged_group]
    
    spd = priv['selection_rate'] - unpriv['selection_rate']
    di = unpriv['selection_rate'] / priv['selection_rate'] if priv['selection_rate'] > 0 else 1.0
    eod = abs(priv['tpr'] - unpriv['tpr'])
    aod = 0.5 * (abs(priv['fpr'] - unpriv['fpr']) + abs(priv['tpr'] - unpriv['tpr']))
    
    return {
        'spd': spd, 'di': di, 'eod': eod, 'aod': aod,
        'group_metrics': group_metrics,
        'privileged': privileged_group,
        'unprivileged': unprivileged_group
    }

# --- ADVANCED MODULES ---

def audit_robustness(model, X):
    """
    Measures model stability via input perturbation.
    Checks if small changes to numerical inputs flip the prediction.
    """
    X_perturbed = X.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns
    
    # Inject 5% Gaussian noise into numerical columns
    for col in num_cols:
        std = X[col].std()
        X_perturbed[col] = X[col] + np.random.normal(0, 0.05 * std, size=len(X))
    
    y_orig = model.predict(X)
    y_pert = model.predict(X_perturbed)
    
    stability = (y_orig == y_pert).mean()
    return float(stability)

def audit_privacy(y_pred, sensitive_features):
    """
    Measures Attribute Inference Risk.
    Calculates how much information the model's predictions leak about the sensitive attribute.
    """
    # Simple heuristic: If prediction distribution is heavily correlated with sensitive group
    # we can 'guess' the attribute from the outcome.
    df = pd.DataFrame({'pred': y_pred, 'attr': sensitive_features})
    contingency = pd.crosstab(df['pred'], df['attr'], normalize='index')
    
    # Max probability of guessing attribute correctly given prediction
    risk = contingency.max(axis=1).mean()
    # Normalize: 0 is no leak (uniform), 1 is full leak
    normalized_risk = (risk - (1/len(df['attr'].unique()))) / (1 - (1/len(df['attr'].unique())))
    return float(max(0, 1 - normalized_risk)) # Returns Privacy Score (Higher is better)

def generate_mitigation_plan(fairness_results):
    """
    Suggests post-processing threshold adjustments to reach Equal Opportunity.
    """
    privileged = fairness_results['privileged']
    unprivileged = fairness_results['unprivileged']
    
    priv_tpr = fairness_results['group_metrics'][privileged]['tpr']
    unpriv_tpr = fairness_results['group_metrics'][unprivileged]['tpr']
    
    diff = priv_tpr - unpriv_tpr
    if diff > 0.05:
        return f"Lower the decision threshold for group '{unprivileged}' by {diff/2:.2f} to equalize True Positive Rates."
    return "No urgent threshold mitigation required."

def generate_compliance_report(ethics_score, fairness_metrics):
    """
    Maps audit results to regulatory frameworks (EU AI Act, NITI Aayog).
    """
    risk_level = "High" if fairness_metrics['spd'] > 0.2 or ethics_score < 5 else "Limited" if ethics_score < 8 else "Minimal"
    
    return {
        'frameworks': [
            {
                'name': 'EU AI Act',
                'status': 'Non-Compliant' if risk_level == "High" else 'Compliant',
                'requirement': 'Article 10 (Data & Governance) - Bias Mitigation',
                'details': 'High-risk AI systems must implement bias detection and correction.'
            },
            {
                'name': 'NITI Aayog (India)',
                'status': 'Needs Review' if risk_level != "Minimal" else 'Ethical',
                'requirement': 'Principle of Equality & Non-Discrimination',
                'details': 'AI must not perpetuate systemic biases against protected groups.'
            }
        ],
        'modelCard': {
            'intendedUse': 'Credit scoring or recruitment screening (Adult Income Dataset proxy).',
            'limitations': 'Model trained on historical census data which contains inherent systemic bias.',
            'fairnessPhilosophy': 'Equal Opportunity (Equalizing TPR across demographic groups).'
        }
    }

# --- ROUTES ---

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        model_file = request.files.get('modelFile')
        dataset_file = request.files.get('datasetFile')
        sensitive_column = request.form.get('sensitiveColumn')
        evaluation_type = request.form.get('evaluationType', 'model')
        
        if not dataset_file or not sensitive_column:
            return jsonify({'error': 'Dataset and sensitive column are required'}), 400
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            dataset_file.save(tmp.name)
            df = pd.read_csv(tmp.name)
        os.unlink(tmp.name)
        
        if sensitive_column not in df.columns:
            return jsonify({'error': f'Sensitive column "{sensitive_column}" not found'}), 400
            
        if evaluation_type == 'model':
            if not model_file:
                return jsonify({'error': 'Model file is required'}), 400
                
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_m:
                model_file.save(tmp_m.name)
                model = joblib.load(tmp_m.name)
            os.unlink(tmp_m.name)
            
            # Prediction
            X = df.drop('income', axis=1) if 'income' in df.columns else df
            y_true = df['income'].apply(lambda x: 1 if str(x).strip() == '>50K' or x == 1 else 0) if 'income' in df.columns else None
            y_pred = model.predict(X)
            
            # Audits
            fairness = calculate_fairness_metrics(y_true, y_pred, df[sensitive_column])
            robustness_score = audit_robustness(model, X)
            privacy_score = audit_privacy(y_pred, df[sensitive_column])
            
            # Scores (0-10)
            f_score = 10 * (1 - min(1, fairness['spd'])) * 0.5 + 10 * min(1, fairness['di']) * 0.5
            r_score = robustness_score * 10
            p_score = privacy_score * 10
            
            # Overall Ethics
            ethics_score = (f_score * 0.4) + (r_score * 0.3) + (p_score * 0.3)
            
            # Mitigation & Compliance
            mitigation_rec = generate_mitigation_plan(fairness)
            compliance = generate_compliance_report(ethics_score, fairness)
            
            return jsonify({
                'ethicsScore': round(float(ethics_score), 1),
                'fairnessScore': round(float(f_score), 1),
                'integrityScore': round(float(r_score), 1), # Re-using integrity UI for Robustness
                'transparencyScore': round(float(p_score), 1), # Re-using transparency UI for Privacy
                'biasDetected': bool(fairness['spd'] > 0.1),
                'riskLevel': "High" if ethics_score < 6 else "Medium" if ethics_score < 8 else "Low",
                'metrics': {
                    'demographicParityDifference': round(float(fairness['spd']), 3),
                    'equalizedOddsDifference': round(float(fairness['eod']), 3),
                    'disparateImpact': round(float(fairness['di']), 3),
                    'robustness': round(float(robustness_score), 3),
                    'privacy': round(float(privacy_score), 3)
                },
                'groupPerformance': {
                    str(k): {'selectionRate': float(v['selection_rate']), 'accuracy': float(v['accuracy']), 'count': int(v['count'])}
                    for k, v in fairness['group_metrics'].items()
                },
                'recommendations': [mitigation_rec] + (compliance['modelCard']['limitations'].split('. ')),
                'compliance': compliance,
                'chartData': [
                    {'group': str(k), 'value': round(float(v['selection_rate'] * 100), 1), 'label': 'Selection Rate'}
                    for k, v in fairness['group_metrics'].items()
                ]
            })
        else:
            # Basic Dataset Audit
            counts = df[sensitive_column].value_counts()
            balance = float(counts.min() / counts.max())
            return jsonify({
                'ethicsScore': round(float(balance * 10), 1),
                'fairnessScore': 0,
                'integrityScore': round(float(balance * 10), 1),
                'transparencyScore': 9.0,
                'biasDetected': balance < 0.5,
                'riskLevel': "High" if balance < 0.3 else "Medium" if balance < 0.7 else "Low",
                'metrics': {'demographicParityDifference': 0, 'disparateImpact': round(balance, 3)},
                'groupPerformance': {str(k): {'count': int(v), 'percentage': float(v/len(df))} for k, v in counts.items()},
                'recommendations': ["Collect more data for underrepresented groups."],
                'chartData': [{'group': str(k), 'value': round(float(v/len(df)*100), 1), 'label': 'Representation'} for k, v in counts.items()]
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
