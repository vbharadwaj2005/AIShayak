from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def detect_model_framework(model):
    name = type(model).__module__
    if 'sklearn' in name:
        return 'sklearn'
    if 'xgboost' in name:
        return 'xgboost'
    if 'torch' in name or 'pytorch' in name:
        return 'pytorch'
    if 'tensorflow' in name or 'keras' in name:
        return 'tensorflow'
    return 'unknown'

def predict_model(model, X, framework=None):
    if framework is None:
        framework = detect_model_framework(model)
    if framework == 'xgboost':
        return model.predict(X)
    if framework == 'pytorch':
        model.eval()
        import torch
        with torch.no_grad():
            X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
            out = model(X_tensor)
            if out.dim() > 1 and out.shape[1] > 1:
                return out.argmax(dim=1).numpy()
            return (out.squeeze() > 0.5).numpy().astype(int)
    if framework == 'tensorflow':
        out = model.predict(X, verbose=0)
        if out.ndim > 1 and out.shape[1] > 1:
            return out.argmax(axis=1)
        return (out.squeeze() > 0.5).astype(int)
    return model.predict(X)

def predict_proba_model(model, X, framework=None):
    if framework is None:
        framework = detect_model_framework(model)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    if framework == 'xgboost':
        return model.predict(X, output_margin=False)
    if framework == 'pytorch':
        import torch
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
            out = torch.softmax(model(X_tensor), dim=1) if model(X_tensor).dim() > 1 else torch.sigmoid(model(X_tensor))
            return out.numpy()
    if framework == 'tensorflow':
        return model.predict(X, verbose=0)
    return model.predict(X)

def load_model(model_path, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.pkl', '.joblib'):
        model = joblib.load(model_path)
        return model, detect_model_framework(model)
    if ext == '.json':
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model, 'xgboost'
    if ext == '.ubj':
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model, 'xgboost'
    if ext in ('.pt', '.pth'):
        try:
            import torch
            model = torch.jit.load(model_path)
            model.eval()
            return model, 'pytorch'
        except Exception:
            model = torch.load(model_path, map_location='cpu')
            return model, 'pytorch'
    if ext in ('.h5', '.keras', '.pb'):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, 'tensorflow'
    raise ValueError(f"Unsupported model format: {ext}")

def calculate_fairness_metrics(y_true, y_pred, sensitive_features):
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

def calculate_regression_fairness(y_true, y_pred, sensitive_features):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': sensitive_features})
    groups = df['group'].unique()
    group_metrics = {}
    overall_mean = y_pred.mean()
    for group in groups:
        g = df[df['group'] == group]
        group_metrics[group] = {
            'count': int(len(g)),
            'mean_prediction': float(g['y_pred'].mean()),
            'mean_actual': float(g['y_true'].mean()),
            'mae': float(abs(g['y_pred'] - g['y_true']).mean())
        }
    groups_list = sorted(group_metrics.keys())
    mean_diff = 0.0
    max_mae_gap = 0.0
    if len(groups_list) >= 2:
        vals = [group_metrics[g]['mean_prediction'] for g in groups_list]
        mean_diff = float(max(vals) - min(vals))
        mae_vals = [group_metrics[g]['mae'] for g in groups_list]
        max_mae_gap = float(max(mae_vals) - min(mae_vals))
    return {
        'group_metrics': group_metrics,
        'mean_prediction_difference': round(mean_diff, 4),
        'max_mae_disparity': round(max_mae_gap, 4),
        'overall_mean': float(overall_mean)
    }

def intersectional_analysis(y_true, y_pred, sensitive_columns_df, columns):
    results = {}
    combined = sensitive_columns_df[columns[0]].astype(str)
    for col in columns[1:]:
        combined = combined + '_' + sensitive_columns_df[col].astype(str)
    combined_series = combined
    results['intersection_groups'] = combined_series.value_counts().to_dict()
    fair = calculate_fairness_metrics(y_true, y_pred, combined_series)
    results['fairness'] = {
        'spd': fair['spd'],
        'di': fair['di'],
        'eod': fair['eod'],
        'aod': fair['aod'],
        'privileged': fair['privileged'],
        'unprivileged': fair['unprivileged']
    }
    results['group_metrics'] = {
        str(k): {
            'selectionRate': float(v['selection_rate']),
            'accuracy': float(v['accuracy']),
            'count': int(v['count'])
        } for k, v in fair['group_metrics'].items()
    }
    results['chartData'] = [
        {'group': str(k), 'value': round(float(v['selection_rate'] * 100), 1), 'label': 'Selection Rate'}
        for k, v in fair['group_metrics'].items()
    ]
    return results

def calibration_parity(y_true, y_pred_proba, sensitive_features, n_bins=10):
    df = pd.DataFrame({'y_true': y_true, 'proba': y_pred_proba, 'group': sensitive_features})
    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(n_bins)]
    df['bin'] = pd.cut(df['proba'], bins=bins, labels=bin_labels, include_lowest=True)
    groups = df['group'].unique()
    result = {}
    for group in groups:
        g = df[df['group'] == group]
        cal = g.groupby('bin', observed=True).apply(
            lambda x: {'mean_pred': float(x['proba'].mean()), 'mean_actual': float(x['y_true'].mean()), 'count': int(len(x))},
            include_groups=False
        ).to_dict()
        result[str(group)] = cal
    cal_error = 0.0
    for group, cal_data in result.items():
        for b, v in cal_data.items():
            if v['count'] > 0:
                cal_error += abs(v['mean_pred'] - v['mean_actual']) * v['count']
    cal_error /= len(df)
    return {
        'calibration_data': result,
        'calibration_error': round(float(cal_error), 4),
        'bins': bin_labels
    }

def tune_thresholds(y_true, y_pred_proba, sensitive_features):
    thresholds = np.linspace(0.05, 0.95, 19)
    results = []
    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)
        fair = calculate_fairness_metrics(y_true, y_pred_t, sensitive_features)
        acc = (y_pred_t == y_true).mean()
        results.append({
            'threshold': round(float(t), 2),
            'spd': round(float(fair['spd']), 4),
            'di': round(float(fair['di']), 4),
            'accuracy': round(float(acc), 4),
            'biasDetected': bool(fair['spd'] > 0.1)
        })
    best_fairness = min(results, key=lambda r: abs(r['spd']))
    best_accuracy = max(results, key=lambda r: r['accuracy'])
    best_tradeoff = min(results, key=lambda r: abs(r['spd']) * 2 + (1 - r['accuracy']))
    return {
        'thresholds': results,
        'optimalFairness': best_fairness,
        'optimalAccuracy': best_accuracy,
        'recommended': best_tradeoff,
        'recommendation': f"Recommended threshold: {best_tradeoff['threshold']} (SPD={best_tradeoff['spd']}, Accuracy={best_tradeoff['accuracy']})"
    }

def generate_retraining_suggestions(X, y_true, sensitive_column):
    suggestions = []
    for col in X.select_dtypes(include=[np.number]).columns[:10]:
        corr = abs(X[col].corr(pd.factorize(X[sensitive_column])[0]))
        suggestions.append({'feature': col, 'correlation_with_sensitive': round(float(corr), 3)})
    suggestions.sort(key=lambda x: x['correlation_with_sensitive'], reverse=True)
    biased_features = [s['feature'] for s in suggestions[:5] if s['correlation_with_sensitive'] > 0.1]
    result = {
        'featureCorrelations': suggestions[:10],
        'potentialBiasedFeatures': biased_features,
        'resamplingSuggestion': f"Resampling: Upsample underrepresented groups or downsample overrepresented groups in the training data." if biased_features else "No significant feature-group correlations detected.",
        'reweightingSuggestion': f"Reweighting: Assign higher sample weights to unprivileged group samples during training." if biased_features else "No reweighting needed based on feature analysis.",
        'adversarialDebiasingSuggestion': "For neural models: Add an adversarial branch that predicts the sensitive attribute from model embeddings, and train to minimize this."
    }
    return result

def adversarial_robustness(model, X, epsilon=0.1, framework=None):
    if framework is None:
        framework = detect_model_framework(model)
    rng = np.random.default_rng(42)
    X_np = X.values if hasattr(X, 'values') else np.array(X)
    num_col_mask = [i for i, col in enumerate(X.columns) if X[col].dtype in (np.number,)]
    if not num_col_mask:
        return {'accuracy_under_attack': 1.0, 'attack_success_rate': 0.0, 'perturbation': 0.0}
    y_orig = predict_model(model, X, framework)
    perturbed = X_np.copy()
    for idx in num_col_mask:
        col_name = X.columns[idx]
        std = X[col_name].std()
        if std == 0:
            continue
        direction = rng.choice([-1, 1], size=len(X))
        perturbed[:, idx] = perturbed[:, idx] + direction * epsilon * std
    perturbed_df = pd.DataFrame(perturbed, columns=X.columns)
    y_attacked = predict_model(model, perturbed_df, framework)
    flip_rate = 1.0 - (y_orig == y_attacked).mean()
    return {
        'accuracyUnderAttack': round(float(1.0 - flip_rate), 4),
        'attackSuccessRate': round(float(flip_rate), 4),
        'perturbationEpsilon': epsilon,
        'perturbationDescription': f"Directional ±{epsilon}·σ perturbation on numerical features"
    }

def differential_privacy_audit(model, X, y_true, sensitive_features, framework=None):
    if framework is None:
        framework = detect_model_framework(model)
    rng = np.random.default_rng(42)
    n = len(X)
    influence_scores = []
    sample_indices = rng.choice(n, size=min(50, n), replace=False)
    y_pred_full = predict_model(model, X, framework)
    for i in sample_indices:
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        y_pred_loo = predict_model(model, X.iloc[mask], framework) if hasattr(model, 'predict') else y_pred_full
        influence = 0.0
        if y_true is not None and len(y_pred_loo) >= n - 1:
            acc_full = (y_pred_full == y_true).mean()
            y_true_loo = y_true.iloc[mask].values if hasattr(y_true, 'iloc') else np.delete(y_true, i)
            acc_loo = (y_pred_loo[:len(y_true_loo)] == y_true_loo).mean()
            influence = acc_full - acc_loo
        influence_scores.append({
            'sampleIndex': int(i),
            'influence': round(float(influence), 6)
        })
    max_influence = max(abs(s['influence']) for s in influence_scores) if influence_scores else 0.0
    epsilon_estimate = min(10.0, max_influence * n / 2)
    return {
        'influenceScores': influence_scores,
        'maxInfluence': round(float(max_influence), 6),
        'estimatedEpsilon': round(float(epsilon_estimate), 3),
        'interpretation': f"ε ≈ {epsilon_estimate:.2f}-DP (higher = less privacy). Values > 1 indicate measurable privacy risk.",
        'riskLevel': "High" if epsilon_estimate > 5 else "Medium" if epsilon_estimate > 1 else "Low"
    }

def membership_inference_attack(model, X, y_true, sensitive_features, framework=None):
    if framework is None:
        framework = detect_model_framework(model)
    rng = np.random.default_rng(42)
    n = len(X)
    n_test = min(100, n // 4)
    indices = np.arange(n)
    rng.shuffle(indices)
    shadow_train_idx = indices[:n_test]
    shadow_test_idx = indices[n_test:2*n_test]
    if len(shadow_train_idx) < 10 or len(shadow_test_idx) < 10:
        return {'error': 'Insufficient samples for membership inference', 'riskScore': 0.5}
    X_shadow_train = X.iloc[shadow_train_idx]
    y_shadow_train = y_true.iloc[shadow_train_idx] if y_true is not None else None
    try:
        if framework in ('sklearn', 'xgboost', 'unknown'):
            from sklearn.ensemble import RandomForestClassifier
            shadow = RandomForestClassifier(n_estimators=20)
            if y_shadow_train is not None:
                shadow.fit(X_shadow_train, y_shadow_train)
            else:
                shadow.fit(X_shadow_train, predict_model(model, X_shadow_train, framework))
        else:
            return {'error': 'Membership inference not supported for this framework', 'riskScore': 0.5}
        preds_train = predict_proba_model(shadow, X_shadow_train, 'sklearn')
        preds_test = predict_proba_model(shadow, X.iloc[shadow_test_idx], 'sklearn')
        if preds_train.ndim > 1:
            conf_train = np.max(preds_train, axis=1)
            conf_test = np.max(preds_test, axis=1)
        else:
            conf_train = preds_train
            conf_test = preds_test
        member_conf = float(np.mean(conf_train))
        nonmember_conf = float(np.mean(conf_test))
        risk_score = float(member_conf - nonmember_conf)
        return {
            'memberConfidence': round(member_conf, 4),
            'nonMemberConfidence': round(nonmember_conf, 4),
            'riskScore': round(risk_score, 4),
            'interpretation': f"Attack confidence gap: {risk_score:.3f} (>0.1 indicates vulnerability)",
            'riskLevel': "High" if risk_score > 0.2 else "Medium" if risk_score > 0.1 else "Low",
            'attackAccuracy': round(float((np.concatenate([conf_train, conf_test]) > 0.5).mean()), 4)
        }
    except Exception as e:
        return {'error': f'Membership inference failed: {str(e)}', 'riskScore': 0.5}

def audit_robustness(model, X):
    rng = np.random.default_rng(42)
    X_perturbed = X.copy()
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        std = X[col].std()
        if std == 0:
            continue
        X_perturbed[col] = X[col] + rng.normal(0, 0.05 * std, size=len(X))
    y_orig = model.predict(X)
    y_pert = model.predict(X_perturbed)
    stability = (y_orig == y_pert).mean()
    return float(stability)

def audit_privacy(y_pred, sensitive_features):
    df = pd.DataFrame({'pred': y_pred, 'attr': sensitive_features})
    unique_attrs = df['attr'].nunique()
    if unique_attrs <= 1:
        return 1.0
    contingency = pd.crosstab(df['pred'], df['attr'], normalize='index')
    risk = contingency.max(axis=1).mean()
    expected_random = 1.0 / unique_attrs
    denom = 1.0 - expected_random
    if denom == 0:
        return 1.0
    normalized_risk = (risk - expected_random) / denom
    return float(max(0, 1 - normalized_risk))

def generate_mitigation_plan(fairness_results):
    privileged = fairness_results['privileged']
    unprivileged = fairness_results['unprivileged']
    priv_tpr = fairness_results['group_metrics'][privileged]['tpr']
    unpriv_tpr = fairness_results['group_metrics'][unprivileged]['tpr']
    diff = priv_tpr - unpriv_tpr
    if diff > 0.05:
        return f"Lower the decision threshold for group '{unprivileged}' by {diff/2:.2f} to equalize True Positive Rates."
    return "No urgent threshold mitigation required."

def generate_compliance_report(ethics_score, fairness_metrics):
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

def load_dataset(dataset_file, temp_files):
    tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    dataset_file.save(tmp_csv.name)
    temp_files.append(tmp_csv.name)
    df = pd.read_csv(tmp_csv.name)
    return df

def extract_target(df, target_col='income'):
    if target_col in df.columns:
        y_true = df[target_col].apply(lambda x: 1 if str(x).strip() in ('>50K', '1', '1.0') else 0)
        X = df.drop(target_col, axis=1)
        return X, y_true
    return df, None

@app.route('/evaluate', methods=['POST'])
def evaluate():
    temp_files = []
    try:
        model_file = request.files.get('modelFile')
        dataset_file = request.files.get('datasetFile')
        sensitive_column = request.form.get('sensitiveColumn', '')
        evaluation_type = request.form.get('evaluationType', 'model')
        sensitive_column = sensitive_column.strip()
        if not dataset_file:
            return jsonify({'error': 'Dataset file is required'}), 400
        if not sensitive_column:
            return jsonify({'error': 'Sensitive column name is required'}), 400
        df = load_dataset(dataset_file, temp_files)
        if sensitive_column not in df.columns:
            return jsonify({'error': f'Sensitive column "{sensitive_column}" not found in dataset columns: {list(df.columns)}'}), 400
        if evaluation_type == 'model':
            if not model_file:
                return jsonify({'error': 'Model file is required for model evaluation'}), 400
            model_file.stream.seek(0)
            if model_file.content_length and model_file.content_length > 500 * 1024 * 1024:
                raise ValueError('Model file exceeds maximum size (500MB)')
            tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.filename or '.pkl')[1])
            model_file.save(tmp_model.name)
            temp_files.append(tmp_model.name)
            model, framework = load_model(tmp_model.name, model_file.filename or 'model.pkl')
            X, y_true = extract_target(df)
            logger.info("Running prediction on %d samples (framework: %s)...", len(X), framework)
            y_pred = predict_model(model, X, framework)
            fair = calculate_fairness_metrics(y_true, y_pred, df[sensitive_column])
            robustness_score = audit_robustness(model, X)
            privacy_score = audit_privacy(y_pred, df[sensitive_column])
            f_score = 10 * (1 - min(1, fair['spd'])) * 0.5 + 10 * min(1, fair['di']) * 0.5
            r_score = robustness_score * 10
            p_score = privacy_score * 10
            ethics_score = (f_score * 0.4) + (r_score * 0.3) + (p_score * 0.3)
            mitigation_rec = generate_mitigation_plan(fair)
            compliance = generate_compliance_report(ethics_score, fair)
            logger.info("Audit complete — Ethics Score: %.1f (framework: %s)", ethics_score, framework)
            return jsonify({
                'ethicsScore': round(float(ethics_score), 1),
                'fairnessScore': round(float(f_score), 1),
                'integrityScore': round(float(r_score), 1),
                'transparencyScore': round(float(p_score), 1),
                'biasDetected': bool(fair['spd'] > 0.1),
                'riskLevel': "High" if ethics_score < 6 else "Medium" if ethics_score < 8 else "Low",
                'framework': framework,
                'isClassification': True,
                'metrics': {
                    'demographicParityDifference': round(float(fair['spd']), 3),
                    'equalizedOddsDifference': round(float(fair['eod']), 3),
                    'disparateImpact': round(float(fair['di']), 3),
                    'robustness': round(float(robustness_score), 3),
                    'privacy': round(float(privacy_score), 3)
                },
                'groupPerformance': {
                    str(k): {'selectionRate': float(v['selection_rate']), 'accuracy': float(v['accuracy']), 'count': int(v['count'])}
                    for k, v in fair['group_metrics'].items()
                },
                'recommendations': [mitigation_rec] + (compliance['modelCard']['limitations'].split('. ')),
                'compliance': compliance,
                'chartData': [
                    {'group': str(k), 'value': round(float(v['selection_rate'] * 100), 1), 'label': 'Selection Rate'}
                    for k, v in fair['group_metrics'].items()
                ]
            })
        else:
            counts = df[sensitive_column].value_counts()
            if counts.empty:
                raise ValueError('Sensitive column has no values')
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
        logger.error("Evaluation failed: %s", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", tmp_path, e)

@app.route('/deep-audit', methods=['POST'])
def deep_audit():
    temp_files = []
    try:
        model_file = request.files.get('modelFile')
        dataset_file = request.files.get('datasetFile')
        sensitive_columns_str = request.form.get('sensitiveColumns', '')
        audit_type = request.form.get('auditType', 'intersectional')
        sensitive_columns = [c.strip() for c in sensitive_columns_str.split(',') if c.strip()]
        if not dataset_file or not sensitive_columns:
            return jsonify({'error': 'Dataset file and at least one sensitive column are required'}), 400
        df = load_dataset(dataset_file, temp_files)
        for col in sensitive_columns:
            if col not in df.columns:
                return jsonify({'error': f'Column "{col}" not found in dataset'}), 400
        if audit_type == 'intersectional':
            if len(sensitive_columns) < 2:
                return jsonify({'error': 'Intersectional analysis requires at least 2 sensitive columns'}), 400
            if not model_file:
                return jsonify({'error': 'Model file required for intersectional analysis'}), 400
            model_file.stream.seek(0)
            if model_file.content_length and model_file.content_length > 500 * 1024 * 1024:
                raise ValueError('Model file exceeds maximum size (500MB)')
            tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.filename or '.pkl')[1])
            model_file.save(tmp_model.name)
            temp_files.append(tmp_model.name)
            model, framework = load_model(tmp_model.name, model_file.filename or 'model.pkl')
            X, y_true = extract_target(df)
            y_pred = predict_model(model, X, framework)
            results = intersectional_analysis(y_true, y_pred, df[sensitive_columns], sensitive_columns)
            return jsonify(results)
        if audit_type == 'calibration':
            if not model_file:
                return jsonify({'error': 'Model file required for calibration analysis'}), 400
            model_file.stream.seek(0)
            if model_file.content_length and model_file.content_length > 500 * 1024 * 1024:
                raise ValueError('Model file exceeds maximum size (500MB)')
            tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.filename or '.pkl')[1])
            model_file.save(tmp_model.name)
            temp_files.append(tmp_model.name)
            model, framework = load_model(tmp_model.name, model_file.filename or 'model.pkl')
            X, y_true = extract_target(df)
            y_pred_proba = predict_proba_model(model, X, framework)
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
            results = calibration_parity(y_true, y_pred_proba, df[sensitive_columns[0]])
            return jsonify(results)
        return jsonify({'error': f'Unknown audit type: {audit_type}'}), 400
    except Exception as e:
        logger.error("Deep audit failed: %s", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", tmp_path, e)

@app.route('/optimize', methods=['POST'])
def optimize():
    temp_files = []
    try:
        model_file = request.files.get('modelFile')
        dataset_file = request.files.get('datasetFile')
        sensitive_column = request.form.get('sensitiveColumn', '')
        optimize_type = request.form.get('optimizeType', 'threshold')
        sensitive_column = sensitive_column.strip()
        if not model_file or not dataset_file or not sensitive_column:
            return jsonify({'error': 'Model, dataset, and sensitive column are required'}), 400
        df = load_dataset(dataset_file, temp_files)
        if sensitive_column not in df.columns:
            return jsonify({'error': f'Sensitive column "{sensitive_column}" not found'}), 400
        model_file.stream.seek(0)
        if model_file.content_length and model_file.content_length > 500 * 1024 * 1024:
            raise ValueError('Model file exceeds maximum size (500MB)')
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.filename or '.pkl')[1])
        model_file.save(tmp_model.name)
        temp_files.append(tmp_model.name)
        model, framework = load_model(tmp_model.name, model_file.filename or 'model.pkl')
        X, y_true = extract_target(df)
        if y_true is None:
            return jsonify({'error': 'Dataset must contain a target column (e.g., "income") for optimization'}), 400
        if optimize_type == 'threshold':
            y_pred_proba = predict_proba_model(model, X, framework)
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]
            results = tune_thresholds(y_true, y_pred_proba, df[sensitive_column])
            return jsonify(results)
        if optimize_type == 'retraining':
            results = generate_retraining_suggestions(X, df, sensitive_column)
            return jsonify(results)
        if optimize_type == 'regression-fairness':
            y_pred = predict_model(model, X, framework)
            results = calculate_regression_fairness(y_true, y_pred, df[sensitive_column])
            return jsonify(results)
        return jsonify({'error': f'Unknown optimization type: {optimize_type}'}), 400
    except Exception as e:
        logger.error("Optimization failed: %s", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", tmp_path, e)

@app.route('/security-audit', methods=['POST'])
def security_audit():
    temp_files = []
    try:
        model_file = request.files.get('modelFile')
        dataset_file = request.files.get('datasetFile')
        sensitive_column = request.form.get('sensitiveColumn', '')
        audit_type = request.form.get('auditType', 'adversarial')
        sensitive_column = sensitive_column.strip()
        if not model_file or not dataset_file or not sensitive_column:
            return jsonify({'error': 'Model, dataset, and sensitive column are required'}), 400
        df = load_dataset(dataset_file, temp_files)
        if sensitive_column not in df.columns:
            return jsonify({'error': f'Sensitive column "{sensitive_column}" not found'}), 400
        model_file.stream.seek(0)
        if model_file.content_length and model_file.content_length > 500 * 1024 * 1024:
            raise ValueError('Model file exceeds maximum size (500MB)')
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(model_file.filename or '.pkl')[1])
        model_file.save(tmp_model.name)
        temp_files.append(tmp_model.name)
        model, framework = load_model(tmp_model.name, model_file.filename or 'model.pkl')
        X, y_true = extract_target(df)
        if audit_type == 'adversarial':
            results = adversarial_robustness(model, X, epsilon=0.1, framework=framework)
            return jsonify(results)
        if audit_type == 'differential-privacy':
            results = differential_privacy_audit(model, X, y_true, df[sensitive_column], framework)
            return jsonify(results)
        if audit_type == 'membership-inference':
            results = membership_inference_attack(model, X, y_true, df[sensitive_column], framework)
            return jsonify(results)
        return jsonify({'error': f'Unknown security audit type: {audit_type}'}), 400
    except Exception as e:
        logger.error("Security audit failed: %s", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", tmp_path, e)

@app.route('/preview', methods=['POST'])
def preview():
    temp_files = []
    try:
        dataset_file = request.files.get('datasetFile')
        if not dataset_file:
            return jsonify({'error': 'Dataset file is required'}), 400
        df = load_dataset(dataset_file, temp_files)
        columns = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'sampleValues': df[col].dropna().head(5).tolist(),
                'missingCount': int(df[col].isna().sum()),
                'uniqueCount': int(df[col].nunique())
            }
            if df[col].dtype in (np.number,):
                col_info['min'] = float(df[col].min()) if pd.notna(df[col].min()) else None
                col_info['max'] = float(df[col].max()) if pd.notna(df[col].max()) else None
                col_info['mean'] = float(df[col].mean()) if pd.notna(df[col].mean()) else None
            columns.append(col_info)
        return jsonify({
            'rowCount': len(df),
            'columnCount': len(df.columns),
            'columns': columns,
            'targetColumn': 'income' if 'income' in df.columns else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        for tmp_path in temp_files:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file %s: %s", tmp_path, e)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
