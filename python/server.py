from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import os
import tempfile
import time

app = Flask(__name__)
CORS(app)

@app.route('/evaluate', methods=['POST'])
def evaluateModel():
    try:
        modelFile = request.files.get('modelFile')
        datasetFile = request.files['datasetFile']
        sensitiveColumn = request.form.get('sensitiveColumn', '')
        evaluationType = request.form.get('evaluationType', 'model')
        
        if not sensitiveColumn:
            return jsonify({'error': 'Sensitive column name is required'}), 400
        
        if evaluationType == 'model' and not modelFile:
            return jsonify({'error': 'Model file is required for model evaluation'}), 400
        
        tempModelPath = None
        tempDatasetPath = None
        modelPipeline = None
        
        try:
            if modelFile:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tempModelFile:
                    modelFile.save(tempModelFile.name)
                    tempModelPath = tempModelFile.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tempDatasetFile:
                datasetFile.save(tempDatasetFile.name)
                tempDatasetPath = tempDatasetFile.name
            
            testData = pd.read_csv(tempDatasetPath)
            
            if evaluationType == 'model' and tempModelPath:
                modelPipeline = joblib.load(tempModelPath)
            
            time.sleep(0.1)
            
        finally:
            if tempModelPath and os.path.exists(tempModelPath):
                try:
                    os.unlink(tempModelPath)
                except OSError:
                    pass
            
            if tempDatasetPath and os.path.exists(tempDatasetPath):
                try:
                    os.unlink(tempDatasetPath)
                except OSError:
                    pass
        
        if sensitiveColumn not in testData.columns:
            return jsonify({'error': f'Sensitive column "{sensitiveColumn}" not found in dataset'}), 400
        
        if evaluationType == 'model':
            if 'income' not in testData.columns:
                return jsonify({'error': 'Target column "income" not found in dataset'}), 400
            
            if modelPipeline is None:
                return jsonify({'error': 'Model pipeline not loaded'}), 400
            
            XTest = testData.drop('income', axis=1)
            yTest = testData['income']
            
            if yTest.dtype == 'object':
                yTest = yTest.apply(lambda x: 1 if x == '>50K' else 0)
            
            yPred = modelPipeline.predict(XTest)
            
            accuracy = accuracy_score(yTest, yPred)
            precision = precision_score(yTest, yPred, zero_division=0)
            recall = recall_score(yTest, yPred, zero_division=0)
            f1Score = f1_score(yTest, yPred, zero_division=0)
        else:
            accuracy = precision = recall = f1Score = 0.0
            yPred = None
        
        sensitiveAttribute = testData[sensitiveColumn]
        
        if evaluationType == 'model' and yPred is not None:
            try:
                dpd = demographic_parity_difference(yTest, yPred, sensitive_features=sensitiveAttribute)
                eod = equalized_odds_difference(yTest, yPred, sensitive_features=sensitiveAttribute)
            except ValueError:
                dpd = 0.0
                eod = 0.0
            
            uniqueGroups = sensitiveAttribute.unique()
            groupPerformance = {}
            
            for group in uniqueGroups:
                groupMask = sensitiveAttribute == group
                groupYTest = yTest[groupMask]
                groupYPred = yPred[groupMask]
                
                if len(groupYTest) > 0:
                    groupAccuracy = accuracy_score(groupYTest, groupYPred)
                    groupPrecision = precision_score(groupYTest, groupYPred, zero_division=0)
                    groupPerformance[str(group)] = {
                        'accuracy': float(groupAccuracy),
                        'precision': float(groupPrecision)
                    }
        else:
            dpd = eod = 0.0
            groupPerformance = {}
            
            uniqueGroups = sensitiveAttribute.unique()
            for group in uniqueGroups:
                groupCount = int((sensitiveAttribute == group).sum())
                groupPercentage = float(groupCount / len(sensitiveAttribute))
                groupPerformance[str(group)] = {
                    'count': groupCount,
                    'percentage': groupPercentage
                }
        
        fairnessScore = max(0, min(10, 10 - (dpd * 50)))
        transparencyScore = 8.2
        
        biasDetected = dpd > 0.1
        riskLevel = "High" if dpd > 0.2 else "Medium" if dpd > 0.1 else "Low"
        
        recommendations = []
        if evaluationType == 'model':
            if biasDetected:
                recommendations.append(f"Address bias in {sensitiveColumn} predictions (DPD: {dpd:.3f} > 0.1 threshold)")
                recommendations.append("Implement fairness constraints using fairlearn library")
                recommendations.append("Consider re-balancing training data across demographic groups")
            recommendations.append("Add explainability features using SHAP values")
        else:
            recommendations.append(f"Dataset analysis complete for {sensitiveColumn} attribute")
            recommendations.append("Consider uploading a trained model for comprehensive bias analysis")
            recommendations.append("Review data distribution across sensitive groups")
            recommendations.append("Ensure balanced representation in training data")
        
        result = {
            'fairnessScore': round(float(fairnessScore), 1),
            'transparencyScore': float(transparencyScore),
            'biasDetected': bool(biasDetected),
            'riskLevel': str(riskLevel),
            'demographicParityDifference': round(float(dpd), 3),
            'equalizedOddsDifference': round(float(eod), 3),
            'accuracyMetrics': {
                'overall': round(float(accuracy), 3),
                'precision': round(float(precision), 3),
                'recall': round(float(recall), 3),
                'f1Score': round(float(f1Score), 3)
            },
            'groupPerformance': groupPerformance,
            'recommendations': recommendations
        }
        
        return jsonify(result)
        
    except (ValueError, KeyError, FileNotFoundError) as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 400
    except (OSError, IOError) as e:
        return jsonify({'error': f'File operation failed: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def healthCheck():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
