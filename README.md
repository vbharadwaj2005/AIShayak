# AI-Shayak: Ethical AI Governance Platform
AI-Shayak is a comprehensive AI governance platform designed to evaluate machine learning models for bias, fairness, robustness, privacy, and security across multiple ML frameworks.

<p align="center">
  <img src="frontend/public/Screenshot1.png" alt="Home" width="45%" />
  <img src="frontend/public/Screenshot2.png" alt="Evaluation" width="45%" />
</p>

## Key Features

### Core Ethics Audit
- **Fairness Auditing**: Statistical Parity Difference (SPD), Disparate Impact (DI), Equal Opportunity Difference (EOD), Average Odds Difference (AOD).
- **Security & Robustness**: Gaussian noise stability testing, attribute inference privacy leakage audit.
- **Regulatory Compliance**: Automated alignment with **EU AI Act (Article 10)** and **India's NITI Aayog** principles.
- **Automated Model Cards**: Standardized documentation with Intended Use, Limitations, and Fairness Philosophy.

### Model Optimization
- **Threshold Tuning**: Grid search over decision thresholds (0.05–0.95) to find the optimal tradeoff between SPD and accuracy.
- **Retraining Suggestions**: Feature correlation analysis with the sensitive attribute, biased feature detection, and actionable resampling/reweighting strategies.

### Deep Fairness Metrics
- **Intersectional Analysis**: Audits across combined sensitive attributes to detect compounded bias.
- **Calibration Parity**: Binned calibration analysis examining mean prediction vs. mean actual per group.

### Security & Adversarial Auditing
- **Adversarial Robustness**: Directional ±ε·σ perturbation attack on numerical features.
- **Differential Privacy Audit**: Leave-one-out influence analysis estimating ε-DP guarantees.
- **Membership Inference Attack**: Shadow model training to measure training set membership detectability.

### Frontend Improvements
- **Drag-and-Drop Upload**: Drop model/dataset files directly onto upload zones.
- **Dataset Schema Preview**: Server-side column analysis showing data types, unique counts, missing values.
- **Enhanced Report Download**: Comprehensive plain-text report covering all audit dimensions.
- **Framework Badge**: Displays detected ML framework next to audit results.

### Multi-Framework Support
- **Framework Auto-Detection**: Automatically identifies **scikit-learn**, **XGBoost**, **PyTorch**, and **TensorFlow/Keras** models.
- **Multi-Format Model Upload**: Supports `.pkl`, `.joblib`, `.json`/`.ubj` (XGBoost), `.pt`/`.pth` (PyTorch), `.h5`/`.keras` (TensorFlow).
- **Unified Prediction Interface**: Single prediction pipeline that adapts to each framework's conventions.
- **Regression Fairness Metrics**: Mean prediction difference and MAE disparity across groups for regression models.

## Project Structure

```
AI-Shayak/
├── backend/
│   ├── main.py           - Flask API server (5 routes)
│   ├── model.py          - Model generation script (UCI Adult Income)
│   ├── model.pkl         - Pre-trained model
│   ├── test_data.csv     - Test dataset
│   └── requirements.txt  - Python dependencies
├── frontend/
│   ├── app/              - Next.js App Router pages
│   ├── components/       - UI components (shadcn/ui)
│   ├── public/           - Static assets
│   ├── package.json
│   └── tsconfig.json
├── start.bat             - Launch script
└── README.md
```

## Usage
1. **Select Audit Scope**: Choose "End-to-End Audit" (Model + Dataset) or "Data-Only Ethics Audit".
2. **Upload Artifacts** (drag-and-drop or click):
   - **Model**: `.pkl`, `.joblib`, `.json`, `.ubj`, `.pt`, `.pth`, `.h5`, `.keras`
   - **Dataset**: `.csv` with target column (`income` or binary target)
3. **Define Sensitive Attribute**: Specify the protected group column (e.g., `sex`, `race`, `age`).
4. **Run Ethics Evaluation**: Multi-pillar audit with fairness, robustness, privacy, and compliance.
5. **Explore Advanced Analysis**:
   - **Optimization** tab: Tune thresholds, get retraining suggestions, regression fairness.
   - **Deep Fairness** tab: Intersectional analysis across multiple attributes, calibration parity.
   - **Security** tab: Adversarial robustness, differential privacy, membership inference.
6. **Download Full Report**: Comprehensive text report covering all audit dimensions.

## How it Works
The platform is built on core AI ethics and machine learning principles:
- **Demographic Parity**: $P(\hat{Y}=1 | G=0) = P(\hat{Y}=1 | G=1)$
- **Disparate Impact**: $\frac{P(\hat{Y}=1 | G=unprivileged)}{P(\hat{Y}=1 | G=privileged)}$ (Applying the 80% rule)
- **Equal Opportunity**: Ensuring $TPR_{unprivileged} = TPR_{privileged}$ through threshold calibration.
- **Intersectional Fairness**: Auditing compounded bias across multiple demographic dimensions.
- **Membership Inference**: Shadow model confidence gap analysis for privacy risk assessment.

## Installation & Setup

### Backend (Python)
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```

### Generate Demo Files
```bash
cd backend
python model.py
```
This trains a LogisticRegression model on the UCI Adult Income dataset and saves `model.pkl` + `test_data.csv`.

---

*Empowering Ethical Machine Learning*
