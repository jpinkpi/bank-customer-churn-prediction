# Bank Customer Churn Prediction

Machine-learning project for identifying bank customers at risk of leaving. The analysis compares classification models, accounts for class imbalance and selects a decision threshold using the validation-set F1 score.

## Business question

Customer retention is usually less expensive than acquisition. A churn model can help a bank prioritize retention actions, but the useful threshold depends on the relative cost of a missed churner and an unnecessary intervention.

## Approach

- Remove identifiers that should not drive predictions.
- Split the data into training, validation and test sets using stratification.
- Preprocess numeric and categorical features with a reproducible pipeline.
- Compare logistic regression, decision tree and random forest models.
- Address class imbalance with class weights.
- Choose the probability threshold on validation data only.
- Report F1, precision, recall, accuracy and ROC-AUC on the untouched test set.

## Verified results

The pipeline was executed against the churn dataset on August 24, 2026.

### Validation comparison

| Model | Validation F1 | Selected threshold |
|---|---:|---:|
| Logistic regression | 0.538 | 0.60 |
| Decision tree | 0.615 | 0.68 |
| Random forest | **0.626** | **0.51** |

The random forest was selected using validation performance only.

### Final test performance

| Metric | Score |
|---|---:|
| F1 | **0.598** |
| Precision | 0.576 |
| Recall | 0.622 |
| Accuracy | 0.830 |
| ROC-AUC | **0.849** |

The ROC-AUC indicates good ability to rank higher-risk customers, while the lower F1 shows that converting those probabilities into individual retention decisions remains challenging. The test F1 is also below validation F1, so the results should be presented as a credible baseline rather than an optimized production model.

## Repository structure

```text
.
├── data/
│   ├── churn.csv
│   └── README.md
├── src/train.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
source .venv/Scripts/activate      # Git Bash on Windows
python -m pip install -r requirements.txt
python src/train.py --data data/churn.csv
```

The input must include an `Exited` target column. Common identifier columns such as `RowNumber`, `CustomerId` and `Surname` are removed automatically when present.

## Decision perspective

F1 is used because the target is imbalanced and both missed churners and false alarms matter. In a production setting, the final threshold should be selected with explicit retention-campaign costs rather than a technical metric alone. A probability ranking can be economically useful even when a universal classification threshold is imperfect.

## Academic context

Developed as part of the TripleTen Data Science program and reorganized as a portfolio project by José Pablo Rivera Villa.

## Data

The repository currently includes the educational churn dataset required to reproduce the analysis. Its use is limited to this academic portfolio context; redistribution rights should be confirmed with the original provider.
