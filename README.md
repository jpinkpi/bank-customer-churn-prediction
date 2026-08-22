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

## Repository structure

```text
.
├── data/README.md
├── src/train.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --data data/churn.csv
```

The input must include an `Exited` target column. Common identifier columns such as `RowNumber`, `CustomerId` and `Surname` are removed automatically when present.

## Decision perspective

F1 is used because the target is imbalanced and both missed churners and false alarms matter. In a production setting, the final threshold should be selected with explicit retention-campaign costs rather than a technical metric alone.

## Academic context

Developed as part of the TripleTen Data Science program and reorganized as a portfolio project by José Pablo Rivera Villa.

## Data

The original educational dataset is not redistributed here. See `data/README.md` for the expected schema.
