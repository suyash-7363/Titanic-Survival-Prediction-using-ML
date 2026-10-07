# Titanic Survival Prediction — Mini Project

An interactive Streamlit app that predicts whether a Titanic passenger
survived, comparing **Decision Tree, Random Forest, Naive Bayes, and KNN**.

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Keep `app.py` and `titanic_dataset.csv` in the same folder.

## Files
- `app.py` — the Streamlit application
- `titanic_dataset.csv` — sample dataset (synthetic, 900 passengers)
- `generate_dataset.py` — script used to create the sample dataset
- `requirements.txt` — dependencies

## Dataset columns
| Column | Meaning |
|---|---|
| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Sex | male / female |
| Age | Passenger age |
| SibSp | # of siblings/spouses aboard |
| Parch | # of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation (S/C/Q) |
| Survived | Target — 1 = survived, 0 = did not survive |

## The story to tell your professor
"Women, children, and 1st-class passengers had a much higher chance of
survival because of the 'women and children first' evacuation policy and
lifeboat access being closer to 1st-class cabins. The models learn this
same pattern directly from the data."

## Why this topic is quick to explain
- Everyone already knows the Titanic story — no business context needed.
- Only 4 simple, intuitive features drive the outcome (class, sex, age, fare).
- The dataset is almost perfectly balanced (52% survived / 48% did not),
  so accuracy alone is a fair, easy-to-explain metric — no need to get into
  precision/recall trade-offs unless asked.
