"""
Generates a realistic synthetic Titanic-style passenger dataset.
Mirrors the structure of the world-famous 'Titanic - Machine Learning
from Disaster' dataset, with the same well-known survival patterns:
women, children, and 1st class passengers had higher survival odds.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 900

pclass = np.random.choice([1, 2, 3], size=N, p=[0.24, 0.21, 0.55])
sex = np.random.choice(["male", "female"], size=N, p=[0.65, 0.35])
age = np.clip(np.random.normal(29, 13, N), 0.5, 75).round(1)
sibsp = np.random.choice([0, 1, 2, 3, 4], size=N, p=[0.68, 0.20, 0.08, 0.03, 0.01])
parch = np.random.choice([0, 1, 2, 3], size=N, p=[0.76, 0.13, 0.08, 0.03])
fare = np.round(np.clip(np.random.exponential(30, N) + (3 - pclass) * 15, 4, 512), 2)
embarked = np.random.choice(["S", "C", "Q"], size=N, p=[0.72, 0.19, 0.09])

# Build survival probability from the well-known real-world drivers
logit = (
    -1.0
    + 1.8 * (sex == "female")
    + 1.1 * (pclass == 1)
    + 0.4 * (pclass == 2)
    + 0.9 * (age < 12)
    - 0.3 * (age > 60)
    + 0.01 * fare / 10
    - 0.15 * sibsp
    + np.random.normal(0, 0.5, N)
)
prob = 1 / (1 + np.exp(-logit))
survived = (np.random.rand(N) < prob).astype(int)

df = pd.DataFrame({
    "PassengerId": np.arange(1, N + 1),
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked,
    "Survived": survived,
})

df.to_csv("titanic_dataset.csv", index=False)
print("Dataset created:", df.shape)
print(df["Survived"].value_counts(normalize=True))
