# F1 EDA and Pit Stop Prediction

## Overview
An end-to-end data science project combining exploratory data analysis (EDA) 
and machine learning on a real Formula 1 strategy dataset. The project analyses 
52,471 laps across 25 races from the 2023 and 2024 F1 seasons, then builds a 
model to predict whether a driver will pit on the next lap.

## Dataset
- **Source:** Kaggle — F1 Strategy Dataset v2
- **Size:** 52,471 rows, 16 columns
- **Coverage:** 25 drivers, 25 races, 2023 and 2024 seasons
- **Columns include:** Driver, LapNumber, Compound, TyreLife, Position, 
LapTime (s), Race, Year, LapTime_Delta, Cumulative_Degradation, PitStop, 
PitNextLap, RaceProgress, Normalized_TyreLife, Position_Change

## Key Findings
- Lap time variation (LapTime_Delta) is the strongest predictor of an 
upcoming pit stop, more so than tyre age alone
- Hard tyres account for roughly 50% of all laps across both seasons
- The Austrian Grand Prix recorded the most pit stops while Monaco 
recorded the fewest
- Full EDA findings and visualisations are documented in the notebook

## Machine Learning — Pit Stop Prediction

### Problem
Binary classification — predict whether a driver will pit on the next lap (PitNextLap: 0 or 1)

### Features Used
- LapTime_Delta, Cumulative_Degradation, LapTime (s), TyreLife
- RaceProgress, Position, Stint, Compound

### Algorithm
Random Forest Classifier — 100 decision trees, majority vote prediction

### Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 — Will not pit | 0.96 | 0.98 | 0.97 |
| 1 — Will pit next lap | 0.92 | 0.82 | 0.87 |

**Overall Accuracy: 95%**

### Feature Importance
LapTime_Delta was the most important predictor (0.235), followed by 
Cumulative_Degradation (0.162) and LapTime (s) (0.138). Compound type 
was the least important feature (0.041).

## Tools Used
- Python 3.14
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook (VS Code)

## Key Learnings
- Real datasets with genuine patterns allow machine learning models to 
identify meaningful signals — the 95% accuracy reflects genuine learnable 
patterns in pit stop behaviour
- Feature importance analysis reveals what the model actually learned, 
not just how accurate it is
- Outlier detection is necessary even on clean datasets — 16 extreme laps 
above 200 seconds were identified and filtered before analysis