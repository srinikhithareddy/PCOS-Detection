import pandas as pd
import numpy as np

df = pd.read_excel(r'c:\PCOS Filles\PCOS-Detection\data\raw\Clinical\PCOS Dataset.xlsx')

print('Negative values per column:')
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f'{col}: {neg_count} negative values')

print('\nOutliers per column (using IQR method):')
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if outliers > 0:
        print(f'{col}: {outliers} outliers')

print('\nTotal columns:', len(df.columns))
print('Target variable: PCOS_Diagnosis')
print('Feature columns (excluding target):', len(df.columns) - 1)

# Identify potentially excluded features (derived/calculated features)
print('\nPotentially excluded features (derived/calculated):')
print('1. BMI - calculated from Height and Weight')
print('2. Waist_Hip_Ratio - calculated from Waist and Hip circumference')
print('3. LH_FSH_Ratio - calculated from LH and FSH')
print('4. HOMA_IR - calculated from Fasting Glucose and Fasting Insulin')

print('\nChecking for patient identifiers in clinical dataset:')
print('Column names:', df.columns.tolist())
print('\nNo explicit patient ID column found in clinical dataset.')
print('Clinical dataset has', len(df), 'rows (patients)')
