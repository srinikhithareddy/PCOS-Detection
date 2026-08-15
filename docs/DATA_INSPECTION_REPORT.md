# PCOS Dataset Inspection Report

## Overview
This report provides a comprehensive inspection of the PCOS detection dataset, including both ultrasound images and clinical data.

---

## 1. Ultrasound Dataset Structure

### Dataset Location
- **Path**: `c:\PCOS Filles\PCOS-Detection\data\raw\UltraSound\Ovarian_US\`

### Categories and Image Counts
The ultrasound dataset is organized into 5 categories:

| Category | Image Count | Notes |
|----------|-------------|-------|
| dominant_follicle | 1,296 | Excludes desktop.ini |
| healthy | 1,344 | Excludes desktop.ini |
| poly_cyst | 1,422 | Excludes desktop.ini |
| simple_cyst | 1,326 | Excludes desktop.ini |
| complex_cyst | 1,368 | Excludes desktop.ini |
| **Total** | **6,756** | Total ultrasound images |

### File Format
- All images are in `.jpg` format
- Each category contains a `desktop.ini` file (system file, excluded from counts)
- Images are sequentially numbered (e.g., dominant_follicle_0001.jpg)

---

## 2. Clinical Dataset Structure

### Dataset Location
- **Path**: `c:\PCOS Filles\PCOS-Detection\data\raw\Clinical\PCOS Dataset.xlsx`
- **Format**: Excel (.xlsx)
- **Shape**: 468 rows × 52 columns

### Column List (52 total)
1. Age
2. Height_cm
3. Weight_kg
4. BMI
5. Waist_Circumference_cm
6. Hip_Circumference_cm
7. Waist_Hip_Ratio
8. Age_at_Menarche
9. Menstrual_Cycle_Length_days
10. Menstrual_Irregularity
11. Gravidity
12. Parity
13. Hirsutism_Score_FG
14. Acne_Severity
15. Alopecia
16. Skin_Darkening_Acanthosis
17. Blood_Pressure_Systolic
18. Blood_Pressure_Diastolic
19. Physical_Activity_Level
20. Smoking_Status
21. Alcohol_Intake
22. Dietary_Sugar_Intake
23. Sleep_Hours
24. FSH_mIU_mL
25. LH_mIU_mL
26. LH_FSH_Ratio
27. Total_Testosterone_ng_dL
28. Free_Testosterone_pg_mL
29. DHEAS_ug_dL
30. Prolactin_ng_mL
31. Estradiol_pg_mL
32. Progesterone_ng_mL
33. SHBG_nmol_L
34. Fasting_Glucose_mg_dL
35. Fasting_Insulin_uIU_mL
36. HOMA_IR
37. HbA1c_percent
38. Total_Cholesterol_mg_dL
39. HDL_mg_dL
40. LDL_mg_dL
41. Triglycerides_mg_dL
42. Ovary_Volume_Left_cm3
43. Ovary_Volume_Right_cm3
44. Follicle_Count_Left
45. Follicle_Count_Right
46. CRP_mg_L
47. ALT_U_L
48. AST_U_L
49. TSH_uIU_mL
50. Vitamin_D_ng_mL
51. Hemoglobin_g_dL
52. **PCOS_Diagnosis** (Target Variable)

### Data Types
- **Integer (int64)**: 26 columns
- **Float (float64)**: 26 columns
- **No categorical/string columns**: All data is numeric

---

## 3. Clinical Features Analysis

### Feature Count
- **Total columns**: 52
- **Target variable**: PCOS_Diagnosis (1 column)
- **Feature columns**: 51 columns

### Potentially Excluded Features (Derived/Calculated)
The following 4 features are derived/calculated from other features and may be excluded from modeling to avoid multicollinearity:

1. **BMI** - Calculated from Height_cm and Weight_kg
2. **Waist_Hip_Ratio** - Calculated from Waist_Circumference_cm and Hip_Circumference_cm
3. **LH_FSH_Ratio** - Calculated from LH_mIU_mL and FSH_mIU_mL
4. **HOMA_IR** - Calculated from Fasting_Glucose_mg_dL and Fasting_Insulin_uIU_mL

**Remaining features for modeling**: 47 (51 - 4 excluded)

---

## 4. Missing Values Analysis

### Summary
- **Total missing values**: 0
- **Columns with missing values**: 0
- **Data completeness**: 100%

**Conclusion**: The clinical dataset has no missing values, which is excellent for modeling.

---

## 5. Negative Values (Range Violations)

### Summary
Negative values were detected in 18 columns, which may indicate data quality issues or measurement errors:

| Column | Negative Values |
|--------|-----------------|
| LH_mIU_mL | 5 |
| LH_FSH_Ratio | 2 |
| Free_Testosterone_pg_mL | 5 |
| DHEAS_ug_dL | 5 |
| Prolactin_ng_mL | 6 |
| Estradiol_pg_mL | 4 |
| Progesterone_ng_mL | 19 |
| SHBG_nmol_L | 1 |
| Fasting_Insulin_uIU_mL | 12 |
| HOMA_IR | 7 |
| Triglycerides_mg_dL | 2 |
| Ovary_Volume_Right_cm3 | 1 |
| CRP_mg_L | 37 |
| ALT_U_L | 4 |
| AST_U_L | 6 |
| TSH_uIU_mL | 6 |
| Vitamin_D_ng_mL | 4 |

**Total negative values**: 121 across 18 columns

**Recommendation**: These negative values should be investigated and either:
- Treated as missing values
- Replaced with valid measurements
- Clipped to minimum valid ranges

---

## 6. Outliers Analysis (IQR Method)

### Summary
Outliers were detected in 38 columns using the Interquartile Range (IQR) method (values outside Q1-1.5×IQR or Q3+1.5×IQR):

| Column | Outlier Count |
|--------|---------------|
| Height_cm | 4 |
| Weight_kg | 7 |
| BMI | 2 |
| Waist_Circumference_cm | 3 |
| Hip_Circumference_cm | 9 |
| Waist_Hip_Ratio | 3 |
| Menstrual_Cycle_Length_days | 4 |
| Alopecia | 103 |
| Blood_Pressure_Systolic | 1 |
| Blood_Pressure_Diastolic | 1 |
| Smoking_Status | 35 |
| Alcohol_Intake | 60 |
| Sleep_Hours | 2 |
| FSH_mIU_mL | 6 |
| LH_mIU_mL | 2 |
| LH_FSH_Ratio | 2 |
| Total_Testosterone_ng_dL | 1 |
| Free_Testosterone_pg_mL | 1 |
| DHEAS_ug_dL | 4 |
| Prolactin_ng_mL | 9 |
| Estradiol_pg_mL | 2 |
| Progesterone_ng_mL | 2 |
| Fasting_Glucose_mg_dL | 5 |
| Fasting_Insulin_uIU_mL | 2 |
| HOMA_IR | 5 |
| HbA1c_percent | 4 |
| Total_Cholesterol_mg_dL | 2 |
| LDL_mg_dL | 1 |
| Triglycerides_mg_dL | 2 |
| Ovary_Volume_Left_cm3 | 4 |
| Ovary_Volume_Right_cm3 | 3 |
| CRP_mg_L | 5 |
| ALT_U_L | 2 |
| AST_U_L | 5 |
| TSH_uIU_mL | 3 |
| Vitamin_D_ng_mL | 4 |
| Hemoglobin_g_dL | 3 |

**Note**: High outlier counts in categorical/binary features (Alopecia: 103, Smoking_Status: 35, Alcohol_Intake: 60) may reflect class imbalance rather than data quality issues.

---

## 7. PCOS_Diagnosis Distribution

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| 0 (No PCOS) | 260 | 55.56% |
| 1 (PCOS) | 208 | 44.44% |

**Total samples**: 468

**Conclusion**: The dataset is relatively balanced with a slight bias toward non-PCOS cases (55.56% vs 44.44%). This balance is acceptable for most machine learning models without requiring significant class balancing techniques.

---

## 8. Ultrasound-Clinical Patient Mapping

### Findings
- **Clinical dataset**: 468 patients (rows)
- **Ultrasound dataset**: 6,756 images across 5 categories
- **Patient ID column**: No explicit patient identifier found in clinical dataset
- **Mapping capability**: Cannot establish direct patient-to-image mapping without patient IDs

**Conclusion**: The ultrasound and clinical datasets appear to be independent collections. The ultrasound images are categorized by ovarian condition but are not directly linked to specific patients in the clinical dataset. For multimodal learning, a patient ID mapping would need to be established or the datasets would need to be used separately.

---

## 9. Data Quality Summary

### Strengths
- ✅ No missing values in clinical dataset
- ✅ Balanced target variable distribution (55.56% vs 44.44%)
- ✅ Large ultrasound dataset (6,756 images)
- ✅ Well-organized ultrasound categories
- ✅ Comprehensive clinical features (51 features)

### Issues to Address
- ⚠️ **Negative values**: 121 negative values across 18 clinical features (likely data entry errors)
- ⚠️ **Outliers**: Significant outliers in 38 features (may require investigation)
- ⚠️ **No patient mapping**: Ultrasound and clinical datasets cannot be linked at patient level
- ⚠️ **Derived features**: 4 calculated features may cause multicollinearity

### Recommendations
1. **Data Cleaning**:
   - Investigate and handle negative values (replace with valid ranges or treat as missing)
   - Review outliers in continuous variables for data entry errors
   - Consider removing or transforming derived features (BMI, Waist_Hip_Ratio, LH_FSH_Ratio, HOMA_IR)

2. **Feature Engineering**:
   - Use 47 base features (excluding 4 derived features) for modeling
   - Consider feature scaling for clinical variables
   - Evaluate feature importance for dimensionality reduction

3. **Modeling Strategy**:
   - Use clinical data and ultrasound data separately initially
   - Consider ensemble approaches combining both modalities
   - Address class imbalance if needed (though current balance is acceptable)

4. **Data Validation**:
   - Establish data validation pipeline for future data collection
   - Define valid ranges for each clinical feature
   - Implement patient ID system for linking ultrasound and clinical data

---

## 10. Dataset Statistics Summary

### Clinical Dataset
- **Samples**: 468 patients
- **Features**: 51 (47 base + 4 derived)
- **Target**: PCOS_Diagnosis (binary)
- **Missing values**: 0
- **Data completeness**: 100%

### Ultrasound Dataset
- **Total images**: 6,756
- **Categories**: 5 (dominant_follicle, healthy, poly_cyst, simple_cyst, complex_cyst)
- **Format**: JPG images
- **Average images per category**: ~1,351

---

**Report Generated**: 2025
**Dataset Version**: Initial inspection
