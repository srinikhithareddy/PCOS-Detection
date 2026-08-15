"""
Clinical Data Quality Assessment
Implements missing value detection, outlier detection, range checks, and consistency checks
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

from configs.quality_config import QualityConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityCategory(Enum):
    """Clinical data quality categories"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNUSABLE = "unusable"


class ClinicalQualityAssessor:
    """Assesses clinical data quality for PCOS detection"""
    
    def __init__(self, config: Optional[QualityConfig] = None):
        """Initialize the clinical quality assessor"""
        self.config = config or QualityConfig()
        logger.info("ClinicalQualityAssessor initialized")
    
    def assess_clinical_data_quality(
        self,
        clinical_data: Dict[str, Any],
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Perform comprehensive clinical data quality assessment
        
        Args:
            clinical_data: Dictionary containing 47 clinical features
            patient_id: Unique identifier for the patient
            
        Returns:
            Dictionary containing quality assessment results
        """
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame([clinical_data])
            
            # Step 1: Missing value detection
            missing_analysis = self._detect_missing_values(df, clinical_data)
            
            # Step 2: Range validation
            range_analysis = self._validate_ranges(clinical_data)
            
            # Step 3: Outlier detection
            outlier_analysis = self._detect_outliers(clinical_data)
            
            # Step 4: Consistency checks
            consistency_analysis = self._check_consistency(clinical_data)
            
            # Step 5: Calculate reliability score
            reliability_score = self._calculate_reliability_score(
                missing_analysis,
                range_analysis,
                outlier_analysis,
                consistency_analysis
            )
            
            # Step 6: Determine quality category
            quality_category = self._determine_quality_category(reliability_score)
            
            # Step 7: Generate quality report
            report = self._create_quality_report(
                patient_id=patient_id,
                missing_analysis=missing_analysis,
                range_analysis=range_analysis,
                outlier_analysis=outlier_analysis,
                consistency_analysis=consistency_analysis,
                reliability_score=reliability_score,
                quality_category=quality_category
            )
            
            logger.info(f"Clinical quality assessment completed for patient {patient_id}: "
                       f"{quality_category.value} (reliability: {reliability_score:.2f})")
            
            return report
            
        except Exception as e:
            logger.error(f"Error during clinical quality assessment for patient {patient_id}: {str(e)}")
            return self._create_error_report(patient_id, str(e))
    
    def _detect_missing_values(
        self, 
        df: pd.DataFrame,
        clinical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect missing values in clinical data
        
        Args:
            df: DataFrame containing clinical data
            clinical_data: Original clinical data dictionary
            
        Returns:
            Dictionary with missing value analysis results
        """
        missing_features = []
        total_features = len(clinical_data)
        
        for feature, value in clinical_data.items():
            if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
                missing_features.append(feature)
        
        # Check for critical missing features
        critical_missing = [
            f for f in missing_features 
            if f in self.config.CRITICAL_FEATURES_MUST_BE_PRESENT
        ]
        
        missing_percent = (len(missing_features) / total_features) * 100 if total_features > 0 else 0
        
        return {
            'missing_features': missing_features,
            'missing_count': len(missing_features),
            'missing_percent': missing_percent,
            'critical_missing_features': critical_missing,
            'critical_missing_count': len(critical_missing),
            'exceeds_threshold': missing_percent > self.config.MAX_MISSING_VALUES_PERCENT,
            'has_critical_missing': len(critical_missing) > 0
        }
    
    def _validate_ranges(self, clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate clinical features against expected ranges
        
        Args:
            clinical_data: Dictionary containing clinical features
            
        Returns:
            Dictionary with range validation results
        """
        range_violations = []
        
        for feature, value in clinical_data.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue
            
            # Skip NaN values
            if np.isnan(value):
                continue
            
            # Get expected range for this feature
            feature_range = self.config.get_feature_range(feature)
            
            if feature_range:
                min_val = feature_range.get('min')
                max_val = feature_range.get('max')
                
                if min_val is not None and value < min_val:
                    range_violations.append({
                        'feature': feature,
                        'value': value,
                        'expected_min': min_val,
                        'expected_max': max_val,
                        'violation_type': 'below_minimum'
                    })
                elif max_val is not None and value > max_val:
                    range_violations.append({
                        'feature': feature,
                        'value': value,
                        'expected_min': min_val,
                        'expected_max': max_val,
                        'violation_type': 'above_maximum'
                    })
        
        return {
            'range_violations': range_violations,
            'violation_count': len(range_violations),
            'violated_features': [v['feature'] for v in range_violations]
        }
    
    def _detect_outliers(self, clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect outliers in clinical data using configured method
        
        Args:
            clinical_data: Dictionary containing clinical features
            
        Returns:
            Dictionary with outlier detection results
        """
        outliers = []
        numeric_features = {}
        
        # Extract numeric features
        for feature, value in clinical_data.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                numeric_features[feature] = value
        
        if len(numeric_features) == 0:
            return {
                'outliers': [],
                'outlier_count': 0,
                'outlier_features': [],
                'method_used': self.config.OUTLIER_DETECTION_METHOD
            }
        
        # Convert to series for statistical analysis
        values = np.array(list(numeric_features.values()))
        features = list(numeric_features.keys())
        
        if self.config.OUTLIER_DETECTION_METHOD == 'iqr':
            outliers = self._detect_outliers_iqr(values, features, numeric_features)
        elif self.config.OUTLIER_DETECTION_METHOD == 'zscore':
            outliers = self._detect_outliers_zscore(values, features, numeric_features)
        else:
            # Default to IQR
            outliers = self._detect_outliers_iqr(values, features, numeric_features)
        
        return {
            'outliers': outliers,
            'outlier_count': len(outliers),
            'outlier_features': [o['feature'] for o in outliers],
            'method_used': self.config.OUTLIER_DETECTION_METHOD
        }
    
    def _detect_outliers_iqr(
        self, 
        values: np.ndarray, 
        features: List[str],
        numeric_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect outliers using IQR method"""
        outliers = []
        
        if len(values) < 4:  # Need minimum data for IQR
            return outliers
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return outliers
        
        lower_bound = q1 - (self.config.IQR_MULTIPLIER * iqr)
        upper_bound = q3 + (self.config.IQR_MULTIPLIER * iqr)
        
        for feature, value in numeric_features.items():
            if value < lower_bound or value > upper_bound:
                outliers.append({
                    'feature': feature,
                    'value': value,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'method': 'iqr'
                })
        
        return outliers
    
    def _detect_outliers_zscore(
        self, 
        values: np.ndarray, 
        features: List[str],
        numeric_features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Detect outliers using Z-score method"""
        outliers = []
        
        if len(values) < 2:
            return outliers
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return outliers
        
        for feature, value in numeric_features.items():
            z_score = abs((value - mean) / std)
            if z_score > self.config.Z_SCORE_THRESHOLD:
                outliers.append({
                    'feature': feature,
                    'value': value,
                    'z_score': z_score,
                    'threshold': self.config.Z_SCORE_THRESHOLD,
                    'method': 'zscore'
                })
        
        return outliers
    
    def _check_consistency(self, clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform consistency checks on calculated vs reported values
        
        Args:
            clinical_data: Dictionary containing clinical features
            
        Returns:
            Dictionary with consistency check results
        """
        consistency_violations = []
        
        # BMI consistency check
        if self.config.CONSISTENCY_RULES.get('BMI_calculation'):
            bmi_violation = self._check_bmi_consistency(clinical_data)
            if bmi_violation:
                consistency_violations.append(bmi_violation)
        
        # LH/FSH ratio consistency check
        if self.config.CONSISTENCY_RULES.get('LH_FSH_calculation'):
            lh_fsh_violation = self._check_lh_fsh_consistency(clinical_data)
            if lh_fsh_violation:
                consistency_violations.append(lh_fsh_violation)
        
        # Waist-Hip ratio consistency check
        if self.config.CONSISTENCY_RULES.get('Waist_Hip_calculation'):
            waist_hip_violation = self._check_waist_hip_consistency(clinical_data)
            if waist_hip_violation:
                consistency_violations.append(waist_hip_violation)
        
        # HOMA-IR consistency check
        if self.config.CONSISTENCY_RULES.get('HOMA_IR_calculation'):
            homa_ir_violation = self._check_homa_ir_consistency(clinical_data)
            if homa_ir_violation:
                consistency_violations.append(homa_ir_violation)
        
        return {
            'consistency_violations': consistency_violations,
            'violation_count': len(consistency_violations),
            'violated_checks': [v['check_type'] for v in consistency_violations]
        }
    
    def _check_bmi_consistency(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if BMI matches height and weight"""
        try:
            height = data.get('Height_cm')
            weight = data.get('Weight_kg')
            reported_bmi = data.get('BMI')
            
            if None in [height, weight, reported_bmi]:
                return None
            
            # Calculate expected BMI
            height_m = height / 100  # Convert to meters
            calculated_bmi = weight / (height_m ** 2)
            
            # Check if within tolerance
            tolerance = self.config.CALCULATION_TOLERANCE_PERCENT / 100
            diff_percent = abs(calculated_bmi - reported_bmi) / reported_bmi
            
            if diff_percent > tolerance:
                return {
                    'check_type': 'BMI_calculation',
                    'reported_value': reported_bmi,
                    'calculated_value': calculated_bmi,
                    'difference_percent': diff_percent * 100,
                    'tolerance_percent': self.config.CALCULATION_TOLERANCE_PERCENT
                }
        except (ZeroDivisionError, TypeError):
            pass
        
        return None
    
    def _check_lh_fsh_consistency(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if LH/FSH ratio matches individual values"""
        try:
            lh = data.get('LH_mIU_mL')
            fsh = data.get('FSH_mIU_mL')
            reported_ratio = data.get('LH_FSH_Ratio')
            
            if None in [lh, fsh, reported_ratio] or fsh == 0:
                return None
            
            calculated_ratio = lh / fsh
            
            tolerance = self.config.CALCULATION_TOLERANCE_PERCENT / 100
            diff_percent = abs(calculated_ratio - reported_ratio) / reported_ratio
            
            if diff_percent > tolerance:
                return {
                    'check_type': 'LH_FSH_calculation',
                    'reported_value': reported_ratio,
                    'calculated_value': calculated_ratio,
                    'difference_percent': diff_percent * 100,
                    'tolerance_percent': self.config.CALCULATION_TOLERANCE_PERCENT
                }
        except (ZeroDivisionError, TypeError):
            pass
        
        return None
    
    def _check_waist_hip_consistency(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if waist-hip ratio matches measurements"""
        try:
            waist = data.get('Waist_Circumference_cm')
            hip = data.get('Hip_Circumference_cm')
            reported_ratio = data.get('Waist_Hip_Ratio')
            
            if None in [waist, hip, reported_ratio] or hip == 0:
                return None
            
            calculated_ratio = waist / hip
            
            tolerance = self.config.CALCULATION_TOLERANCE_PERCENT / 100
            diff_percent = abs(calculated_ratio - reported_ratio) / reported_ratio
            
            if diff_percent > tolerance:
                return {
                    'check_type': 'Waist_Hip_calculation',
                    'reported_value': reported_ratio,
                    'calculated_value': calculated_ratio,
                    'difference_percent': diff_percent * 100,
                    'tolerance_percent': self.config.CALCULATION_TOLERANCE_PERCENT
                }
        except (ZeroDivisionError, TypeError):
            pass
        
        return None
    
    def _check_homa_ir_consistency(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if HOMA-IR matches glucose and insulin"""
        try:
            glucose = data.get('Fasting_Glucose_mg_dL')
            insulin = data.get('Fasting_Insulin_uIU_mL')
            reported_homa = data.get('HOMA_IR')
            
            if None in [glucose, insulin, reported_homa]:
                return None
            
            # HOMA-IR = (Glucose × Insulin) / 405
            calculated_homa = (glucose * insulin) / 405
            
            tolerance = self.config.CALCULATION_TOLERANCE_PERCENT / 100
            diff_percent = abs(calculated_homa - reported_homa) / reported_homa
            
            if diff_percent > tolerance:
                return {
                    'check_type': 'HOMA_IR_calculation',
                    'reported_value': reported_homa,
                    'calculated_value': calculated_homa,
                    'difference_percent': diff_percent * 100,
                    'tolerance_percent': self.config.CALCULATION_TOLERANCE_PERCENT
                }
        except (ZeroDivisionError, TypeError):
            pass
        
        return None
    
    def _calculate_reliability_score(
        self,
        missing_analysis: Dict[str, Any],
        range_analysis: Dict[str, Any],
        outlier_analysis: Dict[str, Any],
        consistency_analysis: Dict[str, Any]
    ) -> float:
        """
        Calculate overall clinical data reliability score
        
        Args:
            missing_analysis: Missing value analysis results
            range_analysis: Range validation results
            outlier_analysis: Outlier detection results
            consistency_analysis: Consistency check results
            
        Returns:
            Reliability score between 0.0 and 1.0
        """
        weights = self.config.RELIABILITY_WEIGHTS
        
        # Missing value score (inverse of missing percent)
        missing_score = 1.0 - (missing_analysis['missing_percent'] / 100.0)
        if missing_analysis['has_critical_missing']:
            missing_score = 0.0
        
        # Range violation score (inverse of violation ratio)
        total_features = 47  # Total clinical features
        range_score = 1.0 - (range_analysis['violation_count'] / total_features)
        
        # Outlier score (inverse of outlier ratio)
        outlier_score = 1.0 - (outlier_analysis['outlier_count'] / total_features)
        
        # Consistency score (inverse of violation ratio)
        total_checks = 4  # BMI, LH/FSH, Waist/Hip, HOMA-IR
        consistency_score = 1.0 - (consistency_analysis['violation_count'] / total_checks)
        
        # Weighted average
        reliability_score = (
            weights['missing_values'] * missing_score +
            weights['range_violations'] * range_score +
            weights['outliers'] * outlier_score +
            weights['consistency_violations'] * consistency_score
        )
        
        # Ensure score is between 0 and 1
        reliability_score = max(0.0, min(1.0, reliability_score))
        
        return reliability_score
    
    def _determine_quality_category(self, reliability_score: float) -> DataQualityCategory:
        """
        Determine quality category based on reliability score
        
        Args:
            reliability_score: Calculated reliability score (0-1)
            
        Returns:
            DataQualityCategory enum value
        """
        if reliability_score >= 0.8:
            return DataQualityCategory.HIGH
        elif reliability_score >= 0.6:
            return DataQualityCategory.MEDIUM
        elif reliability_score >= 0.4:
            return DataQualityCategory.LOW
        else:
            return DataQualityCategory.UNUSABLE
    
    def _create_quality_report(
        self,
        patient_id: str,
        missing_analysis: Dict[str, Any],
        range_analysis: Dict[str, Any],
        outlier_analysis: Dict[str, Any],
        consistency_analysis: Dict[str, Any],
        reliability_score: float,
        quality_category: DataQualityCategory
    ) -> Dict[str, Any]:
        """
        Create comprehensive clinical quality report
        
        Args:
            patient_id: Unique patient identifier
            missing_analysis: Missing value analysis results
            range_analysis: Range validation results
            outlier_analysis: Outlier detection results
            consistency_analysis: Consistency check results
            reliability_score: Calculated reliability score
            quality_category: Determined quality category
            
        Returns:
            Clinical quality report dictionary
        """
        return {
            'patient_id': patient_id,
            'assessment_timestamp': self._get_timestamp(),
            'reliability_score': reliability_score,
            'quality_category': quality_category.value,
            'missing_value_analysis': missing_analysis,
            'range_validation': range_analysis,
            'outlier_detection': outlier_analysis,
            'consistency_checks': consistency_analysis,
            'assessment_status': 'completed',
            'requires_manual_review': quality_category in [DataQualityCategory.LOW, DataQualityCategory.UNUSABLE]
        }
    
    def _create_error_report(self, patient_id: str, error_message: str) -> Dict[str, Any]:
        """
        Create error report for failed assessments
        
        Args:
            patient_id: Unique patient identifier
            error_message: Error description
            
        Returns:
            Error report dictionary
        """
        return {
            'patient_id': patient_id,
            'assessment_timestamp': self._get_timestamp(),
            'reliability_score': 0.0,
            'quality_category': DataQualityCategory.UNUSABLE.value,
            'missing_value_analysis': {},
            'range_validation': {},
            'outlier_detection': {},
            'consistency_checks': {},
            'assessment_status': 'failed',
            'error_message': error_message,
            'requires_manual_review': True
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
