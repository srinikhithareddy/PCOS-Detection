"""
Unit tests for quality assessment module
"""

import pytest
import numpy as np
from PIL import Image
import io
from quality_assessment import ImageQualityAssessor, ClinicalQualityAssessor
from quality_config import QualityConfig


class TestImageQualityAssessor:
    """Test suite for ImageQualityAssessor"""
    
    @pytest.fixture
    def assessor(self):
        """Create an instance of ImageQualityAssessor"""
        return ImageQualityAssessor()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple RGB image
        img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.read()
    
    @pytest.fixture
    def corrupted_image(self):
        """Create a corrupted image"""
        return b"This is not an image"
    
    def test_assessor_initialization(self, assessor):
        """Test that assessor initializes correctly"""
        assert assessor is not None
        assert assessor.config is not None
    
    def test_basic_image_validation(self, assessor, sample_image):
        """Test basic image validation"""
        result = assessor._validate_basic_image_properties(
            sample_image, 
            "test_image.png"
        )
        assert result['valid'] is True
        assert result['reason'] is None
    
    def test_invalid_file_format(self, assessor, sample_image):
        """Test rejection of invalid file formats"""
        result = assessor._validate_basic_image_properties(
            sample_image,
            "test_image.exe"
        )
        assert result['valid'] is False
        assert "Unsupported file format" in result['reason']
    
    def test_corrupted_image_detection(self, assessor, corrupted_image):
        """Test detection of corrupted images"""
        result = assessor._validate_basic_image_properties(
            corrupted_image,
            "corrupted.png"
        )
        assert result['valid'] is False
        assert "corrupted" in result['reason'].lower()
    
    def test_image_loading(self, assessor, sample_image):
        """Test image loading functionality"""
        image = assessor._load_image(sample_image)
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # BGR format
    
    def test_quality_metrics_calculation(self, assessor, sample_image):
        """Test quality metrics calculation"""
        image = assessor._load_image(sample_image)
        metrics = assessor._calculate_quality_metrics(image)
        
        assert 'quality_score' in metrics
        assert 'sharpness' in metrics
        assert 'mean_brightness' in metrics
        assert 'std_brightness' in metrics
        assert 'contrast' in metrics
        
        # Check that values are reasonable
        assert metrics['quality_score'] is not None
        assert metrics['sharpness'] >= 0
        assert metrics['mean_brightness'] >= 0
        assert metrics['std_brightness'] >= 0
        assert 0 <= metrics['quality_score'] <= 100
    
    def test_quality_decision_good(self, assessor):
        """Test quality decision for good quality image"""
        # Simulate good quality score (higher is better)
        metrics = {'quality_score': 75.0}
        category, decision = assessor._determine_quality_decision(metrics)
        
        assert category.value == 'good'
        assert decision.value == 'continue'
    
    def test_quality_decision_poor(self, assessor):
        """Test quality decision for poor quality image"""
        # Simulate poor quality score
        metrics = {'quality_score': 45.0}
        category, decision = assessor._determine_quality_decision(metrics)
        
        assert category.value == 'poor'
        assert decision.value == 'enhance'
    
    def test_quality_decision_unusable(self, assessor):
        """Test quality decision for unusable image"""
        # Simulate unusable quality score
        metrics = {'quality_score': 20.0}
        category, decision = assessor._determine_quality_decision(metrics)
        
        assert category.value == 'unusable'
        assert decision.value == 'flag'
    
    def test_complete_quality_assessment(self, assessor, sample_image):
        """Test complete quality assessment workflow"""
        report = assessor.assess_image_quality(
            image_data=sample_image,
            image_id="test_001",
            filename="test_image.png"
        )
        
        assert report['image_id'] == "test_001"
        assert report['filename'] == "test_image.png"
        assert 'quality_category' in report
        assert 'processing_decision' in report
        assert 'quality_metrics' in report
        assert report['assessment_status'] in ['completed', 'failed']
    
    def test_unusable_image_report(self, assessor, corrupted_image):
        """Test report generation for unusable images"""
        report = assessor.assess_image_quality(
            image_data=corrupted_image,
            image_id="test_002",
            filename="corrupted.png"
        )
        
        assert report['quality_category'] == 'unusable'
        assert report['processing_decision'] == 'flag'
        assert report['flagged_for_review'] is True
        assert report['assessment_status'] == 'failed'


class TestClinicalQualityAssessor:
    """Test suite for ClinicalQualityAssessor"""
    
    @pytest.fixture
    def assessor(self):
        """Create an instance of ClinicalQualityAssessor"""
        return ClinicalQualityAssessor()
    
    @pytest.fixture
    def sample_clinical_data(self):
        """Create sample clinical data"""
        return {
            # Demographics
            'Age': 28.0,
            'Height_cm': 165.0,
            'Weight_kg': 65.0,
            'BMI': 23.9,
            'Waist_Circumference_cm': 80.0,
            'Hip_Circumference_cm': 100.0,
            'Waist_Hip_Ratio': 0.8,
            
            # Menstrual & Reproductive
            'Age_at_Menarche': 12.0,
            'Menstrual_Cycle_Length_days': 28.0,
            'Menstrual_Irregularity': 1,
            'Gravidity': 0,
            'Parity': 0,
            
            # Physical Signs
            'Hirsutism_Score_FG': 8,
            'Acne_Severity': 1,
            'Alopecia': 0,
            'Skin_Darkening_Acanthosis': 0,
            
            # Vitals & Lifestyle
            'Blood_Pressure_Systolic': 120,
            'Blood_Pressure_Diastolic': 80,
            'Physical_Activity_Level': 1,
            'Smoking_Status': 0,
            'Alcohol_Intake': 0,
            'Dietary_Sugar_Intake': 1,
            'Sleep_Hours': 7.5,
            
            # Hormonal Panel
            'FSH_mIU_mL': 5.0,
            'LH_mIU_mL': 10.0,
            'LH_FSH_Ratio': 2.0,
            'Total_Testosterone_ng_dL': 45.0,
            'Free_Testosterone_pg_mL': 2.5,
            'DHEAS_ug_dL': 150.0,
            'Prolactin_ng_mL': 15.0,
            'Estradiol_pg_mL': 60.0,
            'Progesterone_ng_mL': 1.0,
            'SHBG_nmol_L': 50.0,
            
            # Metabolic Panel
            'Fasting_Glucose_mg_dL': 95.0,
            'Fasting_Insulin_uIU_mL': 8.0,
            'HOMA_IR': 1.9,
            'HbA1c_percent': 5.5,
            'Total_Cholesterol_mg_dL': 180.0,
            'HDL_mg_dL': 50.0,
            'LDL_mg_dL': 110.0,
            'Triglycerides_mg_dL': 100.0,
            
            # Other Labs
            'CRP_mg_L': 2.0,
            'ALT_U_L': 25.0,
            'AST_U_L': 22.0,
            'TSH_uIU_mL': 2.5,
            'Vitamin_D_ng_mL': 30.0,
            'Hemoglobin_g_dL': 13.0,
        }
    
    @pytest.fixture
    def clinical_data_with_missing(self):
        """Create clinical data with missing values"""
        data = self.sample_clinical_data(self)
        data['Age'] = None
        data['BMI'] = ""
        return data
    
    @pytest.fixture
    def clinical_data_with_outliers(self):
        """Create clinical data with outliers"""
        data = self.sample_clinical_data(self)
        data['Age'] = 150.0  # Unrealistic age
        return data
    
    def test_assessor_initialization(self, assessor):
        """Test that assessor initializes correctly"""
        assert assessor is not None
        assert assessor.config is not None
    
    def test_missing_value_detection(self, assessor):
        """Test missing value detection"""
        import pandas as pd
        
        clinical_data_with_missing = {
            'Age': None,
            'BMI': "",
            'Height_cm': 165.0,
            'Weight_kg': 65.0,
        }
        
        df = pd.DataFrame([clinical_data_with_missing])
        result = assessor._detect_missing_values(df, clinical_data_with_missing)
        
        assert result['missing_count'] >= 2
        assert 'Age' in result['missing_features']
        assert 'BMI' in result['missing_features']
        assert result['missing_percent'] > 0
    
    def test_range_validation(self, assessor, sample_clinical_data):
        """Test range validation"""
        result = assessor._validate_ranges(sample_clinical_data)
        
        assert 'range_violations' in result
        assert 'violation_count' in result
        # Sample data should have minimal violations
        assert result['violation_count'] == 0
    
    def test_range_validation_with_violations(self, assessor):
        """Test range validation with violations"""
        data = {
            'Age': 150.0,  # Above max
            'Height_cm': 50.0,  # Below min
        }
        
        result = assessor._validate_ranges(data)
        
        assert result['violation_count'] > 0
        assert len(result['range_violations']) > 0
    
    def test_outlier_detection_iqr(self, assessor, sample_clinical_data):
        """Test outlier detection using IQR method"""
        result = assessor._detect_outliers(sample_clinical_data)
        
        assert 'outliers' in result
        assert 'outlier_count' in result
        assert result['method_used'] == 'iqr'
    
    def test_consistency_check_bmi(self, assessor):
        """Test BMI consistency check"""
        # Create data with inconsistent BMI
        data = {
            'Height_cm': 170.0,
            'Weight_kg': 70.0,
            'BMI': 50.0  # Wrong BMI
        }
        
        violation = assessor._check_bmi_consistency(data)
        
        assert violation is not None
        assert violation['check_type'] == 'BMI_calculation'
        assert violation['difference_percent'] > 10
    
    def test_consistency_check_bmi_consistent(self, assessor):
        """Test BMI consistency check with consistent data"""
        # Create data with consistent BMI
        data = {
            'Height_cm': 170.0,
            'Weight_kg': 70.0,
            'BMI': 24.2  # Correct BMI (70 / (1.7^2))
        }
        
        violation = assessor._check_bmi_consistency(data)
        
        assert violation is None
    
    def test_consistency_check_lh_fsh(self, assessor):
        """Test LH/FSH ratio consistency check"""
        data = {
            'LH_mIU_mL': 10.0,
            'FSH_mIU_mL': 5.0,
            'LH_FSH_Ratio': 5.0  # Wrong ratio
        }
        
        violation = assessor._check_lh_fsh_consistency(data)
        
        assert violation is not None
        assert violation['check_type'] == 'LH_FSH_calculation'
    
    def test_consistency_check_waist_hip(self, assessor):
        """Test waist-hip ratio consistency check"""
        data = {
            'Waist_Circumference_cm': 80.0,
            'Hip_Circumference_cm': 100.0,
            'Waist_Hip_Ratio': 1.5  # Wrong ratio
        }
        
        violation = assessor._check_waist_hip_consistency(data)
        
        assert violation is not None
        assert violation['check_type'] == 'Waist_Hip_calculation'
    
    def test_consistency_check_homa_ir(self, assessor):
        """Test HOMA-IR consistency check"""
        data = {
            'Fasting_Glucose_mg_dL': 100.0,
            'Fasting_Insulin_uIU_mL': 10.0,
            'HOMA_IR': 10.0  # Wrong HOMA-IR
        }
        
        violation = assessor._check_homa_ir_consistency(data)
        
        assert violation is not None
        assert violation['check_type'] == 'HOMA_IR_calculation'
    
    def test_reliability_score_calculation(self, assessor, sample_clinical_data):
        """Test reliability score calculation"""
        # First, run the individual analyses
        import pandas as pd
        df = pd.DataFrame([sample_clinical_data])
        
        missing = assessor._detect_missing_values(df, sample_clinical_data)
        range_val = assessor._validate_ranges(sample_clinical_data)
        outliers = assessor._detect_outliers(sample_clinical_data)
        consistency = assessor._check_consistency(sample_clinical_data)
        
        # Calculate reliability score
        score = assessor._calculate_reliability_score(
            missing, range_val, outliers, consistency
        )
        
        assert 0.0 <= score <= 1.0
        # Sample data should have high reliability
        assert score >= 0.8
    
    def test_quality_category_determination(self, assessor):
        """Test quality category determination"""
        assert assessor._determine_quality_category(0.9).value == 'high'
        assert assessor._determine_quality_category(0.7).value == 'medium'
        assert assessor._determine_quality_category(0.5).value == 'low'
        assert assessor._determine_quality_category(0.3).value == 'unusable'
    
    def test_complete_clinical_assessment(self, assessor, sample_clinical_data):
        """Test complete clinical quality assessment workflow"""
        report = assessor.assess_clinical_data_quality(
            clinical_data=sample_clinical_data,
            patient_id="patient_001"
        )
        
        assert report['patient_id'] == "patient_001"
        assert 'reliability_score' in report
        assert 'quality_category' in report
        assert 'missing_value_analysis' in report
        assert 'range_validation' in report
        assert 'outlier_detection' in report
        assert 'consistency_checks' in report
        assert report['assessment_status'] in ['completed', 'failed']
    
    def test_clinical_assessment_with_missing_critical(self, assessor):
        """Test assessment with missing critical features"""
        data = {
            'Age': None,  # Critical feature missing
            'BMI': 25.0,
            'Height_cm': 165.0,
            'Weight_kg': 65.0,
            'LH_FSH_Ratio': 2.0,
            'Total_Testosterone_ng_dL': 45.0,
        }
        
        report = assessor.assess_clinical_data_quality(
            clinical_data=data,
            patient_id="patient_002"
        )
        
        # Should have reduced reliability due to critical missing
        assert report['missing_value_analysis']['has_critical_missing'] is True
        assert report['reliability_score'] < 1.0  # Should be reduced but not necessarily < 0.5


class TestQualityConfig:
    """Test suite for QualityConfig"""
    
    def test_config_initialization(self):
        """Test that config initializes correctly"""
        config = QualityConfig()
        assert config is not None
    
    def test_get_image_quality_thresholds(self):
        """Test getting image quality thresholds"""
        thresholds = QualityConfig.get_image_quality_thresholds()
        
        assert 'brisque' in thresholds
        assert 'niqe' in thresholds
        assert 'good' in thresholds['brisque']
        assert 'poor' in thresholds['brisque']
        assert 'unusable' in thresholds['brisque']
    
    def test_get_feature_range(self):
        """Test getting feature range"""
        range_info = QualityConfig.get_feature_range('Age')
        
        assert 'min' in range_info
        assert 'max' in range_info
        assert range_info['min'] == 10.0
        assert range_info['max'] == 60.0
    
    def test_get_feature_range_nonexistent(self):
        """Test getting range for non-existent feature"""
        range_info = QualityConfig.get_feature_range('NonExistentFeature')
        
        assert range_info == {}
    
    def test_update_threshold(self):
        """Test updating threshold"""
        original_value = QualityConfig.BRISQUE_GOOD_THRESHOLD
        
        QualityConfig.update_threshold('brisque', 'good', 30.0)
        assert QualityConfig.BRISQUE_GOOD_THRESHOLD == 30.0
        
        # Restore original value
        QualityConfig.update_threshold('brisque', 'good', original_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
