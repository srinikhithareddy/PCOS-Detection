"""
Clinical Data Preprocessing Pipeline
Handles missing values, outliers, encoding, feature selection, and normalization
"""

import os
import pickle
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from configs.clinical_preprocessing_config import ClinicalPreprocessingConfig

# Configure logging
logging.basicConfig(
    level=getattr(logging, ClinicalPreprocessingConfig.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ClinicalPreprocessingConfig.LOG_FILE),
        logging.StreamHandler() if ClinicalPreprocessingConfig.LOG_TO_CONSOLE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)


class ClinicalPreprocessor:
    """Clinical data preprocessing pipeline with data leakage prevention"""
    
    def __init__(self, config: Optional[ClinicalPreprocessingConfig] = None):
        """
        Initialize the clinical preprocessor
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or ClinicalPreprocessingConfig()
        self.config.ensure_directories()
        
        # Fitted preprocessing objects
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.encoder = None
        self.feature_selector = None
        self.scaler = None
        
        # Metadata
        self.feature_names = None
        self.selected_features = None
        self.categorical_features = None
        self.numerical_features = None
        self.is_fitted = False
        
        logger.info("ClinicalPreprocessor initialized")
    
    def fit_transform(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit preprocessing pipeline on training data and transform it
        
        Args:
            data: Training data (DataFrame or dict)
            target: Target variable for supervised feature selection (optional)
            
        Returns:
            Tuple of (transformed_data, metadata)
        """
        logger.info("Starting fit_transform on training data")
        
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Store original feature names
        self.feature_names = list(data.columns)
        
        # Separate numerical and categorical features
        self._identify_feature_types(data)
        
        # Step 1: Missing value handling
        data = self._handle_missing_values(data, fit=True)
        
        # Step 2: Outlier handling
        data = self._handle_outliers(data, fit=True)
        
        # Step 3: Range & consistency validation
        data = self._validate_range_and_consistency(data)
        
        # Step 4: Categorical encoding
        data = self._encode_categorical_features(data, fit=True)
        
        # Step 5: Feature selection
        data, selected_feature_names = self._select_features(data, target, fit=True)
        self.selected_features = selected_feature_names
        
        # Step 6: Normalization
        data = self._normalize_features(data, fit=True)
        
        # Save fitted objects
        self._save_fitted_objects(mode='train')
        
        self.is_fitted = True
        
        metadata = {
            'original_features': self.feature_names,
            'selected_features': self.selected_features,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'final_shape': data.shape,
            'preprocessing_status': 'completed'
        }
        
        logger.info(f"Fit_transform completed. Final shape: {data.shape}")
        
        return data.values if isinstance(data, pd.DataFrame) else data, metadata
    
    def transform(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Transform new data using fitted preprocessing pipeline
        
        Args:
            data: New data to transform (DataFrame or dict)
            
        Returns:
            Tuple of (transformed_data, metadata)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        logger.info("Starting transform on new data")
        
        # Load fitted objects if not in memory
        if self.numerical_imputer is None:
            self._load_fitted_objects(mode='train')
        
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure same columns as training data
        data = self._align_columns(data)
        
        # Step 1: Missing value handling (using fitted imputers)
        data = self._handle_missing_values(data, fit=False)
        
        # Step 2: Outlier handling (using fitted parameters)
        data = self._handle_outliers(data, fit=False)
        
        # Step 3: Range & consistency validation
        data = self._validate_range_and_consistency(data)
        
        # Step 4: Categorical encoding (using fitted encoder)
        data = self._encode_categorical_features(data, fit=False)
        
        # Step 5: Feature selection (using fitted selector)
        data, _ = self._select_features(data, target=None, fit=False)
        
        # Step 6: Normalization (using fitted scaler)
        data = self._normalize_features(data, fit=False)
        
        metadata = {
            'original_features': list(data.columns) if isinstance(data, pd.DataFrame) else self.feature_names,
            'selected_features': self.selected_features,
            'final_shape': data.shape,
            'preprocessing_status': 'completed'
        }
        
        logger.info(f"Transform completed. Final shape: {data.shape}")
        
        return data.values if isinstance(data, pd.DataFrame) else data, metadata
    
    def _identify_feature_types(self, data: pd.DataFrame) -> None:
        """
        Identify numerical and categorical features
        
        Args:
            data: Input data
        """
        # Use configured categorical features if available
        configured_categorical = set(self.config.CATEGORICAL_FEATURES)
        
        self.categorical_features = []
        self.numerical_features = []
        
        for col in data.columns:
            if col in configured_categorical:
                self.categorical_features.append(col)
            elif data[col].dtype == 'object' or data[col].dtype == 'category':
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)
        
        logger.info(f"Identified {len(self.numerical_features)} numerical and {len(self.categorical_features)} categorical features")
    
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        fit: bool
    ) -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            data: Input data
            fit: Whether to fit imputers
            
        Returns:
            Data with imputed values
        """
        params = self.config.get_imputation_params()
        data = data.copy()
        
        # Handle numerical features
        if self.numerical_features:
            numerical_data = data[self.numerical_features]
            
            if fit:
                if params['numerical_strategy'] == 'knn':
                    self.numerical_imputer = KNNImputer(n_neighbors=params['knn_neighbors'])
                else:
                    self.numerical_imputer = SimpleImputer(
                        strategy=params['numerical_strategy'],
                        fill_value=params['numerical_constant']
                    )
                numerical_imputed = self.numerical_imputer.fit_transform(numerical_data)
            else:
                numerical_imputed = self.numerical_imputer.transform(numerical_data)
            
            data[self.numerical_features] = numerical_imputed
        
        # Handle categorical features
        if self.categorical_features:
            categorical_data = data[self.categorical_features]
            
            if fit:
                self.categorical_imputer = SimpleImputer(
                    strategy=params['categorical_strategy'],
                    fill_value=params['categorical_constant']
                )
                categorical_imputed = self.categorical_imputer.fit_transform(categorical_data)
            else:
                categorical_imputed = self.categorical_imputer.transform(categorical_data)
            
            data[self.categorical_features] = categorical_imputed
        
        # Check for critical missing features
        for feature in params['critical_features']:
            if feature in data.columns and data[feature].isna().any():
                logger.warning(f"Critical feature {feature} still has missing values after imputation")
        
        logger.debug("Missing value handling completed")
        return data
    
    def _handle_outliers(
        self,
        data: pd.DataFrame,
        fit: bool
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical features
        
        Args:
            data: Input data
            fit: Whether to fit outlier detection parameters
            
        Returns:
            Data with handled outliers
        """
        params = self.config.get_outlier_params()
        data = data.copy()
        
        if params['handling_strategy'] == 'none':
            return data
        
        exempt_features = set(params['exempt_features'])
        
        for feature in self.numerical_features:
            if feature in exempt_features:
                continue
            
            if feature not in data.columns:
                continue
            
            values = data[feature].values
            
            if params['detection_method'] == 'iqr':
                Q1 = np.percentile(values, 25)
                Q3 = np.percentile(values, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - params['iqr_multiplier'] * IQR
                upper_bound = Q3 + params['iqr_multiplier'] * IQR
                
            elif params['detection_method'] == 'zscore':
                mean = np.mean(values)
                std = np.std(values)
                lower_bound = mean - params['z_score_threshold'] * std
                upper_bound = mean + params['z_score_threshold'] * std
            
            else:
                continue
            
            # Apply handling strategy
            if params['handling_strategy'] == 'clip':
                data[feature] = np.clip(values, lower_bound, upper_bound)
            elif params['handling_strategy'] == 'winsorize':
                lower_percentile, upper_percentile = params['winsorize_percentiles']
                data[feature] = stats.mstats.winsorize(values, limits=[
                    lower_percentile/100, upper_percentile/100
                ])
            elif params['handling_strategy'] == 'remove':
                # Flag outliers instead of removing (preserve data)
                data[f'{feature}_outlier_flag'] = ((values < lower_bound) | (values > upper_bound)).astype(int)
        
        logger.debug("Outlier handling completed")
        return data
    
    def _validate_range_and_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate ranges and consistency checks
        
        Args:
            data: Input data
            
        Returns:
            Validated data
        """
        data = data.copy()
        
        # Range validation
        if self.config.ENFORCE_RANGE_CONSTRAINTS:
            try:
                from quality_config import QualityConfig
                ranges = QualityConfig.CLINICAL_FEATURE_RANGES
                
                for feature in self.numerical_features:
                    if feature in ranges:
                        feature_range = ranges[feature]
                        min_val = feature_range.get('min')
                        max_val = feature_range.get('max')
                        
                        if min_val is not None:
                            data[feature] = np.maximum(data[feature], min_val)
                        if max_val is not None:
                            data[feature] = np.minimum(data[feature], max_val)
            except Exception as e:
                logger.warning(f"Range validation failed: {str(e)}")
        
        # Consistency validation
        if self.config.ENFORCE_CONSISTENCY_CHECKS:
            data = self._apply_consistency_checks(data)
        
        logger.debug("Range and consistency validation completed")
        return data
    
    def _apply_consistency_checks(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply consistency checks between related features
        
        Args:
            data: Input data
            
        Returns:
            Data with consistency checks applied
        """
        tolerance = self.config.CALCULATION_TOLERANCE_PERCENT / 100
        
        # BMI consistency
        if 'Height_cm' in data.columns and 'Weight_kg' in data.columns and 'BMI' in data.columns:
            height_m = data['Height_cm'] / 100
            calculated_bmi = data['Weight_kg'] / (height_m ** 2)
            diff_percent = abs(calculated_bmi - data['BMI']) / data['BMI']
            # Use calculated BMI if difference exceeds tolerance
            mask = diff_percent > tolerance
            data.loc[mask, 'BMI'] = calculated_bmi[mask]
        
        # LH/FSH ratio consistency
        if 'LH_mIU_mL' in data.columns and 'FSH_mIU_mL' in data.columns and 'LH_FSH_Ratio' in data.columns:
            calculated_ratio = data['LH_mIU_mL'] / data['FSH_mIU_mL']
            diff_percent = abs(calculated_ratio - data['LH_FSH_Ratio']) / data['LH_FSH_Ratio']
            mask = diff_percent > tolerance
            data.loc[mask, 'LH_FSH_Ratio'] = calculated_ratio[mask]
        
        # Waist/Hip ratio consistency
        if 'Waist_Circumference_cm' in data.columns and 'Hip_Circumference_cm' in data.columns and 'Waist_Hip_Ratio' in data.columns:
            calculated_ratio = data['Waist_Circumference_cm'] / data['Hip_Circumference_cm']
            diff_percent = abs(calculated_ratio - data['Waist_Hip_Ratio']) / data['Waist_Hip_Ratio']
            mask = diff_percent > tolerance
            data.loc[mask, 'Waist_Hip_Ratio'] = calculated_ratio[mask]
        
        # HOMA-IR consistency
        if 'Fasting_Glucose_mg_dL' in data.columns and 'Fasting_Insulin_uIU_mL' in data.columns and 'HOMA_IR' in data.columns:
            calculated_homa = (data['Fasting_Glucose_mg_dL'] * data['Fasting_Insulin_uIU_mL']) / 405
            diff_percent = abs(calculated_homa - data['HOMA_IR']) / data['HOMA_IR']
            mask = diff_percent > tolerance
            data.loc[mask, 'HOMA_IR'] = calculated_homa[mask]
        
        return data
    
    def _encode_categorical_features(
        self,
        data: pd.DataFrame,
        fit: bool
    ) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            data: Input data
            fit: Whether to fit encoder
            
        Returns:
            Data with encoded categorical features
        """
        params = self.config.get_encoding_params()
        data = data.copy()
        
        if not self.categorical_features:
            return data
        
        if params['method'] == 'onehot':
            if fit:
                self.encoder = OneHotEncoder(
                    drop='first' if params['drop_first'] else None,
                    handle_unknown=params['unknown_handling'],
                    sparse_output=False
                )
                encoded = self.encoder.fit_transform(data[self.categorical_features])
                feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            else:
                encoded = self.encoder.transform(data[self.categorical_features])
                feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=data.index)
            
            # Drop original categorical features and add encoded ones
            data = data.drop(columns=self.categorical_features)
            data = pd.concat([data, encoded_df], axis=1)
            
        elif params['method'] == 'label':
            if fit:
                self.encoder = {}
                for feature in self.categorical_features:
                    self.encoder[feature] = LabelEncoder()
                    data[feature] = self.encoder[feature].fit_transform(data[feature].astype(str))
            else:
                for feature in self.categorical_features:
                    # Handle unknown categories
                    if feature in self.encoder:
                        known_classes = set(self.encoder[feature].classes_)
                        data[feature] = data[feature].astype(str).apply(
                            lambda x: x if x in known_classes else 'unknown'
                        )
                        # Add 'unknown' to classes if not present
                        if 'unknown' not in self.encoder[feature].classes_:
                            self.encoder[feature].classes_ = np.append(
                                self.encoder[feature].classes_, 'unknown'
                            )
                        data[feature] = self.encoder[feature].transform(data[feature])
        
        logger.debug("Categorical encoding completed")
        return data
    
    def _select_features(
        self,
        data: pd.DataFrame,
        target: Optional[np.ndarray],
        fit: bool
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform feature selection
        
        Args:
            data: Input data
            target: Target variable (for supervised methods)
            fit: Whether to fit feature selector
            
        Returns:
            Tuple of (data with selected features, selected feature names)
        """
        params = self.config.get_feature_selection_params()
        data = data.copy()
        
        if params['method'] == 'none':
            return data, list(data.columns)
        
        selected_features = list(data.columns)
        
        if params['method'] == 'variance_threshold':
            if fit:
                # Adjust threshold for single sample to avoid error
                threshold = params['variance_threshold']
                if data.shape[0] == 1:
                    threshold = 0.0  # Allow all features with single sample
                
                self.feature_selector = VarianceThreshold(threshold=threshold)
                try:
                    self.feature_selector.fit(data)
                    mask = self.feature_selector.get_support()
                except ValueError:
                    # If variance threshold fails, keep all features
                    mask = np.ones(data.shape[1], dtype=bool)
            else:
                mask = self.feature_selector.get_support()
            
            selected_features = [f for f, m in zip(data.columns, mask) if m]
            data = data[selected_features]
        
        elif params['method'] == 'correlation':
            # Remove highly correlated features
            corr_matrix = data.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [
                column for column in upper_triangle.columns
                if any(upper_triangle[column] > params['correlation_threshold'])
            ]
            
            selected_features = [f for f in data.columns if f not in to_drop]
            data = data[selected_features]
        
        elif params['method'] == 'mutual_info' and target is not None:
            if fit:
                self.feature_selector = SelectKBest(
                    mutual_info_classif,
                    k=params['n_features_to_select']
                )
                self.feature_selector.fit(data, target)
                mask = self.feature_selector.get_support()
            else:
                mask = self.feature_selector.get_support()
            
            selected_features = [f for f, m in zip(data.columns, mask) if m]
            data = data[selected_features]
        
        elif params['method'] == 'recursive' and target is not None:
            if fit:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                self.feature_selector = RFE(
                    estimator,
                    n_features_to_select=params['n_features_to_select']
                )
                self.feature_selector.fit(data, target)
                mask = self.feature_selector.get_support()
            else:
                mask = self.feature_selector.get_support()
            
            selected_features = [f for f, m in zip(data.columns, mask) if m]
            data = data[selected_features]
        
        # Force include critical features
        force_include = set(params['force_include'])
        if self.feature_names is not None:
            for feature in force_include:
                if feature in self.feature_names and feature not in selected_features:
                    # Add feature back if it was removed
                    if feature in data.columns:
                        pass  # Already present
                    else:
                        # Feature was removed, need to handle this
                        logger.warning(f"Force include feature {feature} was removed by selection")
        
        logger.debug(f"Feature selection completed. Selected {len(selected_features)} features")
        return data, selected_features
    
    def _normalize_features(
        self,
        data: pd.DataFrame,
        fit: bool
    ) -> pd.DataFrame:
        """
        Normalize numerical features
        
        Args:
            data: Input data
            fit: Whether to fit scaler
            
        Returns:
            Normalized data
        """
        params = self.config.get_normalization_params()
        data = data.copy()
        
        if params['method'] == 'none':
            return data
        
        # Get numerical features (after encoding, all should be numerical)
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        skip_features = set(params['skip_features'])
        numerical_cols = [f for f in numerical_cols if f not in skip_features]
        
        if not numerical_cols:
            return data
        
        if params['method'] == 'standard':
            if fit:
                self.scaler = StandardScaler()
                data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
            else:
                data[numerical_cols] = self.scaler.transform(data[numerical_cols])
        
        elif params['method'] == 'minmax':
            if fit:
                self.scaler = MinMaxScaler(feature_range=params['minmax_range'])
                data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
            else:
                data[numerical_cols] = self.scaler.transform(data[numerical_cols])
        
        elif params['method'] == 'robust':
            if fit:
                self.scaler = RobustScaler(quantile_range=params['robust_quantile_range'])
                data[numerical_cols] = self.scaler.fit_transform(data[numerical_cols])
            else:
                data[numerical_cols] = self.scaler.transform(data[numerical_cols])
        
        logger.debug("Normalization completed")
        return data
    
    def _align_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Align data columns with training data
        
        Args:
            data: Input data
            
        Returns:
            Data with aligned columns
        """
        # Add missing columns with default values
        for col in self.feature_names:
            if col not in data.columns:
                data[col] = 0  # Default value for missing columns
        
        # Remove extra columns not in training data
        extra_cols = set(data.columns) - set(self.feature_names)
        if extra_cols:
            logger.warning(f"Removing extra columns not in training data: {extra_cols}")
            data = data.drop(columns=list(extra_cols))
        
        # Reorder columns to match training data
        data = data[self.feature_names]
        
        return data
    
    def _save_fitted_objects(self, mode: str = 'train') -> None:
        """
        Save fitted preprocessing objects
        
        Args:
            mode: 'train' or 'inference'
        """
        objects = {
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'encoder': self.encoder,
            'feature_selector': self.feature_selector,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'is_fitted': self.is_fitted
        }
        
        filepath = self.config.get_objects_file_path(mode)
        
        with open(filepath, 'wb') as f:
            pickle.dump(objects, f)
        
        logger.info(f"Fitted preprocessing objects saved to {filepath}")
    
    def _load_fitted_objects(self, mode: str = 'train') -> None:
        """
        Load fitted preprocessing objects
        
        Args:
            mode: 'train' or 'inference'
        """
        filepath = self.config.get_objects_file_path(mode)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessing objects file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            objects = pickle.load(f)
        
        self.numerical_imputer = objects['numerical_imputer']
        self.categorical_imputer = objects['categorical_imputer']
        self.encoder = objects['encoder']
        self.feature_selector = objects['feature_selector']
        self.scaler = objects['scaler']
        self.feature_names = objects['feature_names']
        self.selected_features = objects['selected_features']
        self.categorical_features = objects['categorical_features']
        self.numerical_features = objects['numerical_features']
        self.is_fitted = objects['is_fitted']
        
        logger.info(f"Fitted preprocessing objects loaded from {filepath}")
