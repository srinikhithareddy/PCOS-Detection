# End-to-End Preprocessing Pipeline Documentation

## Overview

The end-to-end preprocessing pipeline integrates ultrasound and clinical data preprocessing into a unified, reproducible system. The pipeline handles quality assessment, preprocessing, and validation for both modalities, producing model-ready outputs for PCOS detection.

## Pipeline Architecture

```
                    INPUT
                      ↓
             DATA QUALITY ASSESSMENT
                ↙              ↘
       ULTRASOUND              CLINICAL
           ↓                      ↓
     BRISQUE / NIQE        Missing Values
           ↓               Outliers
     Quality Decision      Range Checks
           ↓               Consistency Checks
     Resize                Reliability Score
           ↓                      ↓
   Lee / SRAD              Missing Value Handling
           ↓               Outlier Handling
        CLAHE              Encoding
           ↓               Feature Selection
    U-Net Segmentation     Normalization
           ↓                      ↓
   Ovarian ROI             Clinical Vector
           ↓                      │
    Normalization                 │
           ↓                      │
   Preprocessed ROI ──────────────┘
```

## Components

### 1. Pipeline Configuration (`pipeline_config.py`)

Configuration for the complete preprocessing pipeline:

- **Pipeline Control**: Enable/disable branches, quality thresholds
- **Output Configuration**: Directory structure, file formats
- **Logging**: Log levels, file paths, format
- **Validation**: Output validation checks and thresholds
- **Reproducibility**: Random seed, metadata saving
- **Error Handling**: Continue on modality failure, critical error types
- **Performance**: Memory limits, parallel workers, GPU usage

### 2. Preprocessing Pipeline (`pipeline/preprocessing_pipeline.py`)

Main pipeline orchestrator that coordinates all preprocessing stages:

- **Initialization**: Sets up quality assessors, preprocessors, segmentation inference
- **Pipeline Execution**: Runs complete pipeline with error handling
- **Modality Handling**: Gracefully handles missing modalities
- **Output Combination**: Combines outputs from both branches
- **Validation**: Validates final outputs
- **Metadata Saving**: Saves complete preprocessing metadata

### 3. API Endpoint (`routes/pipeline.py`)

FastAPI endpoint for pipeline execution:

- `POST /api/pipeline/preprocess` - Main pipeline endpoint
- `GET /api/pipeline/config` - Configuration retrieval
- `POST /api/pipeline/reset` - Pipeline reset

## Pipeline Stages

### Ultrasound Branch

#### Stage 1: Quality Assessment
- **Input**: Raw ultrasound image bytes
- **Process**: BRISQUE/NIQE quality assessment
- **Output**: Quality score, quality decision
- **Threshold**: Minimum quality score required to proceed

#### Stage 2: Preprocessing
- **Resize**: Standardize image size
- **Denoising**: Lee filter or SRAD for speckle noise reduction
- **Enhancement**: CLAHE for local contrast enhancement
- **Output**: Preprocessed image at each stage

#### Stage 3: Segmentation and ROI Extraction
- **Segmentation**: U-Net model for ovarian follicle segmentation
- **ROI Extraction**: Extract ovarian region from segmentation mask
- **Normalization**: Normalize extracted ROI
- **Output**: Segmentation mask, ROI, overlay, bounding box info

### Clinical Branch

#### Stage 1: Quality Assessment
- **Input**: Clinical features dictionary (47 features)
- **Process**: Missing value analysis, outlier detection, range validation, consistency checks
- **Output**: Quality metrics, reliability score, quality decision

#### Stage 2: Preprocessing
- **Missing Value Handling**: Mean/median/most-frequent/KNN imputation
- **Outlier Handling**: IQR/Z-score detection with clip/winsorize/remove
- **Range Validation**: Enforce clinical ranges
- **Consistency Checks**: Validate calculated values (BMI, LH/FSH ratio, HOMA-IR)
- **Categorical Encoding**: One-hot or label encoding
- **Feature Selection**: Configurable methods (variance threshold, correlation, mutual info, recursive)
- **Normalization**: Standard, MinMax, or Robust scaling
- **Output**: Model-ready clinical feature vector

## Data Leakage Prevention

The pipeline prevents data leakage through:

1. **Separate Modes**: `fit_transform` for training, `transform` for inference
2. **Fitted Object Persistence**: Saves all fitted transformers (imputers, encoders, selectors, scalers)
3. **Consistent Transformations**: Applies exact same transformations to new data
4. **Column Alignment**: Ensures new data matches training data structure

## Output Structure

### Final Outputs

1. **Preprocessed Ultrasound ROI**: Normalized ovarian region image
2. **Segmentation Mask**: Binary mask of ovarian follicles
3. **Ultrasound Quality Report**: Quality metrics and decision
4. **Clinical Quality Report**: Quality metrics and reliability score
5. **Clinical Reliability Score**: Overall data quality score
6. **Clinical Feature Vector**: Model-ready normalized features
7. **Preprocessing Metadata**: Complete pipeline execution logs

### Output Directory Structure

```
pipeline_outputs/
├── raw_data/              # Original input data
├── quality_reports/       # Quality assessment reports
├── ultrasound_preprocessed/  # Ultrasound preprocessing outputs
├── clinical_preprocessed/     # Clinical preprocessing outputs
├── final_outputs/        # Combined final outputs
└── logs/                 # Pipeline execution logs
```

## API Usage

### Request Format

```python
POST /api/pipeline/preprocess

Form data:
- patient_id: str (required)
- mode: str ('train' or 'inference', default: 'inference')
- ultrasound_image: file (optional)
- clinical_data: JSON string (optional)
```

### Response Format

```python
{
    "patient_id": str,
    "mode": str,
    "pipeline_start_time": str,
    "pipeline_end_time": str,
    "pipeline_duration_seconds": float,
    "ultrasound_available": bool,
    "clinical_available": bool,
    "ultrasound_pipeline": {
        "status": str,
        "stages_completed": list,
        "quality_assessment": dict,
        "preprocessing": dict,
        "segmentation": dict
    },
    "clinical_pipeline": {
        "status": str,
        "stages_completed": list,
        "quality_assessment": dict,
        "preprocessing": dict
    },
    "final_outputs": {
        "preprocessed_ultrasound_roi": path,
        "segmentation_mask": path,
        "ultrasound_quality_report": dict,
        "clinical_quality_report": dict,
        "clinical_reliability_score": float,
        "clinical_feature_vector": list,
        "output_paths": dict
    },
    "pipeline_status": str,
    "errors": list,
    "validation": dict
}
```

## Configuration

### Key Configuration Parameters

```python
# Pipeline Control
RUN_ULTRASOUND_PIPELINE = True
RUN_CLINICAL_PIPELINE = True
REQUIRE_BOTH_MODALITIES = False
MINIMUM_QUALITY_SCORE = 30.0

# Output
OUTPUT_BASE_DIR = "pipeline_outputs"
IMAGE_OUTPUT_FORMAT = 'png'
CLINICAL_OUTPUT_FORMAT = 'parquet'

# Validation
VALIDATE_OUTPUTS = True
MIN_ROI_SIZE = 100
MAX_ROI_SIZE = 500000
MIN_CLINICAL_VECTOR_LENGTH = 1
MAX_CLINICAL_VECTOR_LENGTH = 1000

# Reproducibility
RANDOM_SEED = 42
SAVE_METADATA = True

# Error Handling
CONTINUE_ON_MODALITY_FAILURE = True
STOP_ON_CRITICAL_ERRORS = True
```

## Error Handling

The pipeline handles errors at multiple levels:

1. **Input Validation**: Validates required inputs and data formats
2. **Modality-Level Errors**: Continues with available modality if one fails
3. **Stage-Level Errors**: Skips stages that fail, logs errors
4. **Critical Errors**: Stops pipeline on critical errors (memory, file not found, permissions)

## Logging

Comprehensive logging at multiple levels:

- **Pipeline Level**: Pipeline start/end, modality availability
- **Branch Level**: Branch start/end, stage completion
- **Stage Level**: Individual stage execution, errors
- **Debug Level**: Detailed operation logs

Log format: `%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s`

## Testing

### Unit Tests

- Pipeline configuration tests
- Pipeline initialization tests
- Modality handling tests
- Stage execution tests
- Output validation tests
- Metadata saving tests

### Integration Tests

- End-to-end pipeline with both modalities
- Ultrasound-only pipeline
- Clinical-only pipeline
- Error handling tests
- Train vs inference mode tests

Run tests:
```bash
pytest tests/test_pipeline.py -v
```

## Reproducibility

The pipeline ensures reproducibility through:

1. **Random Seed**: Fixed random seed for all operations
2. **Metadata Saving**: Complete execution metadata saved
3. **Fitted Object Persistence**: Exact transformations preserved
4. **Configuration Management**: All parameters configurable and documented

## Important Notes

### U-Net Segmentation

The pipeline includes U-Net segmentation infrastructure, but **requires trained weights** for actual segmentation. Without trained weights:
- Segmentation stage is skipped
- Placeholder preprocessing uses thresholding
- Pipeline continues with available outputs

### Clinical Preprocessor Fitting

The clinical preprocessor must be fitted before inference:
- Use `mode='train'` for first run to fit transformers
- Use `mode='inference'` for subsequent runs
- Fitted objects are saved and loaded automatically

### Quality Thresholds

- **Ultrasound**: Minimum quality score of 30.0 required
- **Clinical**: Minimum reliability score of 0.4 required
- Below thresholds, preprocessing is skipped for that modality

## Files

- `pipeline_config.py` - Pipeline configuration
- `pipeline/__init__.py` - Module initialization
- `pipeline/preprocessing_pipeline.py` - Main pipeline implementation
- `schemas/pipeline.py` - Pydantic schemas
- `routes/pipeline.py` - API endpoints
- `tests/test_pipeline.py` - Integration tests

## Dependencies

- FastAPI: API framework
- NumPy: Numerical operations
- Pandas: Data manipulation
- PIL/Pillow: Image processing
- OpenCV: Computer vision operations
- Scikit-learn: Machine learning preprocessing
- TensorFlow: Deep learning (U-Net)

## Future Enhancements

Potential future additions (not implemented):

- Distributed processing for large datasets
- Real-time streaming preprocessing
- Advanced feature selection methods
- Custom preprocessing pipelines per use case
- Performance monitoring and optimization
