# Burner Detection Preprocessing Engine

A comprehensive pipeline for evaluating TensorFlow Lite burner detection models with advanced preprocessing techniques and robust evaluation metrics.

## 🎯 **Overview**

**Pipeline**: Raw Images + Metadata → Dataset Creation → Preprocessing → TFLite Inference → Dual Evaluation

**Key Features**:

-   3 preprocessing methods for different lighting conditions
-   Automatic uint8/float32 model handling
-   Dual evaluation: presence/absence + IoU spatial matching
-   Environment-configurable parameters
-   Progress tracking with tqdm bars
-   Comprehensive visualization tools

## 🚀 **Setup**

1. **Create a virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Create a `.env` file** with your configuration:

    ```
    # Model Configuration
    VIAM_MODEL_NAME=your-burner-detection-model
    VIAM_ORG_ID=your-model-org-id
    VIAM_MODEL_VERSION=2024-XX-XXTXX-XX-XX

    # Data Paths
    METADATA_DIR=metadata
    IMAGES_DIR=data
    MODEL_DIR=models

    # Processing Configuration
    PREPROCESSING_METHOD=simple    # Options: simple, gcn, lcn
    INFERENCE_CONFIDENCE=0.5       # Detection confidence threshold
    IOU_CONFIDENCE=0.5            # IoU matching threshold

    # Advanced LCN Configuration (only used when PREPROCESSING_METHOD=lcn)
    LCN_WINDOW_SIZE=9             # Size of local window (recommend: 5-25)
    LCN_NORMALIZATION_TYPE=divisive # Options: divisive, subtractive, adaptive
    LCN_WINDOW_SHAPE=square       # Options: square, circular, gaussian
    LCN_STATISTICAL_MEASURE=mean  # Options: mean, median, percentile
    LCN_CONTRAST_BOOST=1.0        # Contrast enhancement factor (recommend: 0.5-2.0)
    LCN_EPSILON=1e-8              # Small value to prevent division by zero
    ```

4. **Ensure you're logged in to Viam CLI** (for model download):
    ```bash
    viam login
    ```

## 📊 **Usage**

Run `preprocessing.py` as a Python script or use the Jupyter notebook format:

```bash
python preprocessing.py
```

## 🎛️ **Advanced LCN Configuration**

The Local Contrast Normalization (LCN) method now supports extensive customization through environment variables:

### **Window Size (`LCN_WINDOW_SIZE`)**

Controls the size of the local neighborhood for statistics calculation:

-   **Small values (5-9)**: More local detail enhancement, faster processing
-   **Large values (15-25)**: Smoother normalization, better for global lighting variations
-   **Default**: 9

### **Normalization Type (`LCN_NORMALIZATION_TYPE`)**

-   **`divisive`**: Classic LCN - (pixel - local_mean) / local_std
-   **`subtractive`**: Simpler version - pixel - local_mean
-   **`adaptive`**: Stronger normalization in low-variance regions
-   **Default**: divisive

### **Window Shape (`LCN_WINDOW_SHAPE`)**

-   **`square`**: Standard square window (fastest)
-   **`circular`**: Circular window (more natural neighborhood)
-   **`gaussian`**: Gaussian-weighted window (smoothest results)
-   **Default**: square

### **Statistical Measure (`LCN_STATISTICAL_MEASURE`)**

-   **`mean`**: Use local mean (standard approach)
-   **`median`**: Use local median (more robust to outliers)
-   **`percentile`**: Use 25th percentile (good for bright images)
-   **Default**: mean

### **Contrast Boost (`LCN_CONTRAST_BOOST`)**

-   **Values < 1.0**: Reduce contrast enhancement
-   **Values > 1.0**: Increase contrast enhancement
-   **Default**: 1.0

### **Example Configurations**

```bash
# For images with strong shadows/lighting variations
export LCN_WINDOW_SIZE=15
export LCN_NORMALIZATION_TYPE=adaptive
export LCN_WINDOW_SHAPE=gaussian
export LCN_CONTRAST_BOOST=1.5

# For fine detail enhancement
export LCN_WINDOW_SIZE=5
export LCN_NORMALIZATION_TYPE=divisive
export LCN_WINDOW_SHAPE=square
export LCN_CONTRAST_BOOST=1.2

# For robust processing with outliers
export LCN_WINDOW_SIZE=9
export LCN_NORMALIZATION_TYPE=divisive
export LCN_STATISTICAL_MEASURE=median
export LCN_WINDOW_SHAPE=circular
```

### **Testing LCN Parameters**

The preprocessing module includes helper functions to test different LCN configurations:

```python
from preprocessing import create_dataset_dataframe, test_lcn_parameters, compare_lcn_configurations

# Load a sample image
df = create_dataset_dataframe(max_images=1)
sample_image = df.iloc[0]['image']

# Test different parameter combinations
test_lcn_parameters(sample_image)

# Compare specific configurations
configs = [
    {"window_size": 5, "normalization_type": "divisive"},
    {"window_size": 15, "normalization_type": "adaptive", "contrast_boost": 1.5},
    {"window_size": 9, "window_shape": "gaussian", "statistical_measure": "median"}
]
compare_lcn_configurations(sample_image, configs, ["Fine Detail", "Strong Adaptive", "Robust Gaussian"])
```

**Pipeline Steps**:

1. **Configuration**: Loads environment variables and displays settings
2. **Dataset Creation**: Processes images and metadata with progress bars
3. **Model Loading**: Downloads and loads TensorFlow Lite model
4. **Preprocessing**: Applies selected normalization method
5. **Inference**: Batch processing with tqdm progress tracking
6. **Evaluation**: Dual metrics (binary + spatial) with JSON export

## 🔧 **Preprocessing Methods**

| Method     | Description                   | Use Case                      |
| ---------- | ----------------------------- | ----------------------------- |
| **Simple** | Standard 0-1 normalization    | Baseline, well-lit images     |
| **GCN**    | Global Contrast Normalization | Overall brightness variations |
| **LCN**    | Local Contrast Normalization  | Lighting/shadow variations    |

## 📈 **Evaluation Metrics**

### **Presence/Absence Evaluation**

-   Accuracy, Precision, Recall, F1-Score
-   Confusion matrix analysis
-   Binary classification approach

### **IoU Matching Evaluation**

-   Spatial accuracy with bounding box overlap
-   Precision/Recall for object detection
-   Configurable IoU threshold

## 🔑 **Key Functions**

| Function                            | Purpose                              |
| ----------------------------------- | ------------------------------------ |
| `create_dataset_dataframe()`        | Load images and metadata             |
| `preprocess_image()`                | Apply normalization (simple/GCN/LCN) |
| `run_inference_on_dataframe()`      | Batch inference with progress bars   |
| `evaluate_presence_absence()`       | Binary classification metrics        |
| `evaluate_iou_matching()`           | Spatial detection metrics            |
| `visualize_preprocessing_effects()` | Compare preprocessing methods        |

## 💾 **DataFrame Save and Load Functions**

The preprocessing engine now includes robust functions for saving and loading datasets, making it easy to:

-   Save datasets after inference to avoid re-running expensive operations
-   Load existing datasets to continue analysis or evaluation
-   Create lightweight summaries for sharing or analysis
-   Support multiple file formats (pickle, CSV, JSON)

### **Core Functions**

#### `load_dataset_dataframe(file_path: str = "burner_dataset_complete.pkl") -> pd.DataFrame`

Loads a saved dataset DataFrame from a local file.

**Parameters:**

-   `file_path`: Path to the saved DataFrame file (.pkl, .csv, or .json)

**Returns:**

-   Loaded DataFrame, or empty DataFrame if file not found

**Features:**

-   Supports multiple file formats (pickle, CSV, JSON)
-   Provides detailed dataset summary upon loading
-   Shows available columns and statistics
-   Handles missing files gracefully with helpful error messages

**Example:**

```python
from preprocessing import load_dataset_dataframe

# Load existing dataset
df = load_dataset_dataframe("burner_dataset_complete.pkl")

if not df.empty:
    print(f"Loaded {len(df)} images with {df['num_burners'].sum()} ground truth burners")
```

#### `save_dataset_dataframe(df: pd.DataFrame, file_path: str = "burner_dataset_complete.pkl", save_format: str = "pickle", include_images: bool = True) -> bool`

Saves a dataset DataFrame to a local file.

**Parameters:**

-   `df`: DataFrame to save
-   `file_path`: Path where to save the file
-   `save_format`: Format to save in ('pickle', 'csv', 'json')
-   `include_images`: Whether to include image arrays (only for pickle format)

**Returns:**

-   True if save was successful, False otherwise

**Features:**

-   Supports multiple formats with appropriate warnings
-   Automatically creates directories if needed
-   Provides file size information for pickle files
-   Handles format-specific limitations (e.g., CSV/JSON can't preserve images)

**Example:**

```python
from preprocessing import save_dataset_dataframe

# Save complete dataset with images (pickle format)
success = save_dataset_dataframe(df, "complete_dataset.pkl", "pickle", True)

# Save summary without images (CSV format)
success = save_dataset_dataframe(df, "summary.csv", "csv", False)
```

#### `save_dataset_summary(df: pd.DataFrame, summary_file: str = "dataset_summary.csv", detailed_summary: bool = True) -> bool`

Saves a detailed summary of the dataset (without image arrays).

**Parameters:**

-   `df`: DataFrame to summarize
-   `summary_file`: Path for the summary file
-   `detailed_summary`: Whether to include detailed statistics

**Returns:**

-   True if save was successful, False otherwise

**Features:**

-   Creates lightweight summary files
-   Includes key columns (image_name, num_burners, has_burners, etc.)
-   Optionally creates detailed statistics JSON file
-   Perfect for sharing or analysis without large image data

**Example:**

```python
from preprocessing import save_dataset_summary

# Save basic summary
save_dataset_summary(df, "basic_summary.csv", False)

# Save detailed summary with statistics
save_dataset_summary(df, "detailed_summary.csv", True)
```

### **Enhanced Functions**

#### `run_inference_on_dataframe()` - Enhanced with Auto-Save

The `run_inference_on_dataframe()` function now includes auto-save functionality:

**New Parameters:**

-   `auto_save`: Whether to automatically save the DataFrame after inference (default: True)
-   `save_path`: Path to save the DataFrame (default: "burner_dataset_complete.pkl")

**Example:**

```python
# Run inference with auto-save enabled
df_with_inference = run_inference_on_dataframe(
    df,
    interpreter,
    normalization_method="simple",
    confidence_threshold=0.5,
    auto_save=True,
    save_path="my_results.pkl"
)
```

#### `main()` - Enhanced with Load Options

The `main()` function now supports loading existing datasets:

**New Parameters:**

-   `load_existing`: Whether to load existing dataset instead of creating new one (default: False)
-   `existing_file`: Path to existing dataset file to load (default: "burner_dataset_complete.pkl")

**Convenience Functions:**

-   `main_load_existing()`: Run main with existing dataset loading
-   `main_create_new()`: Run main with new dataset creation

### **Usage Examples**

#### Basic Usage

```python
from preprocessing import load_dataset_dataframe, save_dataset_dataframe

# Load existing dataset
df = load_dataset_dataframe("burner_dataset_complete.pkl")

# Save in different formats
save_dataset_dataframe(df, "backup.pkl", "pickle", True)  # Complete with images
save_dataset_dataframe(df, "summary.csv", "csv", False)   # Summary only
```

#### Pipeline Usage

```python
from preprocessing import main_load_existing, main_create_new

# Run pipeline with existing data (loads if available, creates if not)
main_load_existing()

# Run pipeline with new dataset creation
main_create_new()
```

#### Advanced Usage

```python
from preprocessing import load_dataset_dataframe, save_dataset_summary

# Load and analyze existing data
df = load_dataset_dataframe("my_dataset.pkl")

if not df.empty:
    # Create detailed summary with statistics
    save_dataset_summary(df, "analysis_summary.csv", True)

    # Check what's available
    print(f"Available columns: {list(df.columns)}")
    print(f"Has inference results: {'inferred_burner_bboxes' in df.columns}")
```

### **File Formats**

| Format            | Best For                                       | Preserves                                                     | File Size                   | Use Case                                            |
| ----------------- | ---------------------------------------------- | ------------------------------------------------------------- | --------------------------- | --------------------------------------------------- |
| **Pickle (.pkl)** | Complete datasets with images and complex data | All data types including numpy arrays, images, bounding boxes | Large (includes image data) | Full dataset backup, continuing analysis            |
| **CSV (.csv)**    | Lightweight summaries and analysis             | Basic data types (strings, numbers, booleans)                 | Small                       | Sharing results, quick analysis, spreadsheet import |
| **JSON (.json)**  | Web applications, API responses                | Basic data types and nested structures                        | Small to medium             | Web dashboards, data exchange                       |

### **Best Practices**

1. **Use pickle for complete datasets**: When you need to preserve all data including images
2. **Use CSV/JSON for summaries**: When sharing results or doing analysis without images
3. **Enable auto-save**: Let the pipeline automatically save after inference
4. **Check file existence**: Always verify if files exist before loading
5. **Handle empty DataFrames**: Check if loaded data is empty before proceeding

### **Error Handling**

All functions include robust error handling:

-   Missing files are handled gracefully with helpful messages
-   Format limitations are clearly communicated
-   Directory creation is automatic
-   File size information is provided for large files

### **Integration with Existing Pipeline**

The new functions integrate seamlessly with the existing pipeline:

-   Auto-save is enabled by default in `run_inference_on_dataframe()`
-   The main pipeline can load existing data to avoid re-processing
-   Summary files are created automatically
-   All existing functionality remains unchanged

## 📁 **Project Structure**

```
preprocessingEngine/
├── preprocessing.py          # Main processing pipeline
├── requirements.txt         # Python dependencies
├── .env                    # Configuration (create this)
├── README.md               # This file
├── metadata/               # JSON metadata files
├── data/                   # Image files
├── models/                 # Downloaded TFLite models
└── venv/                   # Virtual environment
```

## 📊 **Output Files**

-   `burner_dataset_complete.pkl` - Complete dataset with results
-   `dataset_summary.csv` - Summary statistics
-   `evaluation_results.json` - Comprehensive metrics
-   Visualization plots for preprocessing and results

## 🛠️ **Configuration Options**

All settings are controlled via environment variables:

```bash
# Change preprocessing method
export PREPROCESSING_METHOD=gcn

# Adjust confidence thresholds
export INFERENCE_CONFIDENCE=0.3
export IOU_CONFIDENCE=0.7

# Run with new settings
python preprocessing.py
```

## 💡 **Features**

-   **Smart Model Handling**: Auto-detects uint8 vs float32 models
-   **Progress Tracking**: Real-time progress bars for long operations
-   **Comprehensive Evaluation**: Multiple metrics for thorough assessment
-   **Flexible Configuration**: Environment-based parameter control
-   **Rich Visualization**: Preprocessing effects and results comparison
-   **JSON Export**: Serializable results for further analysis

## 🔍 **Troubleshooting**

-   **Model Download Issues**: Ensure `viam login` is completed
-   **Missing Images**: Check `IMAGES_DIR` path and file naming
-   **Memory Issues**: Process smaller batches using `max_images` parameter
-   **Preprocessing Errors**: Verify image formats and dimensions

## 📚 **Dependencies**

-   TensorFlow Lite
-   NumPy, Pandas
-   PIL (Image processing)
-   Matplotlib (Visualization)
-   tqdm (Progress bars)
-   SciPy (Advanced preprocessing)
