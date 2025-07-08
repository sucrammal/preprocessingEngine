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
    VIAM_MODEL_ORG_ID=your-model-org-id
    VIAM_MODEL_VERSION=2024-XX-XXTXX-XX-XX

    # Data Paths
    METADATA_DIR=metadata
    IMAGES_DIR=data
    MODEL_DIR=models

    # Processing Configuration
    PREPROCESSING_METHOD=simple    # Options: simple, gcn, lcn
    INFERENCE_CONFIDENCE=0.5       # Detection confidence threshold
    IOU_CONFIDENCE=0.5            # IoU matching threshold
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
