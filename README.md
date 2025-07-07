# Preprocessing Engine

This tool processes images through Viam cloud inference to detect burners and compares results with ground truth metadata.

## Setup

1. **Create a virtual environment** (recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Create a `.env` file** with your Viam configuration:

    ```
    VIAM_MODEL_NAME=your-burner-detection-model
    VIAM_MODEL_ORG_ID=your-model-org-id
    VIAM_MODEL_VERSION=2024-XX-XXTXX-XX-XX
    VIAM_ORG_ID=your-inference-org-id
    METADATA_DIR=metadata
    IMAGES_DIR=images
    ```

4. **Ensure you're logged in to Viam CLI**:
    ```bash
    viam login
    ```

## Usage

Open `preprocessing.py` in Cursor and run individual cells using `Cmd+Shift+Enter` (or `Ctrl+Shift+Enter` on Windows).

1. Start with the configuration cell to verify your settings
2. Run the metadata loading cell to see your data structure
3. Test single inference with `test_single_inference()`
4. Process all images with `process_all_images()`
5. Compare results with ground truth

## Project Structure

```
preprocessingEngine/
├── preprocessing.py    # Main script with Jupyter-style cells
├── requirements.txt    # Python dependencies
├── .env               # Configuration (create this file)
├── .gitignore         # Git ignore file
├── metadata/          # JSON metadata files
├── data/              # Image files (if local)
└── README.md          # This file
```

## Notes

-   Images are processed using their binary data IDs from Viam cloud
-   No need to download images locally - inference runs entirely in the cloud
-   Results can be exported to JSON for further analysis
-   The script includes ground truth comparison functionality
