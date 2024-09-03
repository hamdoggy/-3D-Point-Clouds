
# Point Cloud Processing and Segmentation

## Overview

This project aims to process and segment point cloud data using a combination of Python scripts, deep learning models, and visualization tools. The primary objective is to classify ground and non-ground points from LiDAR sequences.

## Project Structure

```
.
├── .pytest_cache/
├── predict_frames/
├── runs/
├── sequences/
│   ├── 00/
│   ├── 01/
│   ├── 02/
│   ├── 03/
│   ├── 04/
│   ├── 05/
├── 00.avi
├── 000003_segmented.html
├── 000004_segmented.html
├── cross_points.txt
├── output_model.pth
├── question_1.py
├── question_2.py
├── question_3.py
├── read_data.py
├── test.py
├── test_1.py
├── test_2.py
├── test_3.py
├── test_4.py
├── test_csv1.py
├── test_csv2.py
├── test_csv3.py
├── test_csv4.py
├── trained_model.pth
```

## Requirements

- Python 3.7+
- Open3D
- NumPy
- SciPy
- Matplotlib
- PyTorch
- Scikit-learn
- Plotly
- Pandas

Install the required packages using pip:

```bash
pip install open3d numpy scipy matplotlib torch scikit-learn plotly pandas
```

## Description of Files

### Data and Model Files

- `sequences/`: Directory containing point cloud data and labels.
- `output_model.pth` and `trained_model.pth`: Pre-trained model weights.
- `cross_points.txt`: File containing cross points for classification thresholds.

### Python Scripts

- `question_1.py`: Script for processing sequences, segmenting ground and non-ground points, visualizing them, and calculating statistics.
- `question_2.py`: Script for training a point cloud classification model.
- `question_3.py`: Script for feature extraction and label propagation.
- `question_4.py`: Script for pseudo-labeling and enhanced model training.
- `read_data.py`: Script for reading and processing data.
- `test.py` to `test_csv4.py`: Various test scripts for different functionalities.

### Visualization and Results

- `000003_segmented.html` and `000004_segmented.html`: HTML files containing segmented point cloud visualizations.
- `预测结果_1.csv` and `预测结果_2.csv`: CSV files with prediction results.

### Other Files

- `全数据ks检验.py`: Script for conducting Kolmogorov-Smirnov tests on data.

## Dataset Download

You can download the dataset from the following link:
[Dataset Download](https://pan.baidu.com/s/1PRWPRIa2to9KG54fOyElHA?pwd=3o9j)

### Note:

The file format is as shown in the left image:
- The `velodyne` folder in the sequence folder contains the point cloud data `XXXXXX.bin`.
- The `labels` folder provides a corresponding `XXXXXX.label` file, which contains binary format labels for each point. The labels are 32-bit unsigned integers (`uint32_t`), with the lower 16 bits corresponding to the label and the upper 16 bits encoding the instance ID.
- `XXXXXX.tag` is used to generate the distance image, recording the position of each point in the distance image.
- Simple data reading methods are provided in `read_data.py` for use.

## Usage

### Processing and Visualizing Point Clouds

To process and visualize the point clouds, use `question_1.py`:

```bash
python question_1.py
```

This script will read point cloud data, segment ground and non-ground points, visualize them, and calculate and display statistics.

### Training the Model

To train the point cloud classification model, use `question_2.py`:

```bash
python question_2.py
```

This script will load the data, preprocess it, train the model, and save the trained model weights to `trained_model.pth`.

### Feature Extraction and Label Propagation

To extract features and propagate labels, use `question_3.py`:

```bash
python question_3.py
```

This script will perform feature extraction using a pre-trained model, mask some labels to simulate unlabelled data, and propagate labels using cosine similarity.


