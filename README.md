# GNSS Error Prediction

A deep learning project for predicting Global Navigation Satellite System (GNSS) positioning errors using Recurrent Neural Networks (RNNs).

The project generates synthetic GNSS error data, trains a sequential neural network model, and evaluates its ability to predict GNSS positioning errors from time-series measurements.

## Overview

GNSS positioning measurements can be affected by several sources of error, including atmospheric effects, satellite geometry, receiver noise, and multipath propagation.

Since GNSS measurements form sequential time-series data, Recurrent Neural Networks can be used to learn temporal relationships between measurements and estimate future positioning errors.

This project provides a complete pipeline for:

- Generating synthetic GNSS error data
- Preparing sequential datasets for deep learning
- Training an RNN-based prediction model
- Evaluating model performance
- Managing experiment parameters through YAML configuration files

## Project Structure

```text
GNSS-Error-Prediction/
│
├── configs/
│   └── default.yaml
│
├── data/
│   └── synthetic_gnss_errors.csv
│
├── scripts/
│   ├── generate_data.py
│   ├── train.py
│   └── evaluate.py
│
├── src/
│   └── isro_gnss/
│       ├── models/
│       │   ├── __init__.py
│       │   └── rnn.py
│       │
│       ├── __init__.py
│       ├── config.py
│       ├── data_gen.py
│       ├── dataset.py
│       ├── evaluate.py
│       ├── train.py
│       └── utils.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Workflow

The overall pipeline of the project is:

```text
Synthetic GNSS Data
        │
        ▼
Data Preprocessing
        │
        ▼
Sequence Generation
        │
        ▼
RNN Model
        │
        ▼
Model Training
        │
        ▼
Error Prediction
        │
        ▼
Model Evaluation
```

## Model

The project uses a Recurrent Neural Network (RNN) for learning temporal patterns in GNSS measurements.

Given a sequence of previous GNSS observations, the model learns a mapping of the form:

```text
GNSS measurements over time
            ↓
           RNN
            ↓
Predicted positioning error
```

RNNs are suitable for this task because GNSS error measurements can exhibit temporal dependencies that cannot be captured effectively by treating every observation independently.

## Dataset

The repository includes a synthetic GNSS error dataset:

```text
data/synthetic_gnss_errors.csv
```

Synthetic data allows the complete training and evaluation pipeline to be tested without requiring access to specialized GNSS receiver hardware or proprietary datasets.

Additional datasets can be integrated by modifying the data loading and preprocessing pipeline.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aaryankumar1435-maker/GNSS-Error-Prediction.git
cd GNSS-Error-Prediction
```

### 2. Create a virtual environment

Using Python:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Experiment parameters are stored in:

```text
configs/default.yaml
```

This keeps model and training parameters separate from the implementation and makes experiments easier to reproduce.

## Generate Synthetic Data

Run:

```bash
python scripts/generate_data.py
```

The generated GNSS data is stored inside the `data/` directory.

## Train the Model

Run:

```bash
python scripts/train.py
```

The training pipeline:

1. Loads the configured dataset
2. Prepares sequential samples
3. Initializes the RNN model
4. Trains the network
5. Stores generated model artifacts/results

## Evaluate the Model

Run:

```bash
python scripts/evaluate.py
```

The evaluation pipeline loads the trained model and measures its performance on GNSS error prediction.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- PyYAML
- Recurrent Neural Networks (RNN)
- Time-Series Modeling

## Future Improvements

Possible extensions include:

- LSTM and GRU architectures
- Bidirectional recurrent networks
- Transformer-based time-series models
- Real-world GNSS datasets
- GPS/GNSS receiver integration
- Hyperparameter optimization
- Comparison between RNN, LSTM and GRU models
- Real-time GNSS error prediction
- Improved visualization of prediction results

## Applications

GNSS error prediction can be useful in areas such as:

- Satellite navigation
- Autonomous vehicles
- Drone navigation
- Robotics
- Precision agriculture
- Location-based systems
- Navigation reliability analysis

## Disclaimer

This repository is an educational and experimental project exploring machine-learning-based GNSS error prediction using synthetic data. It is not an official ISRO project or an ISRO-endorsed implementation.

## Author

**Aryan Kumar**

Computer Science and Engineering  
Indian Institute of Information Technology, Pune
