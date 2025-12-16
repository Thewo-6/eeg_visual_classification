# EEG Visual Classification - Project Documentation

## Overview

This project implements deep learning models for EEG-based visual classification, building upon the original research from the papers:
- **TPAMI 2020**: "Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features"
- **CVPR 2017**: "Deep Learning Human Mind for Automated Visual Classification"

The project includes the original implementation along with additional analysis tools and comprehensive experiment comparison capabilities.

## Dataset Information

### Recording Protocol
- **Subjects**: 6 participants
- **Classes**: 40 object classes from ImageNet
- **Images per class**: 50 (total 2,000 images)
- **Presentation**: Block-based with 0.5s per image
- **Inter-block interval**: 10-second black screen

### Data Characteristics
- **Total segments**: 11,964 (36 excluded due to quality issues)
- **Channels**: 128 EEG channels
- **Sampling rate**: 1 kHz
- **Signal length**: 440 samples (after preprocessing)
- **Frequency bands**: 
  - 14-70 Hz (mid-range)
  - 5-95 Hz (broadband)
  - 55-95 Hz (gamma)

### Dataset Access
Download link: https://tinyurl.com/eeg-visual-classification

## Project Structure

```
eeg_visual_classification/
├── models/                           # Neural network architectures
│   ├── EEGChannelNet.py             # TPAMI 2020 model
│   └── lstm.py                       # CVPR 2017 LSTM model
├── eeg_signal_classification.py      # Main training script
├── layers.py                         # Custom neural network layers
├── plot_inter_subject_results.py     # Inter-subject analysis visualization
├── plot_intra_subject_results.py     # Intra-subject analysis visualization
├── requirements.txt                  # Python dependencies
├── run_experiment.sh                 # Single experiment runner
├── run_all_experiments.sh            # Batch experiment runner
├── inter_subject_results.csv         # Inter-subject experiment results
├── intra_subject_exp.csv            # Intra-subject experiment results
├── inter_subject_comparison.txt      # Statistical analysis (inter)
├── intra_subject_comparison.txt      # Statistical analysis (intra)
└── README.md                        # Original dataset documentation
```

## Models

### 1. EEGChannelNet (TPAMI 2020)
**File**: `models/EEGChannelNet.py`

Advanced architecture featuring:
- **Temporal Block**: Multi-scale temporal convolutions with dilated kernels
- **Spatial Block**: Processes spatial patterns across EEG channels
- **Residual Blocks**: Deep feature extraction with skip connections
- **Embedding Layer**: 1000-dimensional feature representation

**Key Parameters**:
- Input: (batch, 1, 128 channels, 440 samples)
- Temporal channels: 10
- Output channels: 50
- Embedding size: 1000
- Classes: 40

### 2. LSTM Model (CVPR 2017)
**File**: `models/lstm.py`

Recurrent architecture with:
- **LSTM layers**: Sequential processing of temporal data
- **Feature extraction**: 128-dimensional LSTM hidden states
- **Classification head**: Two-layer fully connected network

**Key Parameters**:
- Input size: 128
- LSTM size: 128
- LSTM layers: 1
- Output size: 128
- Classes: 40

## Custom Layers

**File**: `layers.py`

Implements specialized building blocks:
- **ConvLayer2D**: Batch normalization + ReLU + 2D convolution + dropout
- **TemporalBlock**: Multi-scale temporal feature extraction
- **SpatialBlock**: Spatial pattern recognition across channels
- **ResidualBlock**: Skip connections for gradient flow

## Training Script

**File**: `eeg_signal_classification.py`

### Command Line Arguments

#### Data Options
```bash
--eeg-dataset         # Path to EEG data (.pth file)
--splits-path         # Path to train/val/test splits
--split-num           # Split number (default: 0)
--subject             # Subject ID (0=all, 1-6=individual)
--time-low            # Start sample (default: 20)
--time-high           # End sample (default: 460)
```

#### Model Options
```bash
--model-type          # Model: 'lstm' or 'EEGChannelNet'
--model-params        # Model-specific parameters
--pretrained-net      # Path to pretrained weights
```

#### Training Options
```bash
--batch-size          # Batch size (default: 16)
--optim               # Optimizer (default: Adam)
--learning-rate       # Initial LR (default: 0.001)
--learning-rate-decay-by    # LR decay factor (default: 0.5)
--learning-rate-decay-every # LR decay period (default: 10)
--epochs              # Training epochs (default: 100)
--weight-decay        # L2 regularization (default: 0.0001)
--early-stopping      # Patience in epochs (default: 20)
--data-workers        # Data loading threads (default: 4)
```

#### Output Options
```bash
--csv-file           # Results CSV file (default: results.csv)
--saveCheck          # Checkpoint save interval
```

### Example Usage

#### Inter-subject (All subjects)
```bash
python eeg_signal_classification.py \
    --eeg-dataset data/block/eeg_5_95_std.pth \
    --splits-path data/block/block_splits_by_image_all.pth \
    --subject 0 \
    --model-type EEGChannelNet \
    --batch-size 16 \
    --epochs 100
```

#### Intra-subject (Single subject)
```bash
python eeg_signal_classification.py \
    --eeg-dataset data/block/eeg_5_95_std.pth \
    --splits-path data/block/block_splits_by_image_single.pth \
    --subject 1 \
    --model-type lstm \
    --batch-size 16 \
    --epochs 100
```

## Analysis and Visualization Tools

### 1. Inter-Subject Analysis
**File**: `plot_inter_subject_results.py`

Generates comprehensive visualizations comparing model performance across all subjects:

**Visualizations**:
1. Best test accuracy by model and frequency band
2. Validation vs test accuracy comparison
3. Training progression across epochs
4. Training time comparison
5. Final accuracy comparison (train/val/test)
6. Detailed performance heatmap

**Output**: `inter_subject_comparison.png`

**Usage**:
```bash
python plot_inter_subject_results.py
```

**Input**: `inter_subject_results.csv`

### 2. Intra-Subject Analysis
**File**: `plot_intra_subject_results.py`

Provides detailed per-subject analysis across all experimental conditions:

**Visualizations** (9 subplots):
1. Average test accuracy by subject and model
2. Average accuracy by frequency band
3. Overall model performance comparison
4. Best/average/worst performance per subject
5. Subject performance heatmap (test accuracy)
6. Best configuration per subject
7. Frequency band performance distribution
8. Training stability (std deviation analysis)
9. Subject-wise model preference

**Outputs**:
- `intra_subject_results_analysis.png` - Main analysis
- `intra_subject_by_subject_analysis.png` - Per-subject breakdown
- `intra_subject_summary.png` - Summary statistics

**Usage**:
```bash
python plot_intra_subject_results.py
```

**Input**: `intra_subject_exp.csv`

## Experiment Results Structure

### Inter-Subject Results CSV
Columns:
- `model`: Model architecture (lstm/EEGChannelNet)
- `freq_band`: Frequency band (14-70/5-95/55-95)
- `subject`: Always 0 (all subjects)
- `best_test_accuracy`: Peak test accuracy
- `best_val_accuracy`: Peak validation accuracy
- `final_train_accuracy`: Final training accuracy
- `final_val_accuracy`: Final validation accuracy
- `final_test_accuracy`: Final test accuracy
- `training_time`: Total training time (seconds)
- `epochs_trained`: Number of epochs completed

### Intra-Subject Results CSV
Columns: (same as inter-subject, but per individual subject)
- `subject`: 1-6 (individual subjects)
- Additional per-subject metrics

## Installation

### Requirements
```bash
# Create virtual environment (optional but recommended)
python -m venv EEG
source EEG/bin/activate  # On Windows: EEG\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- scipy >= 1.5.0
- numpy >= 1.19.0
- pandas >= 1.1.0
- matplotlib (for visualization)
- seaborn (for visualization)

## Running Experiments

### Single Experiment
```bash
bash run_experiment.sh
```

### Complete Experimental Suite
```bash
bash run_all_experiments.sh
```

This will run all combinations of:
- Models: LSTM and EEGChannelNet
- Frequency bands: 14-70 Hz, 5-95 Hz, 55-95 Hz
- Subjects: All subjects + individual subjects (1-6)

## Key Features and Modifications

### 1. Comprehensive Analysis Framework
- **Statistical comparison**: Automated performance comparison across models and conditions
- **Visualization suite**: Publication-ready plots for all experimental scenarios
- **CSV logging**: Structured results storage for further analysis

### 2. Multi-Condition Testing
- **Inter-subject**: Generalization across all subjects
- **Intra-subject**: Subject-specific performance analysis
- **Frequency band analysis**: Comparison across different EEG frequency ranges

### 3. Enhanced Logging
- Training progress tracking
- Best model checkpointing
- Early stopping based on validation performance
- Detailed timing information

### 4. Reproducibility
- Structured experiment scripts
- Consistent data splits
- Random seed control (implicit in PyTorch)

## Expected Performance

### Inter-Subject (Approximate)
- **EEGChannelNet**: ~10-15% test accuracy
- **LSTM**: ~5-10% test accuracy
- **Baseline** (random): 2.5% (40 classes)

### Intra-Subject (Approximate)
- **EEGChannelNet**: ~15-25% test accuracy
- **LSTM**: ~10-20% test accuracy
- Higher variance across subjects

*Note: Actual performance depends on frequency band, subject, and training hyperparameters*

## Citations

If you use this code or dataset, please cite:

```bibtex
@article{palazzo2020decoding,
  title={Decoding Brain Representations by Multimodal Learning of Neural Activity and Visual Features},
  author={Palazzo, Simone and Spampinato, Concetto and Kavasidis, Isaak and Giordano, Daniela and Schmidt, Joseph and Shah, Mubarak},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  doi={10.1109/TPAMI.2020.2995909}
}

@inproceedings{spampinato2017deep,
  title={Deep Learning Human Mind for Automated Visual Classification},
  author={Spampinato, Concetto and Palazzo, Simone and Kavasidis, Isaak and Giordano, Daniela and Souly, Nasim and Shah, Mubarak},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size`
- Reduce `--time-high` value
- Use single subject instead of all subjects

### Poor Performance
- Check data preprocessing (time window selection)
- Verify correct frequency band for task
- Try different learning rates
- Increase training epochs
- Check for data loading issues

### Visualization Errors
- Ensure CSV files exist before running plot scripts
- Check that all required packages are installed
- Verify data format in CSV files

## Project Extensions

The current implementation can be extended with:
- **Cross-validation**: K-fold validation across subjects
- **Data augmentation**: Temporal jittering, channel dropout
- **Transfer learning**: Pre-training on larger EEG datasets
- **Attention mechanisms**: Interpretable feature selection
- **Ensemble methods**: Combining multiple models
- **Real-time inference**: Optimized deployment pipeline

## License

Please refer to the original papers for licensing information regarding the dataset and original models.

## Contact

For questions about the original work, please refer to the papers' authors.
For questions about this implementation and modifications, please open an issue in the repository.

---

**Last Updated**: December 2025
**Version**: 2.0 (with analysis and visualization enhancements)
