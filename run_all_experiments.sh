#!/bin/bash

# EEG Visual Classification - Run All Experiments
# This script runs experiments for all subjects with both models and all frequency bands

# Activate virtual environment
source EEG/Scripts/activate

# Configuration
EPOCHS=100
BATCH_SIZE=16
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001
EARLY_STOPPING=20
TIME_LOW=20
TIME_HIGH=460
SPLITS_PATH="/D/DATASETS/EEG_Based_visual_classification/data/block_splits_by_image_single.pth"
CSV_FILE="experiment_results.csv"

# Remove old CSV file if it exists
if [ -f "$CSV_FILE" ]; then
    echo "Removing old results file..."
    rm "$CSV_FILE"
fi

echo "==========================================="
echo "Starting experiments for all subjects"
echo "Models: LSTM and EEGChannelNet"
echo "Subjects: 1-6"
echo "Frequency bands: 5-95Hz, 14-70Hz, 55-95Hz"
echo "Total experiments: 36 (2 models × 6 subjects × 3 bands)"
echo "==========================================="

# Array of model types
models=("lstm" "EEGChannelNet")

# Array of datasets with different frequency bands
declare -A datasets
datasets["5-95Hz"]="/D/DATASETS/EEG_Based_visual_classification/data/eeg_5_95_std.pth"
datasets["14-70Hz"]="/D/DATASETS/EEG_Based_visual_classification/data/eeg_14_70_std.pth"
datasets["55-95Hz"]="/D/DATASETS/EEG_Based_visual_classification/data/eeg_55_95_std.pth"

# Loop through each frequency band
for freq_band in "5-95Hz" "14-70Hz" "55-95Hz"
do
    EEG_DATASET="${datasets[$freq_band]}"
    
    # Loop through each subject (1 to 6)
    for subject in {1..6}
    do
        # Loop through each model type
        for model in "${models[@]}"
        do
            echo ""
            echo "==========================================="
            echo "Running: Model=$model, Subject=$subject, Band=$freq_band"
            echo "==========================================="
            
            python eeg_signal_classification.py \
                --eeg-dataset "$EEG_DATASET" \
                --splits-path "$SPLITS_PATH" \
                --model_type "$model" \
                --subject "$subject" \
                --time_low "$TIME_LOW" \
                --time_high "$TIME_HIGH" \
                --epochs "$EPOCHS" \
                --batch_size "$BATCH_SIZE" \
                --learning-rate "$LEARNING_RATE" \
                --weight-decay "$WEIGHT_DECAY" \
                --early-stopping "$EARLY_STOPPING" \
                --learning-rate-decay-by 0.5 \
                --learning-rate-decay-every 10 \
                --optim Adam \
                --saveCheck 100 \
                --csv-file "$CSV_FILE"
            
            echo "Completed: Model=$model, Subject=$subject, Band=$freq_band"
            echo "==========================================="
        done
    done
done

echo ""
echo "==========================================="
echo "All experiments completed!"
echo "Results saved to: $CSV_FILE"
echo "==========================================="
echo ""
echo "Summary of results:"
python -c "
import pandas as pd
df = pd.read_csv('$CSV_FILE')
print(df.to_string())
print('\n')
print('Average accuracy by model:')
print(df.groupby('model')['best_test_accuracy'].agg(['mean', 'std']))
print('\n')
print('Average accuracy by subject:')
print(df.groupby('subject')['best_test_accuracy'].agg(['mean', 'std']))
print('\n')
print('Average accuracy by frequency band (from time_low/time_high):')
df['freq_band'] = df.apply(lambda x: '5-95Hz' if x['time_low'] == 20 else ('14-70Hz' if x['time_high'] == 460 else '55-95Hz'), axis=1)
print(df.groupby('freq_band')['best_test_accuracy'].agg(['mean', 'std']))
"
