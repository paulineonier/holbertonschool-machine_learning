# CIFAR-10 Transfer Learning

This project trains a CIFAR-10 classifier using transfer learning from a
Keras Application (EfficientNetB0). The script computes frozen bottleneck
features once to speed up training, trains a small classifier on top,
optionally fine-tunes some top layers, and saves the compiled model as
`cifar10.h5`.

## Files
- `0-transfer.py` - The main script. Contains `preprocess_data(X, Y)` and the
  training pipeline. When run directly it trains and saves `cifar10.h5`.
- `cifar10.h5` - Produced by running `0-transfer.py` (not included).

## Usage
Make the script executable and run it:

```bash
chmod +x 0-transfer.py
./0-transfer.py
