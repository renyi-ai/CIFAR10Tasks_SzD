# Environment setup

## Install environment
Follow _install pytorch_ instructions:
https://pytorch.org/

## Test environment
With proper installation the training should run without any error:
```bash
python train.py
```

# Tasks

Solve every task on a different branch.

## 1. Multi resolution input
The neural network in its current state only supports input images with a fix size (32x32).

Change the network architecture to handle multiple resolutions during training.

Use `multi_resolution` branch, where the dataset is already changed to produce images 
with different resolutions in every iteration.

(The branch in its current state raises error during training, it is expected)

## 2. Extra augmentation
Add two extra image augmentation steps:

- Flip the image with probability p=0.5
- Gray scale and blur the image slightly (apply both) with probability p=0.5

## 3. Model saving
Save the best model parameters based on test accuracy.

## 4. Improve accuracy
Improve test accuracy compared to original.

You don't need to use the solutions of the previous tasks, 
and allowed to modify anything (model architecture, augmentation, etc.).
