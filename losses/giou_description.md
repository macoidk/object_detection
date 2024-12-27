# Loss Function Parameters Analysis

## 1. lambda_giou (Box Regression Weight)
**Default value:** 2.0  
**Valid range:** 0.1 - 10.0

### Purpose:
Controls the weight of the GIoU loss component in the total loss function, balancing box regression against classification.

### Effects:
- **Higher values** (>2.0):
  - Stronger emphasis on accurate bounding box prediction
  - Better localization precision
  - Slower convergence
  - Risk of classification performance degradation if too high
  - May lead to unstable training above 5.0

- **Lower values** (<2.0):
  - Less precise bounding boxes
  - Faster convergence
  - Better classification performance
  - Risk of poor localization if too low
  - May miss small objects below 0.5

### Interaction with other parameters:
- Must be balanced with `lambda_cls`
- Higher values require lower learning rates
- Should be increased when `alpha` is high

### Recommended settings:
- Small objects: 2.5 - 3.0
- Large objects: 1.5 - 2.0
- Mixed sizes: 2.0 (default)

## 2. lambda_cls (Classification Weight)
**Default value:** 1.0  
**Valid range:** 0.1 - 5.0

### Purpose:
Controls the weight of the classification loss component in the total loss function.

### Effects:
- **Higher values** (>1.0):
  - Better class separation
  - More confident predictions
  - May overshadow localization learning
  - Risk of overconfident false positives above 3.0

- **Lower values** (<1.0):
  - Less confident class predictions
  - Better balance with localization
  - Risk of class confusion if too low
  - May lead to missed detections below 0.3

### Interaction with other parameters:
- Inversely related to `lambda_giou`
- Should be balanced with `alpha`
- Higher values work better with higher `gamma`

### Recommended settings:
- Multi-class: 1.0 - 1.5
- Binary classification: 0.8 - 1.0
- Complex backgrounds: 1.2 - 1.5

## 3. alpha (Focal Loss Alpha)
**Default value:** 2.0  
**Valid range:** 0.25 - 4.0

### Purpose:
Controls the weight of positive examples in focal loss, helping balance positive/negative samples.

### Effects:
- **Higher values** (>2.0):
  - More emphasis on positive samples
  - Better detection of rare objects
  - Risk of false positives
  - May destabilize training above 3.0

- **Lower values** (<2.0):
  - More emphasis on negative samples
  - Better false positive suppression
  - Risk of missed detections
  - May ignore small objects below 0.5

### Interaction with other parameters:
- Works in conjunction with `gamma`
- Should be balanced with `lambda_cls`
- Higher values need lower learning rates

### Recommended settings:
- Balanced dataset: 1.5 - 2.0
- Imbalanced dataset: 2.0 - 2.5
- Rare objects: 2.5 - 3.0

## 4. gamma (Focal Loss Gamma)
**Default value:** 4.0  
**Valid range:** 0.5 - 5.0

### Purpose:
Controls the rate at which easy examples are down-weighted in focal loss.

### Effects:
- **Higher values** (>4.0):
  - Stronger focus on hard examples
  - Better handling of difficult cases
  - Slower convergence
  - Risk of instability above 4.5

- **Lower values** (<4.0):
  - More balanced example weighting
  - Faster convergence
  - Better for simple datasets
  - May struggle with hard examples below 2.0

### Interaction with other parameters:
- Should be balanced with `alpha`
- Affects effective learning rate
- Higher values need smaller `lambda_cls`

### Recommended settings:
- Simple scenes: 2.0 - 3.0
- Complex scenes: 3.0 - 4.0
- Occluded objects: 4.0 - 4.5

## 5. delta (Numerical Stability)
**Default value:** 1e-6  
**Valid range:** 1e-8 - 1e-4

### Purpose:
Prevents division by zero and numerical instability in loss calculations.

### Effects:
- **Higher values** (>1e-6):
  - More stable training
  - Slightly less precise calculations
  - May affect small object detection
  - Safe choice for mixed precision training

- **Lower values** (<1e-6):
  - More precise calculations
  - Risk of numerical instability
  - May cause NaN losses
  - Requires full precision training

### Interaction with other parameters:
- More important with high `lambda_giou`
- Critical for high `gamma` values
- Should be higher with mixed precision

### Recommended settings:
- Mixed precision: 1e-5
- Full precision: 1e-6
- Small objects: 1e-7

## Parameter Interaction Matrix

| Parameter 1 | Parameter 2 | Interaction Effect |
|-------------|-------------|-------------------|
| lambda_giou ↑ | lambda_cls ↓ | Better localization, weaker classification |
| lambda_cls ↑ | alpha ↑ | Stronger positive sample focus |
| gamma ↑ | alpha ↑ | Very strong focus on hard positive samples |
| lambda_giou ↑ | delta ↑ | Needed for training stability |

## Training Recommendations

### For High Precision:
```python
loss_dict = {
    "lambda_giou": 2.5,
    "lambda_cls": 1.0,
    "alpha": 2.0,
    "gamma": 4.0,
    "delta": 1e-6
}
```

### For Fast Convergence:
```python
loss_dict = {
    "lambda_giou": 1.5,
    "lambda_cls": 1.2,
    "alpha": 1.5,
    "gamma": 2.0,
    "delta": 1e-5
}
```

### For Small Objects:
```python
loss_dict = {
    "lambda_giou": 3.0,
    "lambda_cls": 1.3,
    "alpha": 2.5,
    "gamma": 4.0,
    "delta": 1e-7
}
```

### For Stable Training:
```python
loss_dict = {
    "lambda_giou": 1.8,
    "lambda_cls": 1.0,
    "alpha": 1.8,
    "gamma": 3.0,
    "delta": 1e-5
}
```