# Chemlara Predictor - Time Estimation Features

## 🕐 New Time Estimation Features

### 1. **Pre-Training Time Estimation**
- Calculates estimated training time based on:
  - Dataset size (number of molecules)
  - Feature complexity (fingerprint type and size)
  - TPOT parameters (generations, population size, CV folds)
- Shows confirmation dialog before starting training

### 2. **Real-Time Progress Tracking**
- **Adaptive Progress Bar**: Updates based on actual training speed
- **Time Remaining**: Continuously recalculates based on current performance
- **Elapsed Time**: Shows how long training has been running
- **Completion Percentage**: Visual progress indicator

### 3. **Smart Time Estimation Algorithm**
```python
def estimate_training_time(n_samples, n_features, generations, population_size, cv_folds):
    base_time_per_pipeline = 2.0  # seconds
    complexity_factor = (n_samples * n_features) / 10000
    cv_factor = cv_folds * 1.2
    generation_factor = generations * population_size * 0.1
    
    estimated_seconds = base_time_per_pipeline * complexity_factor * cv_factor + generation_factor
    return max(estimated_seconds * 1.2, 30)  # 20% buffer, minimum 30s
```

### 4. **Progress Display Features**
- **Evaluation Counter**: Shows current vs total evaluations (e.g., "45/100 evaluations")
- **Time Formatting**: Displays time in human-readable format (e.g., "2m 30s", "1h 15m")
- **Status Updates**: Progress milestones with encouraging messages
- **Completion Summary**: Compares actual vs estimated time

### 5. **Training Confirmation Dialog**
Before starting training, users see:
- Estimated training duration
- Total number of ML pipelines to evaluate
- Dataset size information
- Warning about non-pausable process

### 6. **Adaptive Progress Updates**
- **Initial Phase**: Uses estimated time per evaluation
- **Learning Phase**: After 3-5 evaluations, adapts based on actual performance
- **Smoothing**: Uses exponential moving average to avoid jerky updates
- **Final Phase**: Adds buffer for final optimization steps

## 📊 Example Time Estimates

| Dataset Size | Generations | CV Folds | Estimated Time |
|-------------|-------------|----------|----------------|
| 50 molecules | 5 | 5 | ~2-3 minutes |
| 100 molecules | 10 | 5 | ~8-12 minutes |
| 500 molecules | 15 | 5 | ~25-35 minutes |
| 1000+ molecules | 20 | 10 | ~1-2 hours |

## 🎯 Benefits

1. **Better User Experience**: Users know what to expect
2. **Informed Decisions**: Can choose appropriate parameters based on time constraints
3. **Progress Visibility**: Real-time feedback during long training sessions
4. **Adaptive Accuracy**: Time estimates improve as training progresses
5. **Mobile-Friendly**: Works well on all devices with clear, visual progress indicators

## 💡 Tips for Users

- **Start Small**: Begin with fewer generations to test your dataset
- **Plan Ahead**: Check estimated time before starting training
- **Stay Connected**: Don't close the browser during training
- **Monitor Progress**: Watch for status updates and time remaining
- **Be Patient**: Complex datasets and high generations take longer but produce better models
