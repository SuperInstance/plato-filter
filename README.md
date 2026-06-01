# plato-filter

> Digital signal processing filters for PLATO tile streams — moving average, Kalman, Savitzky-Golay, Butterworth, and more

## What This Does

plato-filter provides a comprehensive suite of DSP filters for cleaning noisy sensor data before further processing. It includes basic filters (moving average, exponential, median), advanced filters (Savitzky-Golay, Kalman, Butterworth), and utility filters (DC removal, derivative, integration).

## The Key Idea

Raw sensor data is noisy. A temperature reading of 22.5°C might actually be 22.3, 22.8, 22.1, 22.6, 22.4 — the noise obscures the signal. Filters extract the signal by trading off time resolution for noise reduction. Moving average is the simplest: average the last N readings. Kalman is the gold standard: a recursive estimator that maintains a belief about the true state and updates it with each noisy measurement.

## Install

```bash
cargo add plato-filter
```

## Quick Start

```rust
use plato_filter::*;

let noisy = vec![22.1, 22.8, 22.3, 23.5, 22.0, 22.4, 22.6, 21.9, 22.2, 22.5];

// Simple moving average (window = 3)
let smoothed = moving_average(&noisy, 3);

// Exponential smoothing (alpha = 0.3)
let exp = exponential_smoothing(&noisy, 0.3);

// Median filter (robust to outliers)
let med = median_filter(&noisy, 3);
```

## API Reference

### Basic Filters

| Function | Description |
|---|---|
| `moving_average(data, window)` | Simple unweighted average over sliding window |
| `exponential_smoothing(data, alpha)` | Weighted: newer values matter more. alpha ∈ (0,1). |
| `median_filter(data, window)` | Median of sliding window — robust to outliers |

### Advanced Filters

| Function | Description |
|---|---|
| `savitzky_golay(data, window, poly_order)` | Polynomial fit over window — preserves peaks better than moving average |
| `kalman_filter(data, process_noise, measurement_noise)` | Optimal recursive estimator. Models state + noise. |
| `butterworth_lowpass(data, cutoff, sample_rate, order)` | Frequency-domain filter removing high-frequency noise |

### Utility Filters

| Function | Description |
|---|---|
| `dc_removal(data)` | Remove DC offset (mean subtraction) |
| `derivative(data)` | First derivative (rate of change) |
| `integration(data)` | Cumulative sum |

| Type | Description |
|---|---|
| `FilterType` | `LowPass` / `HighPass` / `BandPass` / `BandStop` / `Notch` |
| `FilterConfig { filter_type, cutoff_freq, sample_rate, order }` | Filter configuration |
| `FilterResult { filtered, delay, group_delay }` | Filtered output with delay metadata |

## How It Works

**Kalman Filter**: Maintains estimate x and error P. Each step: predict (x = x, P = P + Q), then update with measurement (K = P/(P+R), x = x + K*(z-x), P = (1-K)*P). Q = process noise, R = measurement noise.

**Savitzky-Golay**: Fits a polynomial of degree `poly_order` to each window, then evaluates at the center point. Preserves peak shapes better than moving average.

**Butterworth**: Designs a maximally-flat frequency response filter. Higher order = sharper cutoff.

## Testing

19 tests: moving average, exponential smoothing, median filter, Kalman convergence, Savitzky-Golay peak preservation, Butterworth frequency response, DC removal, derivative, integration.

## License

Apache-2.0
