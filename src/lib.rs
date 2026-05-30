//! Digital signal processing filters for PLATO tile streams.
//!
//! Provides a suite of filters for cleaning noisy sensor data before
//! JEPA embedding. Includes moving average, exponential, median,
//! Savitzky-Golay, Kalman, Butterworth, notch, derivative, integration,
//! and DC removal filters.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Available filter types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    BandStop,
    Notch,
}

/// Configuration for a filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    pub filter_type: FilterType,
    pub cutoff_freq: f64,
    pub sample_rate: f64,
    pub order: usize,
}

/// Result returned by advanced filters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterResult {
    pub filtered: Vec<f64>,
    pub delay: usize,
    pub group_delay: f64,
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Compute median of a slice. Caller must ensure non-empty.
fn median_of(slice: &[f64]) -> f64 {
    debug_assert!(!slice.is_empty());
    let mut v = slice.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

/// Invert a square matrix via Gauss-Jordan elimination.
fn invert_matrix(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = mat.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = mat[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        assert!(pivot.abs() > 1e-14, "Singular matrix in inversion");
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..2 * n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    inv
}

// ---------------------------------------------------------------------------
// Filter functions
// ---------------------------------------------------------------------------

/// Simple moving average filter.
///
/// Each output sample is the mean of `window` surrounding values (centered).
/// Output length equals input length; edges are handled with available samples.
pub fn moving_average_filter(data: &[f64], window: usize) -> Vec<f64> {
    if data.is_empty() || window == 0 {
        return data.to_vec();
    }
    let n = data.len();
    let mut out = Vec::with_capacity(n);
    let half = window / 2;

    // Use cumulative sum for efficiency
    let mut cumsum = vec![0.0; n + 1];
    for i in 0..n {
        cumsum[i + 1] = cumsum[i] + data[i];
    }

    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let count = hi - lo;
        out.push((cumsum[hi] - cumsum[lo]) / count as f64);
    }
    out
}

/// Exponential moving average (EMA) filter.
///
/// `alpha` ∈ (0, 1]. Higher alpha = less smoothing (faster response).
pub fn exponential_filter(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() {
        return data.to_vec();
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(data.len());
    let mut prev = data[0];
    out.push(prev);
    for &x in &data[1..] {
        let y = alpha * x + (1.0 - alpha) * prev;
        out.push(y);
        prev = y;
    }
    out
}

/// Median filter — removes spike noise that moving average misses.
///
/// Each output sample is the median of `window` surrounding values.
pub fn median_filter(data: &[f64], window: usize) -> Vec<f64> {
    if data.is_empty() || window == 0 {
        return data.to_vec();
    }
    let n = data.len();
    let half = window / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        out.push(median_of(&data[lo..hi]));
    }
    out
}

/// Savitzky-Golay filter — smooths while preserving peaks better than MA.
///
/// Fits a polynomial of `poly_order` to `window` points around each sample
/// and evaluates the central value from the polynomial.
pub fn savgol_filter(data: &[f64], window: usize, poly_order: usize) -> Vec<f64> {
    if data.is_empty() || window == 0 {
        return data.to_vec();
    }
    assert!(window % 2 == 1, "Savitzky-Golay window must be odd");
    assert!(poly_order < window, "poly_order must be less than window");
    let n = data.len();
    let half = window as i64 / 2;
    let m = poly_order + 1;

    // Build design matrix (Vandermonde) for the central window positions
    let mut design: Vec<Vec<f64>> = Vec::with_capacity(window);
    for k in 0..window {
        let t = k as f64 - half as f64;
        let mut row = Vec::with_capacity(m);
        let mut val = 1.0;
        for _ in 0..m {
            row.push(val);
            val *= t;
        }
        design.push(row);
    }

    // Compute least-squares coefficients: c = (A^T A)^-1 A^T
    // A^T A is m×m
    let ata = {
        let mut r = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0;
                for k in 0..window {
                    s += design[k][i] * design[k][j];
                }
                r[i][j] = s;
            }
        }
        r
    };
    let inv = invert_matrix(&ata);

    // coeffs[row] = sum over j of inv[row][j] * design[half][j]
    // But we want the smoothing (convolution) coefficients:
    // c_i = sum_j ( (A^T A)^-1 A^T )[j, half] * ... 
    // Actually: the filter coefficients are c = A (A^T A)^{-1} A^T e_center
    // where e_center is the unit vector at position half
    // So c_i = sum_j design[i][j] * sum_k inv[j][k] * design[half][k]
    let mut coeffs = vec![0.0; window];
    for i in 0..window {
        let mut ci = 0.0;
        for j in 0..m {
            let mut aj_inv = 0.0;
            for k in 0..m {
                aj_inv += inv[j][k] * design[half as usize][k];
            }
            ci += design[i][j] * aj_inv;
        }
        coeffs[i] = ci;
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut y = 0.0;
        for j in 0..window {
            let idx = i as i64 - half + j as i64;
            let val = if idx < 0 {
                data[0]
            } else if idx >= n as i64 {
                data[n - 1]
            } else {
                data[idx as usize]
            };
            y += coeffs[j] * val;
        }
        out.push(y);
    }
    out
}

/// Simple 1-D Kalman filter.
///
/// Tracks a scalar signal with `process_noise` (Q) and `measurement_noise` (R).
pub fn kalman_simple(data: &[f64], process_noise: f64, measurement_noise: f64) -> Vec<f64> {
    if data.is_empty() {
        return data.to_vec();
    }
    let q = process_noise;
    let r = measurement_noise;
    let mut x = data[0];
    let mut p = 1.0;
    let mut out = Vec::with_capacity(data.len());
    for &z in data {
        let p_pred = p + q;
        let k = p_pred / (p_pred + r);
        x = x + k * (z - x);
        p = (1.0 - k) * p_pred;
        out.push(x);
    }
    out
}

/// Butterworth low-pass filter implemented via cascaded 2nd-order sections.
///
/// Applies forward-backward filtering for zero phase delay.
pub fn butterworth_lowpass(data: &[f64], cutoff: f64, sample_rate: f64, order: usize) -> Vec<f64> {
    if data.is_empty() || order == 0 {
        return data.to_vec();
    }
    assert!(cutoff > 0.0 && sample_rate > 0.0 && cutoff < sample_rate / 2.0,
        "Invalid cutoff/sample_rate");

    // Use bilinear transform with frequency prewarping
    let nyq = sample_rate / 2.0;
    let fc = cutoff / nyq; // normalized cutoff 0..1
    let n_sections = (order + 1) / 2;

    let apply_sos = |data: &[f64], b: &[f64; 3], a: &[f64; 2]| -> Vec<f64> {
        let mut out = Vec::with_capacity(data.len());
        let mut w1 = 0.0_f64;
        let mut w2 = 0.0_f64;
        for &x in data {
            let w0 = x - a[0] * w1 - a[1] * w2;
            let y = b[0] * w0 + b[1] * w1 + b[2] * w2;
            w2 = w1;
            w1 = w0;
            out.push(y);
        }
        out
    };

    // Compute second-order sections using analog prototype + bilinear transform
    let mut sections: Vec<([f64; 3], [f64; 2])> = Vec::with_capacity(n_sections);
    let warp = |f: f64| (std::f64::consts::PI * f).tan();

    for k in 1..=n_sections {
        if order % 2 == 1 && k == n_sections && n_sections * 2 > order {
            // First-order section for odd order
            let fcw = warp(fc);
            let b0 = fcw / (1.0 + fcw);
            let b1 = b0;
            let a1 = (fcw - 1.0) / (fcw + 1.0);
            sections.push(([b0, b1, 0.0], [a1, 0.0]));
        } else {
            // Second-order section
            let angle = std::f64::consts::PI * (2.0 * k as f64 - 1.0) / (2.0 * order as f64);
            let sigma = -angle.cos(); // real part of analog pole
            let omega = angle.sin();  // imaginary part

            let fcw = warp(fc);
            let k_val = fcw * fcw;

            let denom = 1.0 - 2.0 * sigma * fcw + k_val;
            let b0 = k_val / denom;
            let b1 = 2.0 * k_val / denom;
            let b2 = k_val / denom;
            let a1 = 2.0 * (k_val - 1.0) / denom;
            let a2 = (1.0 + 2.0 * sigma * fcw + k_val) / denom;

            sections.push(([b0, b1, b2], [a1, a2]));
        }
    }

    // Forward pass
    let mut sig = data.to_vec();
    for (b, a) in &sections {
        sig = apply_sos(&sig, b, a);
    }
    // Backward pass (zero-phase)
    sig.reverse();
    for (b, a) in &sections {
        sig = apply_sos(&sig, b, a);
    }
    sig.reverse();
    sig
}

/// Numerical first derivative using central differences.
///
/// Uses forward/backward differences at the boundaries.
pub fn derivative(data: &[f64], dt: f64) -> Vec<f64> {
    if data.is_empty() {
        return data.to_vec();
    }
    assert!(dt > 0.0, "dt must be positive");
    let n = data.len();
    if n == 1 {
        return vec![0.0];
    }
    let mut out = Vec::with_capacity(n);
    out.push((data[1] - data[0]) / dt);
    for i in 1..n - 1 {
        out.push((data[i + 1] - data[i - 1]) / (2.0 * dt));
    }
    out.push((data[n - 1] - data[n - 2]) / dt);
    out
}

/// Numerical integration using the trapezoidal rule.
pub fn integrate(data: &[f64], dt: f64) -> Vec<f64> {
    if data.is_empty() {
        return data.to_vec();
    }
    assert!(dt > 0.0, "dt must be positive");
    let mut out = Vec::with_capacity(data.len());
    let mut acc = 0.0;
    out.push(0.0);
    for i in 1..data.len() {
        acc += (data[i] + data[i - 1]) / 2.0 * dt;
        out.push(acc);
    }
    out
}

/// Remove DC offset (mean) from the signal.
pub fn remove_dc(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return data.to_vec();
    }
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&x| x - mean).collect()
}

/// Notch (band-stop) filter to remove a specific frequency.
///
/// Uses a 2nd-order IIR notch. `q` controls bandwidth (higher = narrower).
pub fn notch_filter(data: &[f64], freq: f64, sample_rate: f64, q: f64) -> Vec<f64> {
    if data.is_empty() {
        return data.to_vec();
    }
    assert!(freq > 0.0 && sample_rate > 0.0 && freq < sample_rate / 2.0);
    let w0 = 2.0 * std::f64::consts::PI * freq / sample_rate;
    let alpha = w0.sin() / (2.0 * q);

    let a0 = 1.0 + alpha;
    let b0 = 1.0 / a0;
    let b1 = -2.0 * w0.cos() / a0;
    let b2 = 1.0 / a0;
    let a1 = -2.0 * w0.cos() / a0;
    let a2 = (1.0 - alpha) / a0;

    let mut out = Vec::with_capacity(data.len());
    let mut w1 = 0.0_f64;
    let mut w2 = 0.0_f64;
    for &x in data {
        let w0_val = x - a1 * w1 - a2 * w2;
        let y = b0 * w0_val + b1 * w1 + b2 * w2;
        w2 = w1;
        w1 = w0_val;
        out.push(y);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: generate a noisy sine wave.
    fn noisy_sine(n: usize, freq: f64, sr: f64, noise_amp: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                let noise = noise_amp * ((i as f64 * 7.3 + 1.1).sin() * 2.0);
                (2.0 * PI * freq * t).sin() + noise
            })
            .collect()
    }

    // -- Moving Average --------------------------------------------------

    #[test]
    fn ma_smooths_noise() {
        let data: Vec<f64> = (0..100)
            .map(|i| (i as f64 / 10.0).sin() + 0.5 * (i as f64 * 0.3).cos())
            .collect();
        let filtered = moving_average_filter(&data, 5);
        assert_eq!(filtered.len(), data.len());
        let var_orig: f64 = {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
        };
        let var_filt: f64 = {
            let mean = filtered.iter().sum::<f64>() / filtered.len() as f64;
            filtered.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / filtered.len() as f64
        };
        assert!(var_filt < var_orig, "MA should reduce variance");
    }

    #[test]
    fn ma_preserves_linear_trend() {
        let data: Vec<f64> = (0..50).map(|i| i as f64 * 2.0).collect();
        let filtered = moving_average_filter(&data, 5);
        // Away from edges, MA should be very close to original
        for i in 5..45 {
            assert!((data[i] - filtered[i]).abs() < 1e-10,
                "MA should preserve linear trend at index {}: got {} expected {}", i, filtered[i], data[i]);
        }
    }

    // -- Exponential Filter ----------------------------------------------

    #[test]
    fn ema_alpha_sensitivity() {
        let signal = vec![0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let slow = exponential_filter(&signal, 0.1);
        let fast = exponential_filter(&signal, 0.9);
        assert!(fast[4].abs() > slow[4].abs(), "Higher alpha should react faster");
    }

    #[test]
    fn ema_single_value() {
        assert_eq!(exponential_filter(&[42.0], 0.5), vec![42.0]);
    }

    // -- Median Filter ---------------------------------------------------

    #[test]
    fn median_removes_spikes() {
        let mut data = vec![1.0; 20];
        data[5] = 100.0;
        data[15] = -50.0;
        let ma = moving_average_filter(&data, 3);
        let med = median_filter(&data, 3);
        assert!((ma[5] - 1.0).abs() > (med[5] - 1.0).abs());
        assert!((ma[15] - 1.0).abs() > (med[15] - 1.0).abs());
    }

    // -- Savitzky-Golay ---------------------------------------------------

    #[test]
    fn savgol_preserves_peaks() {
        let data: Vec<f64> = (0..50)
            .map(|i| {
                let x = i as f64 / 10.0 - 2.5;
                (-x * x / 2.0).exp()
            })
            .collect();
        let ma = moving_average_filter(&data, 7);
        let sg = savgol_filter(&data, 7, 3);
        let orig_peak = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ma_peak = ma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sg_peak = sg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ma_err = (ma_peak - orig_peak).abs();
        let sg_err = (sg_peak - orig_peak).abs();
        assert!(sg_err <= ma_err + 0.01,
            "SG should preserve peak: orig={}, ma={}, sg={}", orig_peak, ma_peak, sg_peak);
    }

    // -- Kalman -----------------------------------------------------------

    #[test]
    fn kalman_tracks_signal() {
        let signal: Vec<f64> = (0..200).map(|i| (2.0 * PI * i as f64 / 50.0).sin()).collect();
        let noisy: Vec<f64> = signal.iter().enumerate().map(|(i, v)| {
            v + 0.3 * ((i * 7 + 3) as f64 * 0.1).sin()
        }).collect();
        // Use larger process noise so Kalman can track the oscillation
        let filtered = kalman_simple(&noisy, 0.1, 0.01);
        let error_noisy: f64 = noisy.iter().zip(&signal).map(|(n, s)| (n - s).powi(2)).sum();
        let error_filt: f64 = filtered.iter().zip(&signal).map(|(f, s)| (f - s).powi(2)).sum();
        assert!(error_filt < error_noisy, "Kalman should reduce MSE: filt={}, noisy={}", error_filt, error_noisy);
    }

    // -- Derivative -------------------------------------------------------

    #[test]
    fn derivative_known_function() {
        let dt = 0.01;
        let x: Vec<f64> = (0..100).map(|i| i as f64 * dt).collect();
        let fx: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let dfx = derivative(&fx, dt);
        let mid = 50;
        let expected = 2.0 * x[mid];
        assert!((dfx[mid] - expected).abs() < 0.05,
            "Derivative should approximate 2x, got {} expected {}", dfx[mid], expected);
    }

    // -- Integration ------------------------------------------------------

    #[test]
    fn integration_roundtrip_with_derivative() {
        let dt = 0.01;
        let n = 200;
        let fx: Vec<f64> = (0..n).map(|i| (i as f64 * dt * 2.0).sin()).collect();
        let integrated = integrate(&fx, dt);
        let recovered = derivative(&integrated, dt);
        for i in 20..n - 20 {
            let err = (recovered[i] - fx[i]).abs();
            assert!(err < 0.05, "Roundtrip error too large at {}: {}", i, err);
        }
    }

    // -- DC Removal -------------------------------------------------------

    #[test]
    fn dc_removal_zero_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nodc = remove_dc(&data);
        let mean: f64 = nodc.iter().sum::<f64>() / nodc.len() as f64;
        assert!(mean.abs() < 1e-12, "DC removal should produce zero mean, got {}", mean);
    }

    #[test]
    fn dc_removal_preserves_shape() {
        let data = vec![10.0, 11.0, 12.0, 11.0, 10.0];
        let nodc = remove_dc(&data);
        // Mean = 10.8
        let expected = vec![-0.8, 0.2, 1.2, 0.2, -0.8];
        for (a, b) in nodc.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-12, "got {} expected {}", a, b);
        }
    }

    // -- Notch Filter -----------------------------------------------------

    #[test]
    fn notch_removes_frequency() {
        let sr = 1000.0;
        let freq = 50.0;
        let n = 2000;
        let signal: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / sr;
            (2.0 * PI * 10.0 * t).sin() + (2.0 * PI * freq * t).sin()
        }).collect();
        let filtered = notch_filter(&signal, freq, sr, 30.0);
        let pure: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / sr;
            (2.0 * PI * 10.0 * t).sin()
        }).collect();
        let error_before: f64 = signal.iter().zip(&pure).map(|(s, p)| (s - p).powi(2)).sum::<f64>() / n as f64;
        let error_after: f64 = filtered.iter().zip(&pure).map(|(f, p)| (f - p).powi(2)).sum::<f64>() / n as f64;
        assert!(error_after < error_before, "Notch should reduce 50Hz: after={}, before={}", error_after, error_before);
    }

    // -- Edge Cases -------------------------------------------------------

    #[test]
    fn empty_input() {
        let empty: Vec<f64> = vec![];
        assert_eq!(moving_average_filter(&empty, 5), empty);
        assert_eq!(exponential_filter(&empty, 0.5), empty);
        assert_eq!(median_filter(&empty, 3), empty);
        assert_eq!(kalman_simple(&empty, 0.01, 0.1), empty);
        assert_eq!(butterworth_lowpass(&empty, 10.0, 100.0, 2), empty);
        assert_eq!(derivative(&empty, 0.01), empty);
        assert_eq!(integrate(&empty, 0.01), empty);
        assert_eq!(remove_dc(&empty), empty);
        assert_eq!(notch_filter(&empty, 50.0, 1000.0, 30.0), empty);
    }

    #[test]
    fn single_point() {
        let single = vec![42.0];
        assert_eq!(moving_average_filter(&single, 5), single);
        assert_eq!(exponential_filter(&single, 0.5), single);
        assert_eq!(median_filter(&single, 3), single);
        assert_eq!(kalman_simple(&single, 0.01, 0.1), single);
        assert_eq!(remove_dc(&single), vec![0.0]);
        assert_eq!(derivative(&single, 0.1), vec![0.0]);
        assert_eq!(integrate(&single, 0.1), vec![0.0]);
    }

    #[test]
    fn constant_signal() {
        let c = vec![5.0; 100];
        let ma = moving_average_filter(&c, 11);
        let med = median_filter(&c, 7);
        let kf = kalman_simple(&c, 0.001, 0.1);
        for (a, b) in ma.iter().zip(&c) {
            assert!((a - b).abs() < 1e-10, "MA should preserve constant");
        }
        for (a, b) in med.iter().zip(&c) {
            assert!((a - b).abs() < 1e-10, "Median should preserve constant");
        }
        // Kalman should converge to constant
        let last_20_avg: f64 = kf[80..].iter().sum::<f64>() / 20.0;
        assert!((last_20_avg - 5.0).abs() < 0.5, "Kalman should converge: got {}", last_20_avg);
    }

    #[test]
    fn step_function() {
        let step: Vec<f64> = (0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();
        let ma = moving_average_filter(&step, 5);
        assert!(ma[40] < 0.1, "Before step, MA should be near 0: got {}", ma[40]);
        assert!(ma[60] > 0.9, "After step, MA should be near 1: got {}", ma[60]);
    }

    // -- Filter Comparison ------------------------------------------------

    #[test]
    fn compare_filters_on_noisy_sine() {
        let noisy = noisy_sine(200, 5.0, 100.0, 0.3);
        let ma = moving_average_filter(&noisy, 7);
        let ema = exponential_filter(&noisy, 0.3);
        let med = median_filter(&noisy, 7);
        let kf = kalman_simple(&noisy, 0.01, 0.1);
        assert_eq!(ma.len(), noisy.len());
        assert_eq!(ema.len(), noisy.len());
        assert_eq!(med.len(), noisy.len());
        assert_eq!(kf.len(), noisy.len());
    }

    // -- Butterworth -------------------------------------------------------

    #[test]
    fn butterworth_attenuates_high_freq() {
        let sr = 1000.0;
        let n = 2000;
        let signal: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / sr;
            (2.0 * PI * 10.0 * t).sin() + (2.0 * PI * 200.0 * t).sin()
        }).collect();
        let filtered = butterworth_lowpass(&signal, 50.0, sr, 4);
        let target: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / sr;
            (2.0 * PI * 10.0 * t).sin()
        }).collect();
        // Skip transient (first 200 samples)
        let err_filt: f64 = filtered[200..].iter().zip(&target[200..]).map(|(f, t)| (f - t).powi(2)).sum::<f64>() / (n - 200) as f64;
        let err_orig: f64 = signal[200..].iter().zip(&target[200..]).map(|(s, t)| (s - t).powi(2)).sum::<f64>() / (n - 200) as f64;
        assert!(err_filt < err_orig * 0.3,
            "Butterworth should attenuate high freq: err_filt={} err_orig={}", err_filt, err_orig);
    }

    // -- Additional tests ------------------------------------------------

    #[test]
    fn savgol_linear_signal_is_ma() {
        // SG with poly_order=1 on a linear signal should reproduce MA exactly
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let sg = savgol_filter(&data, 5, 1);
        let ma = moving_average_filter(&data, 5);
        // Interior points should match
        for i in 5..25 {
            assert!((sg[i] - ma[i]).abs() < 1e-10,
                "SG(5,1) should match MA(5) at {}: sg={} ma={}", i, sg[i], ma[i]);
        }
    }

    #[test]
    fn integration_of_constant() {
        let c = vec![3.0; 10];
        let integ = integrate(&c, 1.0);
        for i in 0..10 {
            assert!((integ[i] - 3.0 * i as f64).abs() < 1e-10,
                "integ[{}] = {} expected {}", i, integ[i], 3.0 * i as f64);
        }
    }

    #[test]
    fn filter_config_serde_roundtrip() {
        let config = FilterConfig {
            filter_type: FilterType::LowPass,
            cutoff_freq: 50.0,
            sample_rate: 1000.0,
            order: 4,
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: FilterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.filter_type, FilterType::LowPass);
        assert!((back.cutoff_freq - 50.0).abs() < 1e-10);
        assert_eq!(back.order, 4);
    }

    #[test]
    fn filter_result_serde_roundtrip() {
        let result = FilterResult {
            filtered: vec![1.0, 2.0, 3.0],
            delay: 2,
            group_delay: 1.5,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: FilterResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.filtered, vec![1.0, 2.0, 3.0]);
        assert_eq!(back.delay, 2);
    }

    #[test]
    fn filter_type_variants() {
        let types = vec![FilterType::LowPass, FilterType::HighPass, FilterType::BandPass, FilterType::BandStop, FilterType::Notch];
        for ft in types {
            let json = serde_json::to_string(&ft).unwrap();
            let back: FilterType = serde_json::from_str(&json).unwrap();
            assert_eq!(ft, back);
        }
    }

    #[test]
    fn ema_alpha_clamping() {
        let data = vec![1.0, 2.0, 3.0];
        let out_zero = exponential_filter(&data, -1.0);
        let out_one = exponential_filter(&data, 1.0);
        // alpha=1.0 should just pass through
        assert_eq!(out_one, data);
        // alpha clamped to 0 should give constant
        assert_eq!(out_zero, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn derivative_constant_is_zero() {
        let data = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let d = derivative(&data, 0.1);
        for val in &d {
            assert!(val.abs() < 1e-12, "Derivative of constant should be zero: got {}", val);
        }
    }

    #[test]
    fn median_even_window() {
        let data = vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0];
        let result = median_filter(&data, 4);
        assert_eq!(result.len(), data.len());
    }
}
