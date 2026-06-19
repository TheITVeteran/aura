use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[cfg(target_os = "macos")]
use std::os::raw::c_int;

// Extern link to macOS-specific thread QoS APIs.
#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
}

// macOS QoS Class definitions (standard for Apple Silicon)
#[cfg(target_os = "macos")]
const QOS_CLASS_USER_INITIATED: u32 = 0x19;   // P-cores (Fast compute)
#[cfg(target_os = "macos")]
const QOS_CLASS_UTILITY: u32 = 0x15;          // E-cores (Background IO/Sensory)

#[pyfunction]
fn pin_to_p_cores() {
    #[cfg(target_os = "macos")]
    unsafe {
        // Elevate current thread to User Initiated QoS (Apple Silicon P-Cores)
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    }
}

#[pyfunction]
fn pin_to_e_cores() {
    #[cfg(target_os = "macos")]
    unsafe {
        // Set current thread to Utility QoS (Apple Silicon E-Cores for low power/IO)
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    }
}

#[allow(dead_code)] // used only in the non-aarch64 fallback of neon_dot_product
fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// Apple Silicon NEON-accelerated dot product (Zero-copy)
#[pyfunction]
fn neon_dot_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        return neon_dot_product_aarch64(&a, &b);
    }

    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        scalar_dot_product(&a, &b)
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn neon_dot_product_aarch64(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let len = a.len().min(b.len());
    let mut sum = 0.0f32;
    let mut i = 0;
    
    // Process in blocks of 4 using NEON intrinsics
    while i + 4 <= len {
        unsafe {
            let va = vld1q_f32(a[i..].as_ptr());
            let vb = vld1q_f32(b[i..].as_ptr());
            let prod = vmulq_f32(va, vb);
            sum += vaddvq_f32(prod); // Vector across-lane sum
        }
        i += 4;
    }
    
    // Scalar tail for remaining elements
    for j in i..len {
        sum += a[j] * b[j];
    }
    sum
}

// Fused Euler integration step for the continuous-time unified field.
// Mirrors core/consciousness/unified_field._tick exactly:
//   next[i] = clamp(f[i] + (-decay*f[i] + activity[i] + noise[i]) * dt, -1, 1)
// Zero-copy: reads the numpy buffers directly (PyReadonlyArray1) and returns a
// numpy array, avoiding per-call list marshalling. One tight loop replaces ~4
// numpy temporaries (recurrent+input already folded into `activity`).
#[pyfunction]
fn field_integrate<'py>(
    py: Python<'py>,
    f: PyReadonlyArray1<'py, f32>,
    activity: PyReadonlyArray1<'py, f32>,
    noise: PyReadonlyArray1<'py, f32>,
    decay: f32,
    dt: f32,
) -> Bound<'py, PyArray1<f32>> {
    let f = f.as_slice().unwrap_or(&[]);
    let a = activity.as_slice().unwrap_or(&[]);
    let nz = noise.as_slice().unwrap_or(&[]);
    let n = f.len();
    let mut out = vec![0f32; n];
    for i in 0..n {
        let av = if i < a.len() { a[i] } else { 0.0 };
        let nv = if i < nz.len() { nz[i] } else { 0.0 };
        let df = (-decay * f[i] + av + nv) * dt;
        let mut v = f[i] + df;
        if v > 1.0 {
            v = 1.0;
        } else if v < -1.0 {
            v = -1.0;
        }
        out[i] = v;
    }
    PyArray1::from_vec_bound(py, out)
}

#[pymodule]
fn aura_m1_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pin_to_p_cores, m)?)?;
    m.add_function(wrap_pyfunction!(pin_to_e_cores, m)?)?;
    m.add_function(wrap_pyfunction!(neon_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(field_integrate, m)?)?;
    Ok(())
}
