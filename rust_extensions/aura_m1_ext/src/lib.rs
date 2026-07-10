use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[cfg(target_os = "macos")]
use std::os::raw::c_int;

// Extern link to macOS-specific thread QoS APIs.
#[cfg(target_os = "macos")]
extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: c_int) -> c_int;
}

// macOS QoS Class definitions (standard for Apple Silicon)
#[cfg(target_os = "macos")]
const QOS_CLASS_USER_INITIATED: u32 = 0x19; // P-cores (Fast compute)
#[cfg(target_os = "macos")]
const QOS_CLASS_UTILITY: u32 = 0x15; // E-cores (Background IO/Sensory)

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

fn candidate_string(candidate: &Value, key: &str) -> String {
    candidate
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn candidate_line(candidate: &Value) -> i64 {
    candidate
        .get("line")
        .and_then(Value::as_i64)
        .unwrap_or_default()
}

fn compare_candidates(left: &Value, right: &Value) -> Ordering {
    let left_name = candidate_string(left, "name");
    let right_name = candidate_string(right, "name");
    (
        left_name.to_ascii_lowercase(),
        left_name,
        candidate_string(left, "module_path"),
        candidate_string(left, "class_name"),
        candidate_string(left, "source_path"),
        candidate_line(left),
    )
        .cmp(&(
            right_name.to_ascii_lowercase(),
            right_name,
            candidate_string(right, "module_path"),
            candidate_string(right, "class_name"),
            candidate_string(right, "source_path"),
            candidate_line(right),
        ))
}

fn canonicalize_skill_index_json(candidate_json: &str) -> Result<String, String> {
    let payload: Value = serde_json::from_str(candidate_json).map_err(|error| error.to_string())?;
    let mut candidates = payload
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| "skill index payload must contain a candidates array".to_string())?;
    candidates.sort_by(compare_candidates);

    let mut grouped: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for candidate in candidates {
        let name_key = candidate_string(&candidate, "name").to_ascii_lowercase();
        grouped.entry(name_key).or_default().push(candidate);
    }

    let mut accepted = Vec::new();
    let mut duplicates = Vec::new();
    for (name_key, mut group) in grouped {
        if group.len() == 1 {
            accepted.push(group.remove(0));
        } else {
            duplicates.push(json!({"candidates": group, "name_key": name_key}));
        }
    }
    serde_json::to_string(&json!({"accepted": accepted, "duplicates": duplicates}))
        .map_err(|error| error.to_string())
}

#[pyfunction]
#[pyo3(signature = (catalog_json=None))]
fn build_skill_index(py: Python<'_>, catalog_json: Option<String>) -> PyResult<PyObject> {
    let explicit_payload = catalog_json.is_some();
    let discovery = py.import_bound("core.skills.discovery")?;
    let candidate_json = match catalog_json {
        Some(payload) => payload,
        None => discovery
            .getattr("skill_index_candidates_json")?
            .call0()?
            .extract::<String>()?,
    };
    let canonical =
        canonicalize_skill_index_json(&candidate_json).map_err(PyValueError::new_err)?;
    if explicit_payload {
        return Ok(canonical.into_py(py));
    }
    let index = discovery
        .getattr("_index_dict_from_canonical_json")?
        .call1((canonical,))?;
    Ok(index.into_py(py))
}

#[pymodule]
fn aura_m1_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pin_to_p_cores, m)?)?;
    m.add_function(wrap_pyfunction!(pin_to_e_cores, m)?)?;
    m.add_function(wrap_pyfunction!(neon_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(field_integrate, m)?)?;
    m.add_function(wrap_pyfunction!(build_skill_index, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::canonicalize_skill_index_json;
    use serde_json::Value;

    #[test]
    fn canonicalizer_sorts_and_rejects_case_insensitive_duplicates() {
        let input = r#"{"candidates":[
            {"name":"zeta","module_path":"skills.z","class_name":"Z","source_path":"z.py","line":2},
            {"name":"Alpha","module_path":"skills.a","class_name":"A","source_path":"a.py","line":1},
            {"name":"alpha","module_path":"skills.b","class_name":"B","source_path":"b.py","line":1}
        ]}"#;
        let output: Value =
            serde_json::from_str(&canonicalize_skill_index_json(input).unwrap()).unwrap();
        assert_eq!(output["accepted"].as_array().unwrap().len(), 1);
        assert_eq!(output["accepted"][0]["name"], "zeta");
        assert_eq!(output["duplicates"].as_array().unwrap().len(), 1);
        assert_eq!(output["duplicates"][0]["name_key"], "alpha");
        assert_eq!(output["duplicates"][0]["candidates"][0]["name"], "Alpha");
    }
}
