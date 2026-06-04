use pyo3::prelude::*;
use std::fs;
use std::io;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[pyfunction]
pub fn run_python_subprocess(
    python_executable: &str,
    code: &str,
    base_dir: &str,
    timeout_ms: u64,
    max_output_chars: usize,
) -> PyResult<(String, String, i32, u64)> {
    let run_dir = make_run_dir(base_dir)?;
    let script_path = run_dir.join("main.py");
    fs::write(&script_path, code)
        .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;

    let start = Instant::now();
    let mut child = Command::new(python_executable)
        .arg(&script_path)
        .current_dir(&run_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;

    loop {
        if child
            .try_wait()
            .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?
            .is_some()
        {
            break;
        }
        if start.elapsed() >= Duration::from_millis(timeout_ms) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(pyo3::exceptions::PyTimeoutError::new_err(
                "Sandbox execution timed out",
            ));
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    let output = child
        .wait_with_output()
        .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;
    let latency_ms = start.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
    let stdout = truncate_utf8(String::from_utf8_lossy(&output.stdout), max_output_chars);
    let stderr = truncate_utf8(String::from_utf8_lossy(&output.stderr), max_output_chars);
    let exit_code = output.status.code().unwrap_or(-1);
    Ok((stdout, stderr, exit_code, latency_ms))
}

fn make_run_dir(base_dir: &str) -> PyResult<std::path::PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?
        .as_nanos();
    let run_dir = Path::new(base_dir).join(format!("rust-run-{nanos}"));
    fs::create_dir_all(&run_dir)
        .map_err(|error: io::Error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;
    Ok(run_dir)
}

fn truncate_utf8(value: std::borrow::Cow<'_, str>, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}
