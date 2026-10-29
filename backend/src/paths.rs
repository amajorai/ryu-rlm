//! Shared Ryu sidecar data-directory seam.

use std::path::PathBuf;

/// This app's own data dir: `<data dir>/plugin-data/@ryu+rlm`.
pub fn data_dir() -> PathBuf {
    ryu_sidecar_runtime::plugin_data_dir("@ryu/rlm")
}
