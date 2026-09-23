//! Freshness watcher: re-sync the index on workspace changes.
//!
//! Debounced (2 s quiet period) so save bursts trigger one sync.
//! Runs in the foreground until interrupted.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use crate::Error;

/// Quiet period before a burst of events triggers a sync.
const DEBOUNCE: Duration = Duration::from_secs(2);

/// Whether a watcher event path should trigger a re-sync.
#[must_use]
pub fn relevant(path: &Path) -> bool {
    !path.components().any(|c| {
        c.as_os_str() == ".one-grep"
            || c.as_os_str() == ".git"
            || c.as_os_str() == "target"
            || c.as_os_str() == "node_modules"
    })
}

/// Block until interrupted, syncing on relevant changes.
///
/// # Errors
///
/// Returns [`Error`] when the workspace is not a directory, watching
/// fails, or a sync fails.
pub fn run(workspace: &Path) -> Result<(), Error> {
    if !workspace.is_dir() {
        return Err(Error::InvalidInput(format!(
            "workspace is not a directory: {}",
            workspace.display()
        )));
    }
    let workspace: PathBuf = workspace.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher = notify::recommended_watcher(tx)
        .map_err(|e| Error::InvalidInput(format!("watch failed: {e}")))?;
    use notify::Watcher as _;
    watcher
        .watch(&workspace, notify::RecursiveMode::Recursive)
        .map_err(|e| Error::InvalidInput(format!("watch failed: {e}")))?;
    eprintln!("watching {}", workspace.display());

    let mut pending = false;
    let mut quiet_since = Instant::now();
    loop {
        match rx.recv_timeout(DEBOUNCE) {
            Ok(Ok(event)) => {
                use notify::EventKind::*;
                let interesting = matches!(event.kind, Create(_) | Modify(_) | Remove(_))
                    && event.paths.iter().any(|p| relevant(p));
                if interesting {
                    pending = true;
                    quiet_since = Instant::now();
                }
            }
            Ok(Err(e)) => {
                eprintln!("watch error: {e}");
                // Keep watching; transient errors are normal.
                std::thread::sleep(DEBOUNCE);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if pending && quiet_since.elapsed() >= DEBOUNCE {
                    pending = false;
                    let stats = crate::index::sync(&workspace)?;
                    eprintln!(
                        "synced: {} scanned, {} upserted, {} chunks, {} removed",
                        stats.scanned, stats.upserted, stats.chunks, stats.removed
                    );
                    // The index moved without the vectors: mark them stale
                    // instead of embedding inline (model load is too heavy
                    // for a watcher). `embed` clears the marker.
                    if stats.upserted + stats.removed > 0
                        && crate::vectors::store_model(&workspace)
                            .map(|m| m.is_some())
                            .unwrap_or(false)
                    {
                        crate::vectors::mark_stale(&workspace)?;
                        eprintln!("{}", crate::vectors::stale_note(&workspace));
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relevant_filters_generated_dirs() {
        assert!(relevant(Path::new("/ws/src/main.rs")));
        assert!(!relevant(Path::new("/ws/.one-grep/index/foo")));
        assert!(!relevant(Path::new("/ws/.git/refs/heads")));
        assert!(!relevant(Path::new("/ws/target/debug/x")));
        assert!(!relevant(Path::new("/ws/node_modules/x")));
    }
}
