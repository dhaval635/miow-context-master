use anyhow::Result;
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::VectorStore;

/// File watcher for auto-indexing
pub struct FileWatcher {
    watcher: Option<RecommendedWatcher>,
    vector_store: Arc<RwLock<VectorStore>>,
    watched_paths: Vec<PathBuf>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(vector_store: Arc<RwLock<VectorStore>>) -> Self {
        Self {
            watcher: None,
            vector_store,
            watched_paths: Vec::new(),
        }
    }
    
    /// Start watching a directory for changes
    pub fn watch(&mut self, path: impl AsRef<Path>) -> Result<Receiver<Event>> {
        let path = path.as_ref().to_path_buf();
        
        let (tx, rx) = channel();
        
        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                match res {
                    Ok(event) => {
                        if let Err(e) = tx.send(event) {
                            warn!("Failed to send file event: {}", e);
                        }
                    }
                    Err(e) => warn!("Watch error: {:?}", e),
                }
            },
            Config::default(),
        )?;
        
        watcher.watch(&path, RecursiveMode::Recursive)?;
        
        info!("Watching directory: {:?}", path);
        self.watched_paths.push(path);
        self.watcher = Some(watcher);
        
        Ok(rx)
    }
    
    /// Process file events
    pub async fn process_events(&self, rx: Receiver<Event>) -> Result<()> {
        loop {
            match rx.recv() {
                Ok(event) => {
                    if let Err(e) = self.handle_event(event).await {
                        warn!("Error handling file event: {}", e);
                    }
                }
                Err(e) => {
                    warn!("Event receiver error: {}", e);
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    async fn handle_event(&self, event: Event) -> Result<()> {
        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in event.paths {
                    if self.should_index(&path) {
                        debug!("File changed, re-indexing: {:?}", path);
                        self.reindex_file(&path).await?;
                    }
                }
            }
            EventKind::Remove(_) => {
                for path in event.paths {
                    debug!("File removed: {:?}", path);
                    // Could implement deletion from vector store here
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    fn should_index(&self, path: &Path) -> bool {
        // Only index code files
        if let Some(ext) = path.extension() {
            matches!(
                ext.to_str(),
                Some("rs") | Some("ts") | Some("tsx") | Some("js") | Some("jsx") | Some("py")
            )
        } else {
            false
        }
    }
    
    async fn reindex_file(&self, _path: &Path) -> Result<()> {
        // This would trigger re-parsing and re-indexing
        // For now, just a placeholder
        // In production, this would:
        // 1. Parse the file
        // 2. Extract symbols
        // 3. Update vector store
        
        Ok(())
    }
    
    /// Stop watching
    pub fn stop(&mut self) {
        self.watcher = None;
        self.watched_paths.clear();
        info!("File watcher stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_should_index() {
        let store = Arc::new(RwLock::new(
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(VectorStore::new("http://localhost:6333", "test"))
                .unwrap(),
        ));
        let watcher = FileWatcher::new(store);
        
        assert!(watcher.should_index(Path::new("test.rs")));
        assert!(watcher.should_index(Path::new("test.ts")));
        assert!(!watcher.should_index(Path::new("test.md")));
    }
}
