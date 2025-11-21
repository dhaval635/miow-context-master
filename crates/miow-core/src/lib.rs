use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod indexer;
pub mod types;
pub mod project_signature;
pub mod intelligent_detector;
pub mod language_registry;

pub use indexer::CodebaseIndexer;
pub use types::*;
pub use project_signature::ProjectSignature;
pub use intelligent_detector::IntelligentSignatureDetector;
pub use language_registry::{LanguageRegistry, LanguageConfig};

/// Main entry point for indexing a codebase
pub async fn index_codebase(path: PathBuf) -> Result<IndexReport> {
    let mut indexer = CodebaseIndexer::new(path)?;
    indexer.index().await
}

/// Index codebase with vector store
pub async fn index_codebase_with_vector(
    path: PathBuf,
    vector_store: std::sync::Arc<miow_vector::VectorStore>,
) -> Result<IndexReport> {
    let mut indexer = CodebaseIndexer::new(path)?.with_vector_store(vector_store);
    indexer.index().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_index_codebase() {
        // Test with a sample directory
        let result = index_codebase(PathBuf::from(".")).await;
        assert!(result.is_ok());
    }
}
