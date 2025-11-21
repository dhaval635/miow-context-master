use anyhow::{bail, Result};
use reqwest::{Client, StatusCode};
use serde::Serialize;
use serde_json::Value;
use tracing::{debug, info, warn};

pub mod file_watcher;
pub mod hybrid_search;
pub mod smart_chunking;

pub use file_watcher::FileWatcher;
pub use hybrid_search::{HybridSearch, HybridSearchConfig};
pub use smart_chunking::{SmartChunker, ChunkingStrategy, CodeChunk};

/// Vector store for semantic search using Qdrant
pub struct VectorStore {
    qdrant_url: String,
    collection_name: String,
    qdrant_client: Client,
    embedding_client: Client,
    embedding_url: Option<String>,
    gemini_api_key: Option<String>,
}

impl VectorStore {
    /// Create a new vector store
    pub async fn new(url: &str, collection_name: &str) -> Result<Self> {
        let store = Self {
            qdrant_url: url.trim_end_matches('/').to_string(),
            collection_name: collection_name.to_string(),
            qdrant_client: Client::new(),
            embedding_client: Client::new(),
            embedding_url: std::env::var("EMBEDDING_URL").ok(),
            gemini_api_key: std::env::var("GEMINI_API_KEY").ok(),
        };

        store.ensure_collection().await?;
        Ok(store)
    }

    /// Ensure the collection exists
    async fn ensure_collection(&self) -> Result<()> {
        let collection_url = format!("{}/collections/{}", self.qdrant_url, self.collection_name);

        let resp = self.qdrant_client.get(&collection_url).send().await?;
        if resp.status() == StatusCode::NOT_FOUND {
            info!("Creating Qdrant collection: {}", self.collection_name);
                // Use 768 dimensions for Gemini text-embedding-004, fallback to 384 for other services
            // Note: Collection size is fixed, so we use 768 if Gemini is available, otherwise 384
            let embedding_size = if self.gemini_api_key.is_some() { 768 } else { 384 };
            let body = serde_json::json!({
                "vectors": {
                    "size": embedding_size,
                    "distance": "Cosine"
                }
            });

            let create_resp = self
                .qdrant_client
                .put(&collection_url)
                .json(&body)
                .send()
                .await?;

            if !create_resp.status().is_success() {
                let text = create_resp.text().await.unwrap_or_default();
                bail!("Failed to create collection: {}", text);
            }

            info!("Collection created successfully");
        } else if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("Failed to check collection: {}", text);
        } else {
            debug!("Collection {} already exists", self.collection_name);
        }

        Ok(())
    }

    /// Generate embedding for text using Gemini API, custom service, or fallback
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Try Gemini embeddings API first
        if let Some(api_key) = &self.gemini_api_key {
            match self.generate_gemini_embedding(text, api_key).await {
                Ok(embedding) => {
                    debug!("Generated Gemini embedding (size: {})", embedding.len());
                    return Ok(embedding);
                }
                Err(e) => {
                    warn!("Gemini embedding failed: {}, trying fallback", e);
                }
            }
        }

        // Try custom embedding service
        if let Some(url) = &self.embedding_url {
            let response = self
                .embedding_client
                .post(url)
                .json(&serde_json::json!({ "texts": [text] }))
                .send()
                .await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let json: Value = resp.json().await?;
                    if let Some(embeddings) = json.get("embeddings").and_then(|e| e.as_array()) {
                        if let Some(embedding) = embeddings.get(0).and_then(|e| e.as_array()) {
                            return Ok(embedding
                                .iter()
                                .filter_map(|v| v.as_f64().map(|f| f as f32))
                                .collect());
                        }
                    }
                }
                Ok(resp) => {
                    warn!(
                        "Embedding service responded with {}. Falling back to hash embedding",
                        resp.status()
                    );
                }
                Err(err) => {
                    warn!(
                        "Failed to call embedding service: {}. Falling back to hash embedding",
                        err
                    );
                }
            }
        }

        // Fallback: Use simple hash-based embedding
        warn!("Using hash-based embedding (not semantic)");
        Ok(self.simple_embedding(text))
    }

    /// Generate embedding using Gemini API
    async fn generate_gemini_embedding(&self, text: &str, api_key: &str) -> Result<Vec<f32>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={}",
            api_key
        );

        let payload = serde_json::json!({
            "model": "models/text-embedding-004",
            "content": {
                "parts": [{
                    "text": text
                }]
            }
        });

        let response = self
            .embedding_client
            .post(&url)
            .json(&payload)
            .send()
            .await?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            bail!("Gemini API error: {}", text);
        }

        let json: Value = response.json().await?;
        if let Some(embedding) = json
            .get("embedding")
            .and_then(|e| e.get("values"))
            .and_then(|v| v.as_array())
        {
            let vec: Result<Vec<f32>, _> = embedding
                .iter()
                .map(|v| {
                    v.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| anyhow::anyhow!("Invalid embedding value"))
                })
                .collect();
            return vec;
        }

        bail!("Invalid response format from Gemini API")
    }

    /// Simple hash-based embedding (fallback - not semantic but works for testing)
    fn simple_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Match collection size (768 for Gemini, 384 otherwise)
        let size = if self.gemini_api_key.is_some() { 768 } else { 384 };
        let mut embedding = vec![0.0f32; size];
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate().take(384) {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            embedding[i] = ((hash % 10000) as f32 / 10000.0) - 0.5;
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for e in &mut embedding {
                *e /= norm;
            }
        }

        embedding
    }

    /// Insert a symbol with its embedding
    pub async fn insert_symbol(&self, symbol: &SymbolVector) -> Result<()> {
        let text = format!(
            "{} {} {}",
            symbol.name,
            symbol.kind,
            symbol.content.chars().take(500).collect::<String>()
        );

        let embedding = self.generate_embedding(&text).await?;

        let payload = serde_json::json!({
            "name": symbol.name,
            "kind": symbol.kind,
            "content": symbol.content,
            "file_path": symbol.file_path,
            "metadata": symbol.metadata,
            "original_id": symbol.id,
        });

        let point_id =
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, symbol.id.as_bytes()).to_string();

        let body = serde_json::json!({
            "points": [{
                "id": point_id,
                "vector": embedding,
                "payload": payload
            }]
        });

        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.qdrant_url, self.collection_name
        );

        let resp = self.qdrant_client.put(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("Failed to upsert point: {}", text);
        }

        Ok(())
    }

    /// Search for similar symbols
    pub async fn search_similar(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SymbolSearchResult>> {
        let query_embedding = self.generate_embedding(query).await?;
        self.search_with_embedding(query_embedding, limit).await
    }

    /// Search by embedding vector
    pub async fn search_by_embedding(
        &self,
        embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<SymbolSearchResult>> {
        self.search_with_embedding(embedding, limit).await
    }

    async fn search_with_embedding(
        &self,
        embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<SymbolSearchResult>> {
        let url = format!(
            "{}/collections/{}/points/search",
            self.qdrant_url, self.collection_name
        );

        let body = serde_json::json!({
            "vector": embedding,
            "limit": limit,
            "with_payload": true
        });

        let resp = self.qdrant_client.post(&url).json(&body).send().await?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("Failed to search points: {}", text);
        }

        let json: Value = resp.json().await?;
        let mut results = Vec::new();

        if let Some(items) = json.get("result").and_then(|v| v.as_array()) {
            for item in items {
                let score = item.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
                let payload = item.get("payload").and_then(|p| p.as_object());
                if let Some(payload) = payload {
                    let symbol = SymbolVector {
                        id: payload
                            .get("original_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        name: payload
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        kind: payload
                            .get("kind")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        content: payload
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        file_path: payload
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        metadata: payload
                            .get("metadata")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    };

                    results.push(SymbolSearchResult { symbol, score });
                }
            }
        }

        Ok(results)
    }
}

/// Symbol representation for vector storage
#[derive(Debug, Clone, Serialize)]
pub struct SymbolVector {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub content: String,
    pub file_path: String,
    pub metadata: String,
}

/// Search result with similarity score
#[derive(Debug, Clone)]
pub struct SymbolSearchResult {
    pub symbol: SymbolVector,
    pub score: f32,
}
