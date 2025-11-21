use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{SymbolSearchResult, VectorStore};

/// Hybrid search that combines multiple search strategies
pub struct HybridSearch {
    vector_store: VectorStore,
    keyword_index: KeywordIndex,
    recency_tracker: RecencyTracker,
    popularity_tracker: PopularityTracker,
}

/// Keyword index for exact/fuzzy matching
struct KeywordIndex {
    symbols: HashMap<String, Vec<SymbolEntry>>,
}

#[derive(Clone)]
struct SymbolEntry {
    id: String,
    name: String,
    kind: String,
    file_path: String,
    content: String,
}

/// Tracks recently accessed symbols
struct RecencyTracker {
    access_times: HashMap<String, i64>, // symbol_id -> timestamp
}

/// Tracks symbol popularity (access count)
struct PopularityTracker {
    access_counts: HashMap<String, usize>, // symbol_id -> count
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchConfig {
    /// Weight for vector similarity (0.0-1.0)
    pub vector_weight: f32,
    
    /// Weight for keyword matching (0.0-1.0)
    pub keyword_weight: f32,
    
    /// Weight for recency (0.0-1.0)
    pub recency_weight: f32,
    
    /// Weight for popularity (0.0-1.0)
    pub popularity_weight: f32,
    
    /// Minimum score threshold
    pub min_score: f32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.5,
            keyword_weight: 0.3,
            recency_weight: 0.1,
            popularity_weight: 0.1,
            min_score: 0.1,
        }
    }
}

impl HybridSearch {
    pub fn new(vector_store: VectorStore) -> Self {
        Self {
            vector_store,
            keyword_index: KeywordIndex::new(),
            recency_tracker: RecencyTracker::new(),
            popularity_tracker: PopularityTracker::new(),
        }
    }
    
    /// Perform hybrid search
    pub async fn search(
        &mut self,
        query: &str,
        limit: usize,
        config: &HybridSearchConfig,
    ) -> Result<Vec<SymbolSearchResult>> {
        // 1. Vector search
        let vector_results = self.vector_store.search_similar(query, limit * 2).await?;
        
        // 2. Keyword search
        let keyword_results = self.keyword_index.search(query, limit * 2);
        
        // 3. Combine and score
        let mut combined_scores: HashMap<String, f32> = HashMap::new();
        
        // Add vector scores
        for result in &vector_results {
            let score = result.score * config.vector_weight;
            *combined_scores.entry(result.symbol.id.clone()).or_insert(0.0) += score;
        }
        
        // Add keyword scores
        for (id, score) in keyword_results {
            let weighted_score = score * config.keyword_weight;
            *combined_scores.entry(id.clone()).or_insert(0.0) += weighted_score;
        }
        
        // Add recency scores
        for (id, score) in self.recency_tracker.get_scores() {
            let weighted_score = score * config.recency_weight;
            *combined_scores.entry(id.clone()).or_insert(0.0) += weighted_score;
        }
        
        // Add popularity scores
        for (id, score) in self.popularity_tracker.get_scores() {
            let weighted_score = score * config.popularity_weight;
            *combined_scores.entry(id.clone()).or_insert(0.0) += weighted_score;
        }
        
        // 4. Sort by combined score
        let mut scored_results: Vec<_> = vector_results
            .into_iter()
            .map(|mut result| {
                if let Some(&score) = combined_scores.get(&result.symbol.id) {
                    result.score = score;
                }
                result
            })
            .filter(|r| r.score >= config.min_score)
            .collect();
        
        scored_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored_results.truncate(limit);
        
        // 5. Track access for recency and popularity
        for result in &scored_results {
            self.recency_tracker.record_access(&result.symbol.id);
            self.popularity_tracker.record_access(&result.symbol.id);
        }
        
        Ok(scored_results)
    }
    
    /// Index a symbol for keyword search
    pub fn index_symbol(&mut self, id: String, name: String, kind: String, file_path: String, content: String) {
        self.keyword_index.add(SymbolEntry {
            id,
            name,
            kind,
            file_path,
            content,
        });
    }
}

impl KeywordIndex {
    fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }
    
    fn add(&mut self, entry: SymbolEntry) {
        // Index by name
        self.symbols
            .entry(entry.name.to_lowercase())
            .or_insert_with(Vec::new)
            .push(entry.clone());
        
        // Index by words in name (for partial matching)
        for word in entry.name.split('_').chain(entry.name.split("::")) {
            if !word.is_empty() {
                self.symbols
                    .entry(word.to_lowercase())
                    .or_insert_with(Vec::new)
                    .push(entry.clone());
            }
        }
    }
    
    fn search(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        let query_lower = query.to_lowercase();
        let mut scores: HashMap<String, f32> = HashMap::new();
        
        // Exact match
        if let Some(entries) = self.symbols.get(&query_lower) {
            for entry in entries {
                *scores.entry(entry.id.clone()).or_insert(0.0) += 1.0;
            }
        }
        
        // Partial match
        for word in query_lower.split_whitespace() {
            if let Some(entries) = self.symbols.get(word) {
                for entry in entries {
                    *scores.entry(entry.id.clone()).or_insert(0.0) += 0.5;
                }
            }
        }
        
        // Fuzzy match (contains)
        for (key, entries) in &self.symbols {
            if key.contains(&query_lower) || query_lower.contains(key) {
                for entry in entries {
                    *scores.entry(entry.id.clone()).or_insert(0.0) += 0.3;
                }
            }
        }
        
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);
        
        results
    }
}

impl RecencyTracker {
    fn new() -> Self {
        Self {
            access_times: HashMap::new(),
        }
    }
    
    fn record_access(&mut self, symbol_id: &str) {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        self.access_times.insert(symbol_id.to_string(), timestamp);
    }
    
    fn get_scores(&self) -> Vec<(String, f32)> {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        self.access_times
            .iter()
            .map(|(id, &timestamp)| {
                let age_seconds = (now - timestamp).max(1);
                // Decay score over time (1 hour = 3600 seconds)
                let score = 1.0 / (1.0 + (age_seconds as f32 / 3600.0));
                (id.clone(), score)
            })
            .collect()
    }
}

impl PopularityTracker {
    fn new() -> Self {
        Self {
            access_counts: HashMap::new(),
        }
    }
    
    fn record_access(&mut self, symbol_id: &str) {
        *self.access_counts.entry(symbol_id.to_string()).or_insert(0) += 1;
    }
    
    fn get_scores(&self) -> Vec<(String, f32)> {
        let max_count = self.access_counts.values().max().copied().unwrap_or(1);
        
        self.access_counts
            .iter()
            .map(|(id, &count)| {
                let score = count as f32 / max_count as f32;
                (id.clone(), score)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_keyword_index() {
        let mut index = KeywordIndex::new();
        
        index.add(SymbolEntry {
            id: "1".to_string(),
            name: "test_function".to_string(),
            kind: "function".to_string(),
            file_path: "test.rs".to_string(),
            content: "fn test_function() {}".to_string(),
        });
        
        let results = index.search("test", 10);
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_recency_tracker() {
        let mut tracker = RecencyTracker::new();
        
        tracker.record_access("symbol1");
        let scores = tracker.get_scores();
        
        assert_eq!(scores.len(), 1);
        assert!(scores[0].1 > 0.0);
    }
}
