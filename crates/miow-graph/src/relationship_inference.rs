use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Generic LLM provider trait to avoid circular dependencies
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn generate(&self, prompt: &str) -> Result<LLMResponse>;
}

#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub content: String,
}

/// Automatic relationship inference using LLM
pub struct RelationshipInferencer {
    llm: Arc<dyn LLMProvider>,
    cache: HashMap<String, Vec<InferredRelationship>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferredRelationship {
    pub from_symbol: String,
    pub to_symbol: String,
    pub relationship_type: RelationshipType,
    pub confidence: f32,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub enum RelationshipType {
    Uses,
    Implements,
    Extends,
    Calls,
    DependsOn,
    Similar,
    Opposite,
    Wraps,
    Configures,
}

impl RelationshipInferencer {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self {
            llm,
            cache: HashMap::new(),
        }
    }
    
    /// Infer relationships between a symbol and candidates
    pub async fn infer_relationships(
        &mut self,
        symbol_name: &str,
        symbol_content: &str,
        candidates: &[(String, String)], // (name, content)
    ) -> Result<Vec<InferredRelationship>> {
        // Check cache
        let cache_key = format!("{}:{}", symbol_name, candidates.len());
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }
        
        // Build prompt
        let prompt = self.build_inference_prompt(symbol_name, symbol_content, candidates);
        
        // Call LLM
        let response = self.llm.generate(&prompt).await?;
        
        // Parse relationships
        let relationships = self.parse_relationships(&response.content)?;
        
        // Cache result
        self.cache.insert(cache_key, relationships.clone());
        
        Ok(relationships)
    }
    
    fn build_inference_prompt(
        &self,
        symbol_name: &str,
        symbol_content: &str,
        candidates: &[(String, String)],
    ) -> String {
        let candidates_text = candidates
            .iter()
            .enumerate()
            .map(|(i, (name, content))| {
                format!(
                    "{}. {} (preview: {}...)",
                    i + 1,
                    name,
                    &content.chars().take(100).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        
        format!(
            r#"Analyze this code symbol and infer relationships with other symbols:

Target Symbol: {}
Content:
{}

Candidate Symbols:
{}

Identify relationships such as:
- Uses: Symbol A uses Symbol B
- Implements: Symbol A implements interface/trait B
- Extends: Symbol A extends/inherits from B
- Calls: Symbol A calls function/method B
- DependsOn: Symbol A depends on B
- Similar: Symbol A is similar to B (same purpose/pattern)
- Opposite: Symbol A is opposite of B (inverse operation)
- Wraps: Symbol A wraps/decorates B
- Configures: Symbol A configures B

Respond with JSON array:
[
  {{
    "from_symbol": "{}",
    "to_symbol": "candidate_name",
    "relationship_type": "Uses|Implements|Extends|Calls|DependsOn|Similar|Opposite|Wraps|Configures",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
  }}
]

Only include relationships with confidence >= 0.6.
Respond ONLY with valid JSON array."#,
            symbol_name,
            &symbol_content.chars().take(500).collect::<String>(),
            candidates_text,
            symbol_name
        )
    }
    
    fn parse_relationships(&self, response: &str) -> Result<Vec<InferredRelationship>> {
        // Extract JSON from response
        let json_str = if let Some(start) = response.find('[') {
            if let Some(end) = response.rfind(']') {
                &response[start..=end]
            } else {
                response
            }
        } else {
            response
        };
        
        let relationships: Vec<InferredRelationship> = serde_json::from_str(json_str)
            .unwrap_or_default();
        
        // Filter by confidence
        Ok(relationships
            .into_iter()
            .filter(|r| r.confidence >= 0.6)
            .collect())
    }
    
    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_relationship_type_serialization() {
        let rel = InferredRelationship {
            from_symbol: "A".to_string(),
            to_symbol: "B".to_string(),
            relationship_type: RelationshipType::Uses,
            confidence: 0.9,
            reasoning: "Test".to_string(),
        };
        
        let json = serde_json::to_string(&rel).unwrap();
        assert!(json.contains("Uses"));
    }
}
