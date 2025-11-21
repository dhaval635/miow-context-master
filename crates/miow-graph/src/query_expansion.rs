use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use super::relationship_inference::{LLMProvider, LLMResponse};

/// Query expansion for better search coverage
pub struct QueryExpander {
    llm: Arc<dyn LLMProvider>,
    cache: HashMap<String, ExpandedQuery>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedQuery {
    pub original: String,
    pub synonyms: Vec<String>,
    pub related_terms: Vec<String>,
    pub abbreviations: Vec<String>,
    pub expansions: Vec<String>,
}

impl QueryExpander {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self {
            llm,
            cache: HashMap::new(),
        }
    }
    
    /// Expand query with related programming terms
    pub async fn expand(&mut self, query: &str) -> Result<ExpandedQuery> {
        // Check cache
        if let Some(cached) = self.cache.get(query) {
            return Ok(cached.clone());
        }
        
        // Use LLM to expand
        let prompt = self.build_expansion_prompt(query);
        let response = self.llm.generate(&prompt).await?;
        
        // Parse expansion
        let expanded = self.parse_expansion(&response.content, query)?;
        
        // Cache result
        self.cache.insert(query.to_string(), expanded.clone());
        
        Ok(expanded)
    }
    
    fn build_expansion_prompt(&self, query: &str) -> String {
        format!(
            r#"Expand this programming search query with related terms:

Query: "{}"

Provide:
1. Synonyms (alternative names for the same concept)
2. Related concepts (closely related programming terms)
3. Abbreviations (if query is expanded form)
4. Expansions (if query is abbreviated)

Examples:
- Query: "auth" → Synonyms: ["authentication", "authorization"], Abbreviations: ["auth"], Expansions: ["authentication", "authorization"]
- Query: "db" → Synonyms: ["database", "datastore"], Abbreviations: ["db"], Expansions: ["database"]
- Query: "component" → Synonyms: ["widget", "element"], Related: ["view", "template", "ui"]

Respond with JSON:
{{
  "synonyms": ["term1", "term2"],
  "related_terms": ["term3", "term4"],
  "abbreviations": ["abbr1"],
  "expansions": ["expansion1", "expansion2"]
}}

Keep it programming-focused and relevant.
Respond ONLY with valid JSON."#,
            query
        )
    }
    
    fn parse_expansion(&self, response: &str, original: &str) -> Result<ExpandedQuery> {
        // Extract JSON from response
        let json_str = if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                &response[start..=end]
            } else {
                response
            }
        } else {
            response
        };
        
        #[derive(Deserialize)]
        struct ExpansionResponse {
            synonyms: Vec<String>,
            related_terms: Vec<String>,
            abbreviations: Vec<String>,
            expansions: Vec<String>,
        }
        
        let parsed: ExpansionResponse = serde_json::from_str(json_str)
            .unwrap_or(ExpansionResponse {
                synonyms: vec![],
                related_terms: vec![],
                abbreviations: vec![],
                expansions: vec![],
            });
        
        Ok(ExpandedQuery {
            original: original.to_string(),
            synonyms: parsed.synonyms,
            related_terms: parsed.related_terms,
            abbreviations: parsed.abbreviations,
            expansions: parsed.expansions,
        })
    }
    
    /// Get all expanded terms as a flat list
    pub fn get_all_terms(&self, expanded: &ExpandedQuery) -> Vec<String> {
        let mut terms = vec![expanded.original.clone()];
        terms.extend(expanded.synonyms.clone());
        terms.extend(expanded.related_terms.clone());
        terms.extend(expanded.abbreviations.clone());
        terms.extend(expanded.expansions.clone());
        
        // Deduplicate
        terms.sort();
        terms.dedup();
        
        terms
    }
    
    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.cache.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_all_terms() {
        let expander = QueryExpander {
            llm: Arc::new(crate::tests::MockLLM),
            cache: HashMap::new(),
        };
        
        let expanded = ExpandedQuery {
            original: "auth".to_string(),
            synonyms: vec!["authentication".to_string()],
            related_terms: vec!["login".to_string()],
            abbreviations: vec!["auth".to_string()],
            expansions: vec!["authentication".to_string()],
        };
        
        let terms = expander.get_all_terms(&expanded);
        assert!(terms.contains(&"auth".to_string()));
        assert!(terms.contains(&"authentication".to_string()));
        assert!(terms.contains(&"login".to_string()));
        
        // Check deduplication
        let auth_count = terms.iter().filter(|t| *t == "auth").count();
        assert_eq!(auth_count, 1);
    }
}
