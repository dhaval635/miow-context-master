use crate::types::*;
use anyhow::{Context, Result};
use miow_llm::LLMProvider;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Semantic information about a code symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticInfo {
    /// What does this code do? (high-level purpose)
    pub purpose: String,
    
    /// Complexity score (0.0 = simple, 1.0 = very complex)
    pub complexity: f32,
    
    /// What does this code depend on?
    pub dependencies: Vec<String>,
    
    /// Design patterns used (e.g., "Singleton", "Factory", "Observer")
    pub patterns: Vec<String>,
    
    /// Best practices followed or violated
    pub best_practices: Vec<BestPractice>,
    
    /// Suggested improvements
    pub improvements: Vec<String>,
    
    /// Similar code in the codebase
    pub similar_to: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestPractice {
    pub name: String,
    pub status: ComplianceStatus,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Followed,
    Violated,
    PartiallyFollowed,
}

/// Semantic analyzer that uses LLM to understand code
pub struct SemanticAnalyzer {
    llm: Arc<dyn LLMProvider>,
    cache: Arc<RwLock<HashMap<String, SemanticInfo>>>,
    language_context: HashMap<String, String>,
}

impl SemanticAnalyzer {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        let mut language_context = HashMap::new();
        
        // Add language-specific context
        language_context.insert(
            "rust".to_string(),
            "Rust best practices: ownership, borrowing, no unwrap() in production, \
             use Result/Option, avoid unsafe unless necessary, prefer iterators, \
             use strong typing, follow naming conventions (snake_case for functions/variables, \
             PascalCase for types)".to_string()
        );
        
        language_context.insert(
            "python".to_string(),
            "Python best practices: PEP 8 style, type hints, docstrings, \
             list comprehensions over loops, context managers, avoid mutable defaults, \
             use pathlib over os.path, prefer f-strings, follow naming conventions \
             (snake_case for functions/variables, PascalCase for classes)".to_string()
        );
        
        language_context.insert(
            "typescript".to_string(),
            "TypeScript best practices: strict mode, explicit types, avoid any, \
             use interfaces for objects, prefer const over let, use async/await, \
             follow naming conventions (camelCase for functions/variables, PascalCase for types), \
             use functional patterns, avoid mutation".to_string()
        );
        
        Self {
            llm,
            cache: Arc::new(RwLock::new(HashMap::new())),
            language_context,
        }
    }
    
    /// Analyze a symbol semantically using LLM
    pub async fn analyze_symbol(
        &self,
        symbol: &Symbol,
        context: &str,
        language: &str,
    ) -> Result<SemanticInfo> {
        // Check cache first
        let cache_key = format!("{}::{}", symbol.name, symbol.content.len());
        
        if let Ok(cache) = self.cache.read() {
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        
        // Get language-specific context
        let lang_context = self.language_context
            .get(language)
            .map(|s| s.as_str())
            .unwrap_or("");
        
        // Build prompt for LLM
        let prompt = self.build_analysis_prompt(symbol, context, language, lang_context);
        
        // Call LLM
        let response = self.llm.generate(&prompt).await?;
        
        // Parse response
        let semantic_info = self.parse_llm_response(&response.content)?;
        
        // Cache result
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(cache_key, semantic_info.clone());
        }
        
        Ok(semantic_info)
    }
    
    /// Analyze multiple symbols in batch
    pub async fn analyze_batch(
        &self,
        symbols: &[Symbol],
        context: &str,
        language: &str,
    ) -> Result<Vec<SemanticInfo>> {
        let mut results = Vec::new();
        
        for symbol in symbols {
            match self.analyze_symbol(symbol, context, language).await {
                Ok(info) => results.push(info),
                Err(e) => {
                    tracing::warn!("Failed to analyze symbol {}: {}", symbol.name, e);
                    // Push default semantic info on error
                    results.push(SemanticInfo::default_for_symbol(symbol));
                }
            }
        }
        
        Ok(results)
    }
    
    fn build_analysis_prompt(
        &self,
        symbol: &Symbol,
        context: &str,
        language: &str,
        lang_context: &str,
    ) -> String {
        format!(
            r#"Analyze this {} code symbol semantically:

Symbol Name: {}
Symbol Type: {:?}
Code:
```{}
{}
```

Context: {}

Language-Specific Guidelines:
{}

Provide a JSON response with the following structure:
{{
  "purpose": "Brief description of what this code does",
  "complexity": 0.0-1.0 (0.0 = very simple, 1.0 = very complex),
  "dependencies": ["list", "of", "dependencies"],
  "patterns": ["design", "patterns", "used"],
  "best_practices": [
    {{
      "name": "practice name",
      "status": "Followed" | "Violated" | "PartiallyFollowed",
      "description": "explanation"
    }}
  ],
  "improvements": ["suggested", "improvements"],
  "similar_to": ["similar code patterns in codebase"]
}}

Consider:
1. Code complexity (cyclomatic complexity, nesting, length)
2. Design patterns (if any)
3. Best practices for {}
4. Potential issues or code smells
5. Opportunities for improvement
6. Reusability and maintainability

Respond ONLY with valid JSON."#,
            language,
            symbol.name,
            symbol.kind,
            language,
            symbol.content,
            context,
            lang_context,
            language
        )
    }
    
    fn parse_llm_response(&self, response: &str) -> Result<SemanticInfo> {
        // Try to extract JSON from response
        let json_str = if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                &response[start..=end]
            } else {
                response
            }
        } else {
            response
        };
        
        serde_json::from_str(json_str)
            .context("Failed to parse LLM response as SemanticInfo")
    }
    
    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        if let Ok(cache) = self.cache.read() {
            (cache.len(), cache.capacity())
        } else {
            (0, 0)
        }
    }
}

impl SemanticInfo {
    /// Create default semantic info for a symbol (when LLM analysis fails)
    pub fn default_for_symbol(symbol: &Symbol) -> Self {
        let complexity = Self::estimate_complexity(&symbol.content);
        
        Self {
            purpose: format!("A {} named {}", Self::kind_to_string(&symbol.kind), symbol.name),
            complexity,
            dependencies: symbol.references.clone(),
            patterns: vec![],
            best_practices: vec![],
            improvements: vec![],
            similar_to: vec![],
        }
    }
    
    fn kind_to_string(kind: &SymbolType) -> &'static str {
        match kind {
            SymbolType::Function => "function",
            SymbolType::Class => "class",
            SymbolType::Method => "method",
            SymbolType::Interface => "interface",
            SymbolType::Component => "component",
            SymbolType::Struct => "struct",
            SymbolType::Enum => "enum",
            _ => "symbol",
        }
    }
    
    fn estimate_complexity(code: &str) -> f32 {
        let lines = code.lines().count();
        let nesting_level = Self::max_nesting_level(code);
        let branches = code.matches("if ").count() 
            + code.matches("match ").count()
            + code.matches("switch ").count()
            + code.matches("case ").count();
        
        // Simple heuristic
        let line_complexity = (lines as f32 / 100.0).min(0.4);
        let nesting_complexity = (nesting_level as f32 / 10.0).min(0.3);
        let branch_complexity = (branches as f32 / 10.0).min(0.3);
        
        (line_complexity + nesting_complexity + branch_complexity).min(1.0)
    }
    
    fn max_nesting_level(code: &str) -> usize {
        let mut max_level = 0;
        let mut current_level: usize = 0;
        
        for ch in code.chars() {
            match ch {
                '{' | '(' | '[' => {
                    current_level += 1;
                    max_level = max_level.max(current_level);
                }
                '}' | ')' | ']' => {
                    current_level = current_level.saturating_sub(1);
                }
                _ => {}
            }
        }
        
        max_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complexity_estimation() {
        let simple_code = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let complexity = SemanticInfo::estimate_complexity(simple_code);
        assert!(complexity < 0.2);
        
        let complex_code = r#"
            fn complex_function() {
                if condition {
                    match value {
                        Some(x) => {
                            if x > 0 {
                                // nested logic
                            }
                        }
                        None => {}
                    }
                }
            }
        "#;
        let complexity = SemanticInfo::estimate_complexity(complex_code);
        assert!(complexity > 0.3);
    }
    
    #[test]
    fn test_nesting_level() {
        let code = "{ { { } } }";
        assert_eq!(SemanticInfo::max_nesting_level(code), 3);
        
        let code2 = "fn test() { if x { match y { A => { } } } }";
        assert!(SemanticInfo::max_nesting_level(code2) > 2);
    }
}
