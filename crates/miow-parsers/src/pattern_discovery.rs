use crate::types::*;
use anyhow::{Context, Result};
use miow_llm::LLMProvider;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// A discovered code pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPattern {
    /// Pattern name (e.g., "Repository Pattern", "Custom Hook Pattern")
    pub name: String,
    
    /// Description of the pattern
    pub description: String,
    
    /// Tree-sitter query to detect this pattern (if applicable)
    pub tree_sitter_query: Option<String>,
    
    /// Extraction logic (pseudo-code or description)
    pub extraction_logic: String,
    
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    
    /// Example occurrences in the codebase
    pub examples: Vec<PatternExample>,
    
    /// Metadata extraction rules
    pub metadata_rules: Vec<MetadataRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExample {
    pub file_path: String,
    pub code_snippet: String,
    pub line_range: (usize, usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataRule {
    pub field_name: String,
    pub extraction_method: String,
    pub description: String,
}

/// Pattern discovery system that learns new patterns from codebase
pub struct PatternDiscovery {
    llm: Arc<dyn LLMProvider>,
    discovered_patterns: Vec<DiscoveredPattern>,
    min_confidence: f32,
}

impl PatternDiscovery {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self {
            llm,
            discovered_patterns: Vec::new(),
            min_confidence: 0.7,
        }
    }
    
    /// Discover patterns in a codebase by sampling files
    pub async fn discover_patterns(
        &mut self,
        codebase_path: &Path,
        sample_size: usize,
    ) -> Result<Vec<DiscoveredPattern>> {
        tracing::info!("Discovering patterns in {:?}", codebase_path);
        
        // 1. Sample files from codebase
        let samples = self.sample_files(codebase_path, sample_size)?;
        
        if samples.is_empty() {
            tracing::warn!("No files sampled from codebase");
            return Ok(vec![]);
        }
        
        // 2. Use LLM to identify recurring patterns
        let patterns = self.identify_patterns(&samples).await?;
        
        // 3. Validate patterns against more files
        let validated = self.validate_patterns(&patterns, codebase_path).await?;
        
        // 4. Store discovered patterns
        self.discovered_patterns.extend(validated.clone());
        
        tracing::info!("Discovered {} patterns", validated.len());
        
        Ok(validated)
    }
    
    /// Sample files from codebase
    fn sample_files(&self, codebase_path: &Path, sample_size: usize) -> Result<Vec<FileSample>> {
        use std::fs;
        use walkdir::WalkDir;
        
        let mut samples = Vec::new();
        let mut file_count = 0;
        
        for entry in WalkDir::new(codebase_path)
            .max_depth(5)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            
            // Skip directories and non-code files
            if !path.is_file() {
                continue;
            }
            
            let extension = path.extension().and_then(|s| s.to_str());
            if !matches!(extension, Some("rs") | Some("py") | Some("ts") | Some("tsx") | Some("js") | Some("jsx")) {
                continue;
            }
            
            // Read file content
            if let Ok(content) = fs::read_to_string(path) {
                samples.push(FileSample {
                    path: path.to_string_lossy().to_string(),
                    content,
                    language: extension.unwrap_or("unknown").to_string(),
                });
                
                file_count += 1;
                if file_count >= sample_size {
                    break;
                }
            }
        }
        
        Ok(samples)
    }
    
    /// Identify patterns using LLM
    async fn identify_patterns(&self, samples: &[FileSample]) -> Result<Vec<DiscoveredPattern>> {
        let prompt = self.build_pattern_identification_prompt(samples);
        
        let response = self.llm.generate(&prompt).await?;
        
        self.parse_patterns_response(&response.content)
    }
    
    fn build_pattern_identification_prompt(&self, samples: &[FileSample]) -> String {
        let mut samples_text = String::new();
        
        for (i, sample) in samples.iter().enumerate().take(10) {
            samples_text.push_str(&format!(
                "\n--- File {}: {} ---\n{}\n",
                i + 1,
                sample.path,
                &sample.content[..sample.content.len().min(1000)]
            ));
        }
        
        format!(
            r#"Analyze these code samples and identify recurring patterns, conventions, or architectural decisions:

{}

Identify patterns such as:
1. Design patterns (Repository, Factory, Singleton, etc.)
2. Architectural patterns (MVC, MVVM, Clean Architecture, etc.)
3. Code organization patterns (file structure, naming conventions)
4. Custom patterns specific to this codebase
5. Framework-specific patterns (React hooks, Vue composables, etc.)

For each pattern, provide:
- Name: Clear, descriptive name
- Description: What the pattern does and why it's used
- Confidence: 0.0-1.0 (how confident you are this is a real pattern)
- Examples: Where you saw this pattern
- Extraction logic: How to detect and extract this pattern

Respond with JSON array:
[
  {{
    "name": "Pattern Name",
    "description": "Pattern description",
    "confidence": 0.0-1.0,
    "examples": [
      {{
        "file_path": "path/to/file",
        "code_snippet": "relevant code",
        "line_range": [start, end]
      }}
    ],
    "extraction_logic": "How to detect this pattern",
    "metadata_rules": [
      {{
        "field_name": "field name",
        "extraction_method": "how to extract",
        "description": "what this field represents"
      }}
    ]
  }}
]

Focus on patterns that appear in multiple files. Ignore one-off code."#,
            samples_text
        )
    }
    
    fn parse_patterns_response(&self, response: &str) -> Result<Vec<DiscoveredPattern>> {
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
        
        let patterns: Vec<DiscoveredPattern> = serde_json::from_str(json_str)
            .context("Failed to parse patterns response")?;
        
        // Filter by confidence
        Ok(patterns
            .into_iter()
            .filter(|p| p.confidence >= self.min_confidence)
            .collect())
    }
    
    /// Validate patterns by checking if they occur in more files
    async fn validate_patterns(
        &self,
        patterns: &[DiscoveredPattern],
        codebase_path: &Path,
    ) -> Result<Vec<DiscoveredPattern>> {
        let mut validated = Vec::new();
        
        // Sample more files for validation
        let validation_samples = self.sample_files(codebase_path, 50)?;
        
        for pattern in patterns {
            let occurrences = self.count_pattern_occurrences(pattern, &validation_samples);
            
            // Pattern is valid if it appears in at least 3 files
            if occurrences >= 3 {
                validated.push(pattern.clone());
            } else {
                tracing::debug!(
                    "Pattern '{}' only found in {} files, skipping",
                    pattern.name,
                    occurrences
                );
            }
        }
        
        Ok(validated)
    }
    
    fn count_pattern_occurrences(
        &self,
        pattern: &DiscoveredPattern,
        samples: &[FileSample],
    ) -> usize {
        let mut count = 0;
        
        // Simple heuristic: check if pattern name or keywords appear in files
        let keywords: Vec<&str> = pattern.name.split_whitespace().collect();
        
        for sample in samples {
            let content_lower = sample.content.to_lowercase();
            let matches = keywords
                .iter()
                .filter(|kw| content_lower.contains(&kw.to_lowercase()))
                .count();
            
            // If more than half the keywords match, count it
            if matches > keywords.len() / 2 {
                count += 1;
            }
        }
        
        count
    }
    
    /// Get all discovered patterns
    pub fn get_patterns(&self) -> &[DiscoveredPattern] {
        &self.discovered_patterns
    }
    
    /// Get patterns by confidence threshold
    pub fn get_patterns_by_confidence(&self, min_confidence: f32) -> Vec<&DiscoveredPattern> {
        self.discovered_patterns
            .iter()
            .filter(|p| p.confidence >= min_confidence)
            .collect()
    }
    
    /// Clear discovered patterns
    pub fn clear_patterns(&mut self) {
        self.discovered_patterns.clear();
    }
    
    /// Set minimum confidence threshold
    pub fn set_min_confidence(&mut self, confidence: f32) {
        self.min_confidence = confidence.clamp(0.0, 1.0);
    }
    
    /// Export patterns to JSON
    pub fn export_patterns(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.discovered_patterns)
            .context("Failed to serialize patterns")
    }
    
    /// Import patterns from JSON
    pub fn import_patterns(&mut self, json: &str) -> Result<()> {
        let patterns: Vec<DiscoveredPattern> = serde_json::from_str(json)
            .context("Failed to deserialize patterns")?;
        
        self.discovered_patterns.extend(patterns);
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct FileSample {
    path: String,
    content: String,
    language: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pattern_confidence_filter() {
        let pattern1 = DiscoveredPattern {
            name: "High Confidence".to_string(),
            description: "Test".to_string(),
            tree_sitter_query: None,
            extraction_logic: "Test".to_string(),
            confidence: 0.9,
            examples: vec![],
            metadata_rules: vec![],
        };
        
        let pattern2 = DiscoveredPattern {
            name: "Low Confidence".to_string(),
            description: "Test".to_string(),
            tree_sitter_query: None,
            extraction_logic: "Test".to_string(),
            confidence: 0.5,
            examples: vec![],
            metadata_rules: vec![],
        };
        
        let patterns = vec![pattern1.clone(), pattern2];
        let filtered: Vec<_> = patterns
            .iter()
            .filter(|p| p.confidence >= 0.7)
            .collect();
        
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "High Confidence");
    }
}
