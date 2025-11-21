use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Dynamic language configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub name: String,
    pub extensions: Vec<String>,
    pub parser_type: ParserType,
    pub framework_indicators: Vec<FrameworkIndicator>,
    pub package_managers: Vec<PackageManager>,
    pub best_practices: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParserType {
    TreeSitter { grammar: String },
    Regex { patterns: Vec<String> },
    Custom { handler: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkIndicator {
    pub name: String,
    pub files: Vec<String>,
    pub dependencies: Vec<String>,
    pub directory_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManager {
    pub name: String,
    pub manifest_file: String,
    pub lock_file: Option<String>,
}

/// Dynamic language registry that can be extended via configuration
pub struct LanguageRegistry {
    languages: HashMap<String, LanguageConfig>,
}

impl LanguageRegistry {
    /// Create a new registry with default languages
    pub fn new() -> Self {
        let mut registry = Self {
            languages: HashMap::new(),
        };
        
        // Register default languages
        registry.register_defaults();
        
        registry
    }
    
    /// Register default languages
    fn register_defaults(&mut self) {
        // TypeScript/JavaScript
        self.register(LanguageConfig {
            name: "typescript".to_string(),
            extensions: vec!["ts".to_string(), "tsx".to_string(), "js".to_string(), "jsx".to_string()],
            parser_type: ParserType::TreeSitter {
                grammar: "tree-sitter-typescript".to_string(),
            },
            framework_indicators: vec![
                FrameworkIndicator {
                    name: "Next.js".to_string(),
                    files: vec!["next.config.js".to_string(), "next.config.mjs".to_string()],
                    dependencies: vec!["next".to_string()],
                    directory_patterns: vec!["app/".to_string(), "pages/".to_string()],
                },
                FrameworkIndicator {
                    name: "React".to_string(),
                    files: vec![],
                    dependencies: vec!["react".to_string()],
                    directory_patterns: vec!["components/".to_string()],
                },
                FrameworkIndicator {
                    name: "Vite".to_string(),
                    files: vec!["vite.config.ts".to_string(), "vite.config.js".to_string()],
                    dependencies: vec!["vite".to_string()],
                    directory_patterns: vec![],
                },
            ],
            package_managers: vec![
                PackageManager {
                    name: "npm".to_string(),
                    manifest_file: "package.json".to_string(),
                    lock_file: Some("package-lock.json".to_string()),
                },
                PackageManager {
                    name: "yarn".to_string(),
                    manifest_file: "package.json".to_string(),
                    lock_file: Some("yarn.lock".to_string()),
                },
                PackageManager {
                    name: "pnpm".to_string(),
                    manifest_file: "package.json".to_string(),
                    lock_file: Some("pnpm-lock.yaml".to_string()),
                },
            ],
            best_practices: vec![
                "Use TypeScript strict mode".to_string(),
                "Prefer const over let".to_string(),
                "Use async/await over promises".to_string(),
                "Follow naming conventions (camelCase for variables, PascalCase for types)".to_string(),
            ],
        });
        
        // Rust
        self.register(LanguageConfig {
            name: "rust".to_string(),
            extensions: vec!["rs".to_string()],
            parser_type: ParserType::TreeSitter {
                grammar: "tree-sitter-rust".to_string(),
            },
            framework_indicators: vec![
                FrameworkIndicator {
                    name: "Axum".to_string(),
                    files: vec![],
                    dependencies: vec!["axum".to_string()],
                    directory_patterns: vec![],
                },
                FrameworkIndicator {
                    name: "Actix Web".to_string(),
                    files: vec![],
                    dependencies: vec!["actix-web".to_string()],
                    directory_patterns: vec![],
                },
                FrameworkIndicator {
                    name: "Rocket".to_string(),
                    files: vec![],
                    dependencies: vec!["rocket".to_string()],
                    directory_patterns: vec![],
                },
            ],
            package_managers: vec![PackageManager {
                name: "cargo".to_string(),
                manifest_file: "Cargo.toml".to_string(),
                lock_file: Some("Cargo.lock".to_string()),
            }],
            best_practices: vec![
                "Use ownership and borrowing correctly".to_string(),
                "Avoid unwrap() in production code".to_string(),
                "Use Result/Option for error handling".to_string(),
                "Follow naming conventions (snake_case for functions, PascalCase for types)".to_string(),
            ],
        });
        
        // Python
        self.register(LanguageConfig {
            name: "python".to_string(),
            extensions: vec!["py".to_string()],
            parser_type: ParserType::TreeSitter {
                grammar: "tree-sitter-python".to_string(),
            },
            framework_indicators: vec![
                FrameworkIndicator {
                    name: "FastAPI".to_string(),
                    files: vec![],
                    dependencies: vec!["fastapi".to_string()],
                    directory_patterns: vec![],
                },
                FrameworkIndicator {
                    name: "Django".to_string(),
                    files: vec!["manage.py".to_string()],
                    dependencies: vec!["django".to_string()],
                    directory_patterns: vec![],
                },
                FrameworkIndicator {
                    name: "Flask".to_string(),
                    files: vec![],
                    dependencies: vec!["flask".to_string()],
                    directory_patterns: vec![],
                },
            ],
            package_managers: vec![
                PackageManager {
                    name: "pip".to_string(),
                    manifest_file: "requirements.txt".to_string(),
                    lock_file: None,
                },
                PackageManager {
                    name: "poetry".to_string(),
                    manifest_file: "pyproject.toml".to_string(),
                    lock_file: Some("poetry.lock".to_string()),
                },
            ],
            best_practices: vec![
                "Follow PEP 8 style guide".to_string(),
                "Use type hints".to_string(),
                "Write docstrings for functions and classes".to_string(),
                "Prefer list comprehensions over loops".to_string(),
            ],
        });
    }
    
    /// Register a new language
    pub fn register(&mut self, config: LanguageConfig) {
        self.languages.insert(config.name.clone(), config);
    }
    
    /// Get language config by name
    pub fn get(&self, name: &str) -> Option<&LanguageConfig> {
        self.languages.get(name)
    }
    
    /// Detect language from file extension
    pub fn detect_from_extension(&self, extension: &str) -> Option<&LanguageConfig> {
        self.languages
            .values()
            .find(|lang| lang.extensions.contains(&extension.to_string()))
    }
    
    /// Detect language from project root
    pub fn detect_from_project(&self, project_root: &Path) -> Option<&LanguageConfig> {
        // Check for package manager files
        for lang in self.languages.values() {
            for pm in &lang.package_managers {
                if project_root.join(&pm.manifest_file).exists() {
                    return Some(lang);
                }
            }
        }
        
        None
    }
    
    /// Get all registered languages
    pub fn all_languages(&self) -> Vec<&LanguageConfig> {
        self.languages.values().collect()
    }
    
    /// Load additional languages from JSON file
    pub fn load_from_file(&mut self, path: &Path) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let configs: Vec<LanguageConfig> = serde_json::from_str(&content)?;
        
        for config in configs {
            self.register(config);
        }
        
        Ok(())
    }
    
    /// Export all languages to JSON
    pub fn export_to_file(&self, path: &Path) -> Result<()> {
        let configs: Vec<_> = self.languages.values().collect();
        let json = serde_json::to_string_pretty(&configs)?;
        fs::write(path, json)?;
        Ok(())
    }
    
    /// Get best practices for a language
    pub fn get_best_practices(&self, language: &str) -> Vec<String> {
        self.get(language)
            .map(|lang| lang.best_practices.clone())
            .unwrap_or_default()
    }
    
    /// Detect framework for a language in a project
    pub fn detect_framework(&self, language: &str, project_root: &Path) -> Option<String> {
        let lang_config = self.get(language)?;
        
        for indicator in &lang_config.framework_indicators {
            // Check files
            for file in &indicator.files {
                if project_root.join(file).exists() {
                    return Some(indicator.name.clone());
                }
            }
            
            // Check directories
            for dir in &indicator.directory_patterns {
                if project_root.join(dir).exists() {
                    return Some(indicator.name.clone());
                }
            }
            
            // Check dependencies (would need to parse manifest)
            // This is a simplified check
        }
        
        None
    }
}

impl Default for LanguageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_from_extension() {
        let registry = LanguageRegistry::new();
        
        let ts_lang = registry.detect_from_extension("ts");
        assert!(ts_lang.is_some());
        assert_eq!(ts_lang.unwrap().name, "typescript");
        
        let rs_lang = registry.detect_from_extension("rs");
        assert!(rs_lang.is_some());
        assert_eq!(rs_lang.unwrap().name, "rust");
    }
    
    #[test]
    fn test_get_best_practices() {
        let registry = LanguageRegistry::new();
        
        let practices = registry.get_best_practices("rust");
        assert!(!practices.is_empty());
        assert!(practices.iter().any(|p| p.contains("ownership")));
    }
}
