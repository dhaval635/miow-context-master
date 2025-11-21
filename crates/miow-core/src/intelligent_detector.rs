use anyhow::{Context, Result};
use miow_llm::LLMProvider;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use crate::ProjectSignature;

/// Intelligent project signature detector using LLM
pub struct IntelligentSignatureDetector {
    llm: Arc<dyn LLMProvider>,
    cache: HashMap<String, ProjectSignature>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LLMProjectAnalysis {
    language: String,
    framework: String,
    package_manager: String,
    ui_library: Option<String>,
    validation_library: Option<String>,
    auth_library: Option<String>,
    styling: Vec<String>,
    features: Vec<String>,
    confidence: f32,
}

impl IntelligentSignatureDetector {
    pub fn new(llm: Arc<dyn LLMProvider>) -> Self {
        Self {
            llm,
            cache: HashMap::new(),
        }
    }
    
    /// Detect project signature using LLM-powered analysis
    pub async fn detect(&mut self, project_root: &Path) -> Result<ProjectSignature> {
        // Check cache
        let cache_key = project_root.to_string_lossy().to_string();
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }
        
        // 1. Gather project structure information
        let structure = self.analyze_structure(project_root)?;
        
        // 2. Use LLM to identify patterns
        let analysis = self.llm_analyze(&structure).await?;
        
        // 3. Convert to ProjectSignature
        let signature = self.convert_to_signature(analysis, project_root)?;
        
        // Cache result
        self.cache.insert(cache_key, signature.clone());
        
        Ok(signature)
    }
    
    fn analyze_structure(&self, project_root: &Path) -> Result<String> {
        let mut structure = String::new();
        
        // List important files and directories
        structure.push_str("Project Structure:\n");
        
        // Root level files
        structure.push_str("\nRoot Files:\n");
        if let Ok(entries) = fs::read_dir(project_root) {
            for entry in entries.flatten() {
                if entry.path().is_file() {
                    if let Some(name) = entry.file_name().to_str() {
                        // Only include important config files
                        if Self::is_important_file(name) {
                            structure.push_str(&format!("  - {}\n", name));
                            
                            // Include content of small config files
                            if Self::should_include_content(name) {
                                if let Ok(content) = fs::read_to_string(entry.path()) {
                                    let preview = if content.len() > 500 {
                                        &content[..500]
                                    } else {
                                        &content
                                    };
                                    structure.push_str(&format!("    Content preview:\n{}\n", preview));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Important directories
        structure.push_str("\nDirectories:\n");
        let important_dirs = ["src", "app", "pages", "components", "lib", "utils", "crates", "tests"];
        for dir in important_dirs {
            let dir_path = project_root.join(dir);
            if dir_path.exists() {
                structure.push_str(&format!("  - {}/\n", dir));
                
                // List subdirectories
                if let Ok(entries) = fs::read_dir(&dir_path) {
                    let subdirs: Vec<_> = entries
                        .flatten()
                        .filter(|e| e.path().is_dir())
                        .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                        .take(10)
                        .collect();
                    
                    if !subdirs.is_empty() {
                        structure.push_str(&format!("    Subdirs: {}\n", subdirs.join(", ")));
                    }
                }
            }
        }
        
        // File type counts
        structure.push_str("\nFile Types:\n");
        let file_counts = self.count_file_types(project_root)?;
        for (ext, count) in file_counts.iter().take(10) {
            structure.push_str(&format!("  {}: {} files\n", ext, count));
        }
        
        Ok(structure)
    }
    
    fn is_important_file(name: &str) -> bool {
        matches!(
            name,
            "package.json" | "Cargo.toml" | "pyproject.toml" | "requirements.txt" |
            "tsconfig.json" | "next.config.js" | "next.config.mjs" | "vite.config.ts" |
            "tailwind.config.js" | "tailwind.config.ts" | ".eslintrc" | ".eslintrc.json" |
            "go.mod" | "pom.xml" | "build.gradle" | "Gemfile"
        )
    }
    
    fn should_include_content(name: &str) -> bool {
        matches!(
            name,
            "package.json" | "Cargo.toml" | "pyproject.toml" | "tsconfig.json" |
            "next.config.js" | "vite.config.ts"
        )
    }
    
    fn count_file_types(&self, project_root: &Path) -> Result<HashMap<String, usize>> {
        use walkdir::WalkDir;
        
        let mut counts = HashMap::new();
        
        for entry in WalkDir::new(project_root)
            .max_depth(5)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().is_file() {
                if let Some(ext) = entry.path().extension() {
                    if let Some(ext_str) = ext.to_str() {
                        *counts.entry(ext_str.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
        
        // Sort by count
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        
        Ok(sorted.into_iter().collect())
    }
    
    async fn llm_analyze(&self, structure: &str) -> Result<LLMProjectAnalysis> {
        let prompt = format!(
            r#"Analyze this project structure and identify:

{}

Provide a JSON response with the following structure:
{{
  "language": "primary programming language (typescript/rust/python/go/java/etc)",
  "framework": "framework or library (Next.js/React/Vite/Axum/FastAPI/etc)",
  "package_manager": "package manager (npm/yarn/pnpm/cargo/pip/etc)",
  "ui_library": "UI library if detected (shadcn/ui/Radix UI/Material-UI/etc or null)",
  "validation_library": "validation library if detected (Zod/Yup/Joi/etc or null)",
  "auth_library": "auth library if detected (NextAuth.js/Auth0/Supabase/etc or null)",
  "styling": ["styling approaches (Tailwind CSS/CSS Modules/Styled Components/etc)"],
  "features": ["key features (TypeScript/Server-Side Rendering/App Router/etc)"],
  "confidence": 0.0-1.0
}}

Consider:
1. File extensions and counts
2. Configuration files
3. Directory structure
4. Dependencies in package.json/Cargo.toml/etc
5. Common patterns for each framework

Respond ONLY with valid JSON."#,
            structure
        );
        
        let response = self.llm.generate(&prompt).await?;
        
        // Parse JSON response
        let json_str = if let Some(start) = response.content.find('{') {
            if let Some(end) = response.content.rfind('}') {
                &response.content[start..=end]
            } else {
                &response.content
            }
        } else {
            &response.content
        };
        
        serde_json::from_str(json_str)
            .context("Failed to parse LLM project analysis")
    }
    
    fn convert_to_signature(
        &self,
        analysis: LLMProjectAnalysis,
        project_root: &Path,
    ) -> Result<ProjectSignature> {
        // Parse dependencies from actual files for accuracy
        let dependencies = self.parse_dependencies(project_root, &analysis.package_manager)?;
        
        Ok(ProjectSignature {
            language: analysis.language,
            framework: analysis.framework,
            package_manager: analysis.package_manager,
            ui_library: analysis.ui_library,
            validation_library: analysis.validation_library,
            auth_library: analysis.auth_library,
            styling: analysis.styling,
            dependencies: dependencies.0,
            dev_dependencies: dependencies.1,
            features: analysis.features,
        })
    }
    
    fn parse_dependencies(
        &self,
        project_root: &Path,
        package_manager: &str,
    ) -> Result<(HashMap<String, String>, HashMap<String, String>)> {
        let mut deps = HashMap::new();
        let mut dev_deps = HashMap::new();
        
        match package_manager {
            "npm" | "yarn" | "pnpm" => {
                if let Ok(content) = fs::read_to_string(project_root.join("package.json")) {
                    if let Ok(package_json) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(dependencies) = package_json["dependencies"].as_object() {
                            for (name, version) in dependencies {
                                deps.insert(
                                    name.clone(),
                                    version.as_str().unwrap_or("").to_string(),
                                );
                            }
                        }
                        if let Some(dev_dependencies) = package_json["devDependencies"].as_object() {
                            for (name, version) in dev_dependencies {
                                dev_deps.insert(
                                    name.clone(),
                                    version.as_str().unwrap_or("").to_string(),
                                );
                            }
                        }
                    }
                }
            }
            "cargo" => {
                if let Ok(content) = fs::read_to_string(project_root.join("Cargo.toml")) {
                    // Simple parsing - could use toml crate
                    for line in content.lines() {
                        if line.contains(" = ") && !line.trim().starts_with('#') {
                            let parts: Vec<&str> = line.split(" = ").collect();
                            if parts.len() == 2 {
                                let name = parts[0].trim().trim_matches('"');
                                let version = parts[1].trim().trim_matches('"');
                                deps.insert(name.to_string(), version.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        
        Ok((deps, dev_deps))
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
    fn test_is_important_file() {
        assert!(IntelligentSignatureDetector::is_important_file("package.json"));
        assert!(IntelligentSignatureDetector::is_important_file("Cargo.toml"));
        assert!(!IntelligentSignatureDetector::is_important_file("README.md"));
    }
}
