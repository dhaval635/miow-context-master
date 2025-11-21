use anyhow::Result;
use serde::{Deserialize, Serialize};

pub mod python;
pub mod rust;
pub mod types;
pub mod typescript;
pub mod style_analyzer;
pub mod semantic;
pub mod pattern_discovery;

pub use python::PythonParser;
pub use rust::RustParser;
pub use types::*;
pub use typescript::TypeScriptParser;
pub use style_analyzer::{StyleAnalyzer, StyleAnalysis};
pub use semantic::{SemanticAnalyzer, SemanticInfo, BestPractice, ComplianceStatus};
pub use pattern_discovery::{PatternDiscovery, DiscoveredPattern};

/// Parse a TypeScript/TSX file and extract symbols
pub fn parse_typescript(content: &str, is_tsx: bool) -> Result<ParsedFile> {
    let parser = TypeScriptParser::new();
    parser.parse(content, is_tsx)
}

/// Parse a Rust file and extract symbols
pub fn parse_rust(content: &str) -> Result<ParsedFile> {
    let parser = RustParser::new();
    parser.parse(content)
}

/// Parse a Python file and extract symbols
pub fn parse_python(content: &str) -> Result<ParsedFile> {
    let parser = PythonParser::new();
    parser.parse(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_component() {
        let code = r#"
            export const Button = ({ children }) => {
                return <button>{children}</button>;
            };
        "#;

        let result = parse_typescript(code, true);
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert!(!parsed.symbols.is_empty());
        assert_eq!(parsed.symbols[0].name, "Button");
    }
}
