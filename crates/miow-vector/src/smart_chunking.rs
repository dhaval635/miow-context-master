use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Smart chunking strategies for code
pub struct SmartChunker {
    strategy: ChunkingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed size chunks
    FixedSize { size: usize, overlap: usize },
    
    /// Semantic chunks (by function/class boundaries)
    Semantic,
    
    /// Structural chunks (by AST nodes)
    Structural,
    
    /// Hybrid (semantic + fixed size fallback)
    Hybrid { max_size: usize },
}

#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub chunk_type: ChunkType,
    pub context: Option<String>, // Surrounding context
}

#[derive(Debug, Clone)]
pub enum ChunkType {
    Function,
    Class,
    Module,
    Block,
    Mixed,
}

impl SmartChunker {
    pub fn new(strategy: ChunkingStrategy) -> Self {
        Self { strategy }
    }
    
    /// Chunk code based on strategy
    pub fn chunk(&self, code: &str, language: &str) -> Result<Vec<CodeChunk>> {
        match &self.strategy {
            ChunkingStrategy::FixedSize { size, overlap } => {
                self.chunk_fixed_size(code, *size, *overlap)
            }
            ChunkingStrategy::Semantic => {
                self.chunk_semantic(code, language)
            }
            ChunkingStrategy::Structural => {
                self.chunk_structural(code, language)
            }
            ChunkingStrategy::Hybrid { max_size } => {
                self.chunk_hybrid(code, language, *max_size)
            }
        }
    }
    
    fn chunk_fixed_size(&self, code: &str, size: usize, overlap: usize) -> Result<Vec<CodeChunk>> {
        let lines: Vec<&str> = code.lines().collect();
        let mut chunks = Vec::new();
        let mut start = 0;
        
        while start < lines.len() {
            let end = (start + size).min(lines.len());
            let chunk_lines = &lines[start..end];
            
            chunks.push(CodeChunk {
                content: chunk_lines.join("\n"),
                start_line: start + 1,
                end_line: end,
                chunk_type: ChunkType::Block,
                context: None,
            });
            
            start += size - overlap;
            if start >= lines.len() {
                break;
            }
        }
        
        Ok(chunks)
    }
    
    fn chunk_semantic(&self, code: &str, language: &str) -> Result<Vec<CodeChunk>> {
        // Semantic chunking based on language constructs
        match language {
            "rust" => self.chunk_rust_semantic(code),
            "python" => self.chunk_python_semantic(code),
            "typescript" | "javascript" => self.chunk_ts_semantic(code),
            _ => self.chunk_fixed_size(code, 50, 10),
        }
    }
    
    fn chunk_rust_semantic(&self, code: &str) -> Result<Vec<CodeChunk>> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        
        let mut current_chunk_start = 0;
        let mut current_chunk_type = ChunkType::Block;
        let mut brace_depth = 0;
        let mut in_function = false;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Detect function start
            if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ") || trimmed.starts_with("async fn ") {
                if in_function && brace_depth == 0 {
                    // Save previous chunk
                    chunks.push(CodeChunk {
                        content: lines[current_chunk_start..i].join("\n"),
                        start_line: current_chunk_start + 1,
                        end_line: i,
                        chunk_type: current_chunk_type.clone(),
                        context: None,
                    });
                    current_chunk_start = i;
                }
                in_function = true;
                current_chunk_type = ChunkType::Function;
            }
            
            // Track braces
            brace_depth += trimmed.matches('{').count() as i32;
            brace_depth -= trimmed.matches('}').count() as i32;
            
            // End of function
            if in_function && brace_depth == 0 && trimmed.contains('}') {
                chunks.push(CodeChunk {
                    content: lines[current_chunk_start..=i].join("\n"),
                    start_line: current_chunk_start + 1,
                    end_line: i + 1,
                    chunk_type: current_chunk_type.clone(),
                    context: None,
                });
                current_chunk_start = i + 1;
                in_function = false;
            }
        }
        
        // Add remaining lines
        if current_chunk_start < lines.len() {
            chunks.push(CodeChunk {
                content: lines[current_chunk_start..].join("\n"),
                start_line: current_chunk_start + 1,
                end_line: lines.len(),
                chunk_type: ChunkType::Block,
                context: None,
            });
        }
        
        Ok(chunks)
    }
    
    fn chunk_python_semantic(&self, code: &str) -> Result<Vec<CodeChunk>> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        
        let mut current_chunk_start = 0;
        let mut current_chunk_type = ChunkType::Block;
        let mut current_indent = 0;
        let mut in_definition = false;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Detect function/class start
            if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
                if in_definition {
                    // Save previous chunk
                    chunks.push(CodeChunk {
                        content: lines[current_chunk_start..i].join("\n"),
                        start_line: current_chunk_start + 1,
                        end_line: i,
                        chunk_type: current_chunk_type.clone(),
                        context: None,
                    });
                    current_chunk_start = i;
                }
                in_definition = true;
                current_chunk_type = ChunkType::Function;
                current_indent = line.len() - line.trim_start().len();
            } else if trimmed.starts_with("class ") {
                if in_definition {
                    chunks.push(CodeChunk {
                        content: lines[current_chunk_start..i].join("\n"),
                        start_line: current_chunk_start + 1,
                        end_line: i,
                        chunk_type: current_chunk_type.clone(),
                        context: None,
                    });
                    current_chunk_start = i;
                }
                in_definition = true;
                current_chunk_type = ChunkType::Class;
                current_indent = line.len() - line.trim_start().len();
            } else if in_definition && !line.trim().is_empty() {
                let line_indent = line.len() - line.trim_start().len();
                
                // End of definition (dedent)
                if line_indent <= current_indent && !trimmed.starts_with("@") {
                    chunks.push(CodeChunk {
                        content: lines[current_chunk_start..i].join("\n"),
                        start_line: current_chunk_start + 1,
                        end_line: i,
                        chunk_type: current_chunk_type.clone(),
                        context: None,
                    });
                    current_chunk_start = i;
                    in_definition = false;
                }
            }
        }
        
        // Add remaining lines
        if current_chunk_start < lines.len() {
            chunks.push(CodeChunk {
                content: lines[current_chunk_start..].join("\n"),
                start_line: current_chunk_start + 1,
                end_line: lines.len(),
                chunk_type: if in_definition { current_chunk_type } else { ChunkType::Block },
                context: None,
            });
        }
        
        Ok(chunks)
    }
    
    fn chunk_ts_semantic(&self, code: &str) -> Result<Vec<CodeChunk>> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = code.lines().collect();
        
        let mut current_chunk_start = 0;
        let mut current_chunk_type = ChunkType::Block;
        let mut brace_depth = 0;
        let mut in_definition = false;
        
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            
            // Detect function/class start
            if trimmed.starts_with("function ") 
                || trimmed.starts_with("const ") && trimmed.contains("=>")
                || trimmed.starts_with("export function ")
                || trimmed.starts_with("async function ") {
                if in_definition && brace_depth == 0 {
                    chunks.push(CodeChunk {
                        content: lines[current_chunk_start..i].join("\n"),
                        start_line: current_chunk_start + 1,
                        end_line: i,
                        chunk_type: current_chunk_type.clone(),
                        context: None,
                    });
                    current_chunk_start = i;
                }
                in_definition = true;
                current_chunk_type = ChunkType::Function;
            } else if trimmed.starts_with("class ") || trimmed.starts_with("export class ") {
                if in_definition && brace_depth == 0 {
                    chunks.push(CodeChunk {
                        content: lines[current_chunk_start..i].join("\n"),
                        start_line: current_chunk_start + 1,
                        end_line: i,
                        chunk_type: current_chunk_type.clone(),
                        context: None,
                    });
                    current_chunk_start = i;
                }
                in_definition = true;
                current_chunk_type = ChunkType::Class;
            }
            
            // Track braces
            brace_depth += trimmed.matches('{').count() as i32;
            brace_depth -= trimmed.matches('}').count() as i32;
            
            // End of definition
            if in_definition && brace_depth == 0 && trimmed.contains('}') {
                chunks.push(CodeChunk {
                    content: lines[current_chunk_start..=i].join("\n"),
                    start_line: current_chunk_start + 1,
                    end_line: i + 1,
                    chunk_type: current_chunk_type.clone(),
                    context: None,
                });
                current_chunk_start = i + 1;
                in_definition = false;
            }
        }
        
        // Add remaining lines
        if current_chunk_start < lines.len() {
            chunks.push(CodeChunk {
                content: lines[current_chunk_start..].join("\n"),
                start_line: current_chunk_start + 1,
                end_line: lines.len(),
                chunk_type: ChunkType::Block,
                context: None,
            });
        }
        
        Ok(chunks)
    }
    
    fn chunk_structural(&self, code: &str, _language: &str) -> Result<Vec<CodeChunk>> {
        // Structural chunking would use tree-sitter AST
        // For now, fallback to semantic
        self.chunk_semantic(code, _language)
    }
    
    fn chunk_hybrid(&self, code: &str, language: &str, max_size: usize) -> Result<Vec<CodeChunk>> {
        // Try semantic first
        let semantic_chunks = self.chunk_semantic(code, language)?;
        
        // Split large chunks
        let mut final_chunks = Vec::new();
        for chunk in semantic_chunks {
            let line_count = chunk.content.lines().count();
            
            if line_count <= max_size {
                final_chunks.push(chunk);
            } else {
                // Split large chunk
                let sub_chunks = self.chunk_fixed_size(&chunk.content, max_size, 10)?;
                final_chunks.extend(sub_chunks);
            }
        }
        
        Ok(final_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fixed_size_chunking() {
        let chunker = SmartChunker::new(ChunkingStrategy::FixedSize {
            size: 10,
            overlap: 2,
        });
        
        let code = (0..50).map(|i| format!("line {}", i)).collect::<Vec<_>>().join("\n");
        let chunks = chunker.chunk(&code, "rust").unwrap();
        
        assert!(!chunks.is_empty());
    }
    
    #[test]
    fn test_rust_semantic_chunking() {
        let chunker = SmartChunker::new(ChunkingStrategy::Semantic);
        
        let code = r#"
fn test1() {
    println!("test");
}

fn test2() {
    println!("test2");
}
"#;
        
        let chunks = chunker.chunk(code, "rust").unwrap();
        assert!(chunks.len() >= 2);
    }
}
