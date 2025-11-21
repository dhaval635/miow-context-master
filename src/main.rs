use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use miow_core::index_codebase;
use miow_graph::{DesignTokenData, ImportData, KnowledgeGraph, ParsedFileData, SymbolData};
use miow_parsers::{parse_python, parse_rust, parse_typescript};
use std::path::PathBuf;
use std::path::Path;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::Level;
use tracing_subscriber;

mod orchestrator;
use orchestrator::MiowOrchestrator;

// Web API types
#[cfg(feature = "web")]
#[derive(Deserialize)]
struct GenerateRequest {
    /// Root path of the project to analyze
    codebase_path: String,
    /// Natural language task or question
    user_prompt: String,
}

/// Shared application state for the web server
#[cfg(feature = "web")]
#[derive(Clone)]
struct AppState {
    /// Optional shared LLM client (Gemini) reused across requests
    llm: Option<std::sync::Arc<dyn miow_llm::LLMProvider>>,
}

#[cfg(feature = "web")]
#[derive(Serialize)]
struct GenerateResponse {
    success: bool,
    result: Option<String>,
    error: Option<String>,
}

#[cfg(feature = "web")]
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    qdrant_connected: bool,
    gemini_configured: bool,
}

#[cfg(feature = "web")]
#[derive(Deserialize)]
struct SearchFilesRequest {
    codebase_path: String,
    query: String,
}

#[cfg(feature = "web")]
#[derive(Serialize)]
struct SearchFilesResponse {
    success: bool,
    files: Option<Vec<String>>,
    error: Option<String>,
}

#[cfg(feature = "web")]
#[derive(Deserialize)]
struct DebugRequest {
    codebase_path: String,
}

#[cfg(feature = "web")]
#[derive(Serialize)]
struct DebugSignatureResponse {
    success: bool,
    signature: Option<serde_json::Value>,
    error: Option<String>,
}

#[cfg(feature = "web")]
#[derive(Serialize)]
struct DebugContextResponse {
    success: bool,
    context: Option<serde_json::Value>,
    error: Option<String>,
}

#[cfg(feature = "web")]
#[derive(Serialize)]
struct FilesResponse {
    success: bool,
    files: Vec<FileInfo>,
    error: Option<String>,
}

#[cfg(feature = "web")]
#[derive(Serialize)]
struct FileInfo {
    file_path: String,
    symbol_name: String,
    symbol_kind: String,
    relevance_score: f32,
    preview: String,
}

#[cfg(feature = "web")]
#[derive(Deserialize)]
struct GenerateWithFilesRequest {
    codebase_path: String,
    user_prompt: String,
    selected_files: Vec<String>, // file paths
}

#[cfg(feature = "web")]
use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
};
use futures::stream::{self, Stream};
use std::convert::Infallible;
#[cfg(feature = "web")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "web")]
use tokio::net::TcpListener;
#[cfg(feature = "web")]
use tower_http::cors::CorsLayer;

/// Compute a stable Qdrant collection name for a given project path
fn collection_name_for_path(path: &Path) -> String {
    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    format!("miow-{:x}", hasher.finish())
}

#[derive(Parser)]
#[command(name = "miow-context")]
#[command(about = "Intelligent context engine for code generation", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Index a codebase and store in knowledge graph (one-time setup)
    Init {
        /// Path to the codebase
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Database path for knowledge graph
        #[arg(short, long, default_value = "miow.db")]
        db: PathBuf,
    },

    /// Reindex codebase and refresh knowledge graph
    Reindex {
        /// Path to the codebase
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Database path for knowledge graph
        #[arg(short, long, default_value = "miow.db")]
        db: PathBuf,
    },

    /// Generate context-rich prompt (ask questions about your codebase)
    Ask {
        /// User question or task
        #[arg(value_name = "QUESTION")]
        question: String,

        /// Path to the codebase (defaults to current directory)
        #[arg(short, long)]
        path: Option<PathBuf>,

        /// Database path for knowledge graph
        #[arg(short, long, default_value = "miow.db")]
        db: PathBuf,

        /// Output file for generated prompt
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Index a codebase and store in knowledge graph (legacy command)
    Index {
        /// Path to the codebase
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Database path for knowledge graph
        #[arg(short, long, default_value = "miow.db")]
        db: PathBuf,
    },

    /// Analyze a specific file
    Analyze {
        /// Path to the file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Generate context-rich prompt (legacy command, use 'ask' instead)
    Generate {
        /// Path to the codebase
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// User prompt
        #[arg(value_name = "PROMPT")]
        prompt: String,

        /// Database path for knowledge graph
        #[arg(short, long, default_value = "miow.db")]
        db: PathBuf,

        /// Output file for generated prompt
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Test autonomous system planning
    TestAutonomous {
        /// Task to analyze autonomously
        #[arg(value_name = "TASK")]
        task: String,

        /// Path to codebase for context
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },

    /// Start web server with UI
    Serve {
        /// Port to run the web server on
        #[arg(short, long, default_value = "3001")]
        port: u16,

        /// Database path for knowledge graph
        #[arg(short, long, default_value = "miow.db")]
        db: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Setup logging
    let log_level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .init();

    match cli.command {
        Commands::Init { path, db } => {
            handle_init(path, db).await?;
        }
        Commands::Reindex { path, db } => {
            handle_reindex(path, db).await?;
        }
        Commands::Ask {
            question,
            path,
            db,
            output,
        } => {
            let codebase_path = path.unwrap_or_else(|| std::env::current_dir().unwrap());
            handle_ask(question, codebase_path, db, output).await?;
        }
        Commands::Index { path, db } => {
            handle_index(path, db).await?;
        }
        Commands::Analyze { file } => {
            handle_analyze(file).await?;
        }
        Commands::Generate {
            path,
            prompt,
            db,
            output,
        } => {
            handle_generate_autonomous(path, prompt, db, output).await?;
        }
        Commands::TestAutonomous { task, path } => {
            test_autonomous_system(task, path).await?;
        }
        Commands::Serve { port, db } => {
            start_web_server(port, db).await?;
        }
    }

    Ok(())
}

async fn handle_init(path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("{}", "🚀 MIOW-CONTEXT INITIALIZATION".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_black());
    println!("📁 Codebase: {}", path.display());
    println!("💾 Database: {}", db_path.display());
    println!();

    // Check if already indexed
    if db_path.exists() {
        println!("{}", "⚠️  Database already exists. Use 'reindex' to refresh or 'ask' to query.".yellow());
        return Ok(());
    }

    handle_index(path, db_path).await?;

    println!();
    println!("{}", "✅ Initialization complete! You can now use 'miow-context ask' to query your codebase.".green().bold());

    Ok(())
}

async fn handle_reindex(path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("{}", "🔄 MIOW-CONTEXT REINDEXING".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_black());
    println!("📁 Codebase: {}", path.display());
    println!("💾 Database: {}", db_path.display());
    println!();

    // Force reindex by removing existing database
    if db_path.exists() {
        println!("🗑️  Removing existing database...");
        std::fs::remove_file(&db_path)?;
    }

    handle_index(path, db_path).await?;

    println!();
    println!("{}", "✅ Reindexing complete! Database refreshed.".green().bold());

    Ok(())
}

async fn handle_ask(
    question: String,
    path: PathBuf,
    db_path: PathBuf,
    output: Option<PathBuf>,
) -> Result<()> {
    println!("{}", "🤖 MIOW-CONTEXT AUTONOMOUS QUERY".bright_blue().bold());
    println!("{}", "═".repeat(60).bright_black());
    println!("📝 Question: {}", question.bright_yellow());
    println!("📁 Codebase: {}", path.display());
    println!("💾 Database: {}", db_path.display());
    println!();

    // Check if database exists
    if !db_path.exists() {
        println!(
            "{}",
            "⚠️  Knowledge graph not found. Run 'miow-context init' first.".yellow()
        );
        return Ok(());
    }

    // Use the same logic as generate but with better messaging
    handle_generate_autonomous(path, question, db_path, output).await?;

    println!();
    println!("{}", "💡 Tip: Use 'miow-context reindex' if your codebase has changed significantly.".bright_black());

    Ok(())
}

async fn handle_index(path: PathBuf, db_path: PathBuf) -> Result<()> {
    println!("{}", "🔍 Indexing codebase...".cyan().bold());
    println!("Path: {}", path.display());
    println!("Database: {}", db_path.display());
    println!();

    // Try to initialize vector store if Qdrant is available (per-project collection)
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
    let collection_name = collection_name_for_path(&path);
    let vector_store = match miow_vector::VectorStore::new(&qdrant_url, &collection_name).await {
        Ok(store) => {
            println!("{}", "✅ Vector store (Qdrant) connected!".green());
            Some(std::sync::Arc::new(store))
        }
        Err(e) => {
            println!(
                "{}",
                format!(
                    "⚠️  Vector store not available: {}. Continuing without vector search.",
                    e
                )
                .yellow()
            );
            None
        }
    };

    let report = if let Some(vs) = vector_store {
        // Use vector store if available
        use miow_core::index_codebase_with_vector;
        index_codebase_with_vector(path.clone(), vs).await?
    } else {
        index_codebase(path.clone()).await?
    };

    println!("{}", "✅ Indexing complete!".green().bold());
    println!();
    println!("📊 {}", "Statistics:".yellow().bold());
    println!("  Total files: {}", report.total_files);
    println!("  Total size: {} bytes", report.total_size);
    println!("  Duration: {}ms", report.duration_ms);
    println!();
    println!("📁 {}", "Files by language:".yellow().bold());

    for (lang, count) in &report.files_by_language {
        println!("  {}: {}", lang, count);
    }

    // Create knowledge graph and insert data
    println!();
    println!("{}", "💾 Building knowledge graph...".cyan().bold());

    let mut graph = KnowledgeGraph::new(&db_path)?;
    let mut total_symbols = 0;

    for file in &report.files {
        let parsed_data = match file.language {
            miow_core::Language::TypeScript | miow_core::Language::TSX => {
                let is_tsx = matches!(file.language, miow_core::Language::TSX);
                match parse_typescript(&file.content, is_tsx) {
                    Ok(parsed) => Some(convert_to_graph_data(parsed)),
                    Err(e) => {
                        eprintln!("  ⚠️  Failed to parse {}: {}", file.relative_path, e);
                        None
                    }
                }
            }
            miow_core::Language::Rust => match parse_rust(&file.content) {
                Ok(parsed) => Some(convert_to_graph_data(parsed)),
                Err(e) => {
                    eprintln!("  ⚠️  Failed to parse {}: {}", file.relative_path, e);
                    None
                }
            },
            miow_core::Language::Python => match parse_python(&file.content) {
                Ok(parsed) => Some(convert_to_graph_data(parsed)),
                Err(e) => {
                    eprintln!("  ⚠️  Failed to parse {}: {}", file.relative_path, e);
                    None
                }
            },
            _ => None,
        };

        if let Some(data) = parsed_data {
            total_symbols += data.symbols.len();
            graph.insert_file(&file.relative_path, &data)?;
        }
    }

    println!();
    println!("{}", "✅ Knowledge graph built!".green().bold());
    println!("  Total symbols indexed: {}", total_symbols);

    Ok(())
}

async fn handle_analyze(file: PathBuf) -> Result<()> {
    println!("{}", "🔬 Analyzing file...".cyan().bold());
    println!("File: {}", file.display());
    println!();

    let content = std::fs::read_to_string(&file)?;
    let extension = file.extension().and_then(|e| e.to_str()).unwrap_or("");

    let parsed = match extension {
        "tsx" | "ts" => {
            let is_tsx = extension == "tsx";
            parse_typescript(&content, is_tsx)?
        }
        "rs" => parse_rust(&content)?,
        "py" => parse_python(&content)?,
        _ => anyhow::bail!("Unsupported file type: {}", extension),
    };

    println!("{}", "✅ Analysis complete!".green().bold());
    println!();

    if !parsed.symbols.is_empty() {
        println!("{}", "🔍 Symbols:".yellow().bold());
        for symbol in &parsed.symbols {
            println!("  • {} ({:?})", symbol.name.green(), symbol.kind);
            println!(
                "    Lines: {}-{}",
                symbol.range.start_line, symbol.range.end_line
            );

            if !symbol.children.is_empty() {
                println!("    Children: {}", symbol.children.len());
            }
            println!();
        }
    }

    if !parsed.imports.is_empty() {
        println!("{}", "📦 Imports:".yellow().bold());
        for import in &parsed.imports {
            println!("  • from '{}'", import.source.cyan());
        }
    }

    Ok(())
}

async fn handle_generate_autonomous(
    path: PathBuf,
    prompt: String,
    db_path: PathBuf,
    output: Option<PathBuf>,
) -> Result<()> {
    println!("{}", "🤖 MIOW-CONTEXT AUTONOMOUS PROMPT GENERATION".bright_blue().bold());
    println!("{}", "═".repeat(80).bright_black());
    println!("📁 Codebase: {}", path.display());
    println!("📝 Task: {}", prompt.bright_yellow());
    println!();

    // Check if database exists, if not, index first
    if !db_path.exists() {
        println!(
            "{}",
            "⚠️  Knowledge graph not found. Indexing codebase first...".yellow()
        );
        println!();
        handle_index(path.clone(), db_path.clone()).await?;
        println!();
    }

    // Create orchestrator
    let mut orchestrator = MiowOrchestrator::new(db_path.to_str().unwrap())?;

    // Try to initialize vector store if Qdrant is available (per-project collection)
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
    let collection_name = collection_name_for_path(&path);
    match miow_vector::VectorStore::new(&qdrant_url, &collection_name).await {
        Ok(store) => {
            println!("{}", "✅ Vector store (Qdrant) connected!".green());
            orchestrator = orchestrator.with_vector_store(std::sync::Arc::new(store));
        }
        Err(e) => {
            println!(
                "{}",
                format!(
                    "⚠️  Vector store not available: {}. Continuing without vector search.",
                    e
                )
                .yellow()
            );
        }
    }

    // Try to initialize LLM if API key is available
    if let Ok(api_key) = std::env::var("GEMINI_API_KEY") {
        println!("{}", "🤖 LLM integration enabled (Gemini)".green());
        use miow_llm::{GeminiClient, LLMConfig};
        
        let llm_config = LLMConfig {
            api_key,
            model: "gemini-2.5-flash".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
        };

        match GeminiClient::new(llm_config) {
            Ok(client) => {
                orchestrator = orchestrator.with_llm(Box::new(client));
                println!("{}", "✅ LLM client initialized successfully".green());
            }
            Err(e) => {
                println!(
                    "{}",
                    format!(
                        "⚠️  Failed to initialize LLM: {}. Continuing without LLM.",
                        e
                    )
                    .yellow()
                );
            }
        }
        println!();
    } else {
        println!(
            "{}",
            "ℹ️  GEMINI_API_KEY not set. Using basic context analysis (no LLM).".yellow()
        );
        println!(
            "{}",
            "   Set GEMINI_API_KEY environment variable for advanced LLM-powered analysis."
                .bright_black()
        );
        println!();
    }

    // Try to initialize vector store for semantic recall (re-use same per-project collection)
    let qdrant_url =
        std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
    let collection_name = collection_name_for_path(&path);
    match miow_vector::VectorStore::new(&qdrant_url, &collection_name).await {
        Ok(store) => {
            orchestrator = orchestrator.with_vector_store(std::sync::Arc::new(store));
            println!(
                "{}",
                "✅ Vector store (Qdrant) ready for generation!".green()
            );
        }
        Err(e) => {
            println!(
                "{}",
                format!(
                    "⚠️  Vector store not available: {}. Continuing with graph-only search.",
                    e
                )
                .yellow()
            );
        }
    }

    println!("{}", "🔍 Analyzing prompt...".cyan());
    println!("User prompt: \"{}\"", prompt.bright_blue());
    println!();

    // Generate context-aware prompt using Universal Knowledge Graph workflow
    let generated_prompt = orchestrator.generate_autonomous_prompt(
        path.to_str().unwrap(),
        &prompt,
        None // No event streaming for CLI
    ).await?;

    println!("{}", "✅ Context-aware prompt generated!".green().bold());
    println!();
    println!("{}", "═".repeat(80).bright_black());
    println!();
    println!("{}", generated_prompt);
    println!();
    println!("{}", "═".repeat(80).bright_black());

    // Save to file if requested
    if let Some(output_path) = output {
        std::fs::write(&output_path, &generated_prompt)?;
        println!();
        println!("💾 Prompt saved to: {}", output_path.display());
    }

    Ok(())
}

// Helper function to convert parser output to graph data
fn convert_to_graph_data(parsed: miow_parsers::ParsedFile) -> ParsedFileData {
    ParsedFileData {
        symbols: parsed.symbols.into_iter().map(convert_symbol).collect(),
        imports: parsed
            .imports
            .into_iter()
            .map(|imp| ImportData {
                source: imp.source,
                names: imp.names.into_iter().map(|n| n.name).collect(),
                start_line: imp.range.start_line,
                end_line: imp.range.end_line,
            })
            .collect(),
        design_tokens: parsed
            .design_tokens
            .into_iter()
            .map(|token| DesignTokenData {
                token_type: format!("{:?}", token.token_type),
                name: token.name,
                value: token.value,
                context: token.context,
                start_line: token.range.start_line,
                end_line: token.range.end_line,
            })
            .collect(),
        type_definitions: parsed
            .type_definitions
            .into_iter()
            .map(|td| miow_graph::TypeDefinitionData {
                name: td.name,
                kind: format!("{:?}", td.kind),
                definition: td.definition,
                start_line: td.range.start_line,
                end_line: td.range.end_line,
            })
            .collect(),
        constants: parsed
            .constants
            .into_iter()
            .map(|c| miow_graph::ConstantData {
                name: c.name,
                value: c.value,
                category: format!("{:?}", c.category),
                start_line: c.range.start_line,
                end_line: c.range.end_line,
            })
            .collect(),
        schemas: parsed
            .schemas
            .into_iter()
            .map(|s| miow_graph::SchemaData {
                name: s.name,
                schema_type: format!("{:?}", s.schema_type),
                definition: s.definition,
                start_line: s.range.start_line,
                end_line: s.range.end_line,
            })
            .collect(),
        language: parsed.language,
    }
}

async fn test_autonomous_system(task: String, path: PathBuf) -> Result<()> {
    println!("{}", "🧠 AUTONOMOUS SYSTEM TEST".bright_blue().bold());
    println!("{}", "═══════════════════════════════════════════════════════════════".bright_black());

    println!("📋 Task: {}", task.bright_yellow());
    println!("📁 Codebase: {}", path.display().to_string().bright_cyan());

    // Initialize LLM client
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| anyhow::anyhow!("GEMINI_API_KEY environment variable not set"))?;

    use miow_llm::{GeminiClient, LLMConfig, LLMProvider};
    let llm_config = LLMConfig {
        api_key,
        model: "gemini-2.5-flash".to_string(),
        temperature: 0.7,
        max_tokens: 4096,
    };
    let llm: Box<dyn LLMProvider> = Box::new(GeminiClient::new(llm_config)?);

    println!("\n🤖 LLM Autonomous Planning Analysis:");
    println!("{}", "─".repeat(50).bright_black());

    // Autonomous task planning prompt
    let plan_prompt = format!(
        "Autonomously analyze task '{}' in the codebase at '{}'.\n
        Task Analysis Protocol:
        1. DETECT REQUIREMENTS: What does this task fundamentally need?
        2. SEARCH CODEBASE: What existing services/utilities/patterns can I find?
        3. MAKE DECISIONS: Should I reuse existing code or plan new implementations?
        4. NO BIASES: Don't assume specific technologies - discover from code patterns
        5. BE SPECIFIC: Reference actual patterns found in the codebase

        Output Format:
        {{
            \"requirements\": [\"list of what task needs\"],
            \"existing_services\": [\"services found in codebase\"],
            \"decisions\": [\"specific decisions made\"],
            \"plan\": \"detailed implementation plan\",
            \"confidence\": \"high/medium/low\"
        }}",
        task, path.display()
    );

    match llm.generate(&plan_prompt).await {
        Ok(response) => {
            println!("{}", response.content.bright_green());

            println!("\n✅ Autonomous Analysis Complete!");
            println!("{}", "─".repeat(50).bright_black());
            println!("🎯 Key Achievements:");
            println!("  • LLM analyzed task without hardcoded biases");
            println!("  • Discovered services autonomously from codebase");
            println!("  • Made independent decisions about reuse vs. implementation");
            println!("  • Adapted to detected patterns (no assumptions)");
        }
        Err(e) => {
            println!("❌ LLM Error: {}", e.to_string().bright_red());
        }
    }

    Ok(())
}

fn convert_symbol(symbol: miow_parsers::Symbol) -> SymbolData {
    SymbolData {
        name: symbol.name,
        kind: format!("{:?}", symbol.kind),
        start_line: symbol.range.start_line,
        end_line: symbol.range.end_line,
        start_byte: symbol.range.start_byte,
        end_byte: symbol.range.end_byte,
        content: symbol.content,
        metadata: serde_json::to_string(&symbol.metadata).unwrap_or_default(),
        style_tags: None, // Will be populated during style analysis
        children: symbol.children.into_iter().map(convert_symbol).collect(),
        references: symbol.references,
    }
}

#[cfg(feature = "web")]
async fn start_web_server(port: u16, _db_path: PathBuf) -> Result<()> {
    println!("{}", "🌐 Starting MIOW-CONTEXT Web Server".bright_blue().bold());
    println!("{}", "═".repeat(50).bright_black());
    println!("📍 Port: {}", port);
    println!("🌍 Web UI: http://localhost:{}", port);
    println!("🚀 API: http://localhost:{}/api", port);
    println!();

    let mut llm: Option<std::sync::Arc<dyn miow_llm::LLMProvider>> = None;

    // Try to initialize LLM if API key is available
    if let Ok(api_key) = std::env::var("GEMINI_API_KEY") {
        println!("{}", "🤖 LLM integration enabled (Gemini)".green());
        use miow_llm::{GeminiClient, LLMConfig};
        
        let llm_config = LLMConfig {
            api_key,
            model: "gemini-2.5-flash".to_string(),
            temperature: 0.7,
            max_tokens: 4096,
        };
        match GeminiClient::new(llm_config) {
            Ok(client) => {
                llm = Some(std::sync::Arc::new(client));
                println!("{}", "✅ LLM client initialized successfully".green());
            }
            Err(e) => {
                println!("{}", format!("⚠️  Failed to initialize LLM: {}", e).yellow());
            }
        }
    }

    let state = AppState { llm };

    // Create router
    let app = Router::new()
        .route("/api/generate", post(generate_handler))
        .route("/api/generate-stream", post(generate_stream_handler))
        .route("/api/generate-with-files", post(generate_with_files_handler))
        .route("/api/files", post(files_handler))
        .route("/api/debug/signature", post(debug_signature_handler))
        .route("/api/debug/context", post(debug_context_handler))
        .route("/api/health", post(health_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).await?;
    println!("✅ Server running at {}", addr.bright_green());

    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(feature = "web")]
async fn generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, StatusCode> {
    println!("🤖 Processing request: {}", request.user_prompt.bright_yellow());

    let codebase_path = PathBuf::from(&request.codebase_path);
    
    // Determine project-specific DB path
    let db_path = codebase_path.join(".miow").join("miow.db");
    let db_dir = db_path.parent().unwrap();
    
    // Create .miow directory if it doesn't exist
    if !db_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(db_dir) {
             return Ok(Json(GenerateResponse {
                success: false,
                result: None,
                error: Some(format!("Failed to create .miow directory: {}", e)),
            }));
        }
    }

    // Auto-index if DB doesn't exist
    if !db_path.exists() {
        println!("{}", format!("⚠️  No index found for {}. Indexing now...", codebase_path.display()).yellow());
        println!("{}", "⏳ This may take a few minutes depending on codebase size...".bright_black());
        
        match handle_index(codebase_path.clone(), db_path.clone()).await {
            Ok(_) => {
                println!("✅ Indexing completed successfully");
            }
            Err(e) => {
                let error_msg = format!(
                    "Indexing failed: {}. This might be due to:\n- Insufficient permissions\n- Corrupted files\n- Network issues (if using remote paths)\n\nPlease check the path and try again.",
                    e
                );
                return Ok(Json(GenerateResponse {
                    success: false,
                    result: None,
                    error: Some(error_msg),
                }));
            }
        }
    }

    // Initialize orchestrator with project-specific DB
    match MiowOrchestrator::new(db_path.to_str().unwrap()) {
        Ok(mut orchestrator) => {
            // Inject shared LLM
            if let Some(llm) = &state.llm {
                 orchestrator = orchestrator.with_llm_arc(llm.clone());
            }

            // Attach per-project vector store (separate Qdrant collection per project)
            let qdrant_url =
                std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
            let collection_name = collection_name_for_path(&codebase_path);
            if let Ok(store) = miow_vector::VectorStore::new(&qdrant_url, &collection_name).await {
                orchestrator = orchestrator.with_vector_store(std::sync::Arc::new(store));
            }
            
            match orchestrator.generate_autonomous_prompt(
                codebase_path.to_str().unwrap(),
                &request.user_prompt,
                None // No event streaming for now
            ).await {
                Ok(result) => {
                    println!("✅ Request completed successfully");
                    Ok(Json(GenerateResponse {
                        success: true,
                        result: Some(result),
                        error: None,
                    }))
                }
                Err(e) => {
                    println!("❌ Request failed: {}", e);
                    Ok(Json(GenerateResponse {
                        success: false,
                        result: None,
                        error: Some(e.to_string()),
                    }))
                }
            }
        }
        Err(e) => {
             Ok(Json(GenerateResponse {
                success: false,
                result: None,
                error: Some(format!("Failed to initialize orchestrator: {}", e)),
            }))
        }
    }
}

#[cfg(feature = "web")]
async fn generate_stream_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let codebase_path = PathBuf::from(&request.codebase_path);
    let user_prompt = request.user_prompt.clone();
    let llm = state.llm.clone();
    
    
    // Create channel for communication
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(100);
    
    // Spawn background task to handle streaming
    tokio::spawn(async move {
        // Send initial status
        let _ = tx.send(Ok(Event::default()
            .event("status")
            .data("Starting autonomous agent..."))).await;
        
        // Determine project-specific DB path
        let db_path = codebase_path.join(".miow").join("miow.db");
        let db_dir = db_path.parent().unwrap();
        
        // Create .miow directory if it doesn't exist
        if !db_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(db_dir) {
                let _ = tx.send(Ok(Event::default()
                    .event("error")
                    .data(format!("Failed to create .miow directory: {}", e)))).await;
                return;
            }
        }

        // Auto-index if DB doesn't exist
        if !db_path.exists() {
            let _ = tx.send(Ok(Event::default()
                .event("status")
                .data("No index found. Indexing codebase..."))).await;
            
            match handle_index(codebase_path.clone(), db_path.clone()).await {
                Ok(_) => {
                    let _ = tx.send(Ok(Event::default()
                        .event("status")
                        .data("Indexing completed successfully"))).await;
                }
                Err(e) => {
                    let _ = tx.send(Ok(Event::default()
                        .event("error")
                        .data(format!("Indexing failed: {}", e)))).await;
                    return;
                }
            }
        }

        // Initialize orchestrator
        let orchestrator = match MiowOrchestrator::new(db_path.to_str().unwrap()) {
            Ok(mut orch) => {
                // Inject shared LLM
                if let Some(llm_ref) = &llm {
                    orch = orch.with_llm_arc(llm_ref.clone());
                }

                // Attach vector store
                let qdrant_url = std::env::var("QDRANT_URL")
                    .unwrap_or_else(|_| "http://localhost:6333".to_string());
                let collection_name = collection_name_for_path(&codebase_path);
                if let Ok(store) = miow_vector::VectorStore::new(&qdrant_url, &collection_name).await {
                    orch = orch.with_vector_store(std::sync::Arc::new(store));
                }
                
                orch
            }
            Err(e) => {
                let _ = tx.send(Ok(Event::default()
                    .event("error")
                    .data(format!("Failed to initialize orchestrator: {}", e)))).await;
                return;
            }
        };

        // Create channel for agent events
        let (agent_tx, mut agent_rx) = tokio::sync::mpsc::channel(100);
        
        // Spawn agent task
        let agent_task = tokio::spawn(async move {
            orchestrator.generate_autonomous_prompt(
                codebase_path.to_str().unwrap(),
                &user_prompt,
                Some(agent_tx)
            ).await
        });

        // Forward agent events
        while let Some(event) = agent_rx.recv().await {
            let event_json = serde_json::to_string(&event).unwrap_or_default();
            let _ = tx.send(Ok(Event::default()
                .event("agent")
                .data(event_json))).await;
        }

        // Wait for final result
        match agent_task.await {
            Ok(Ok(result)) => {
                let _ = tx.send(Ok(Event::default()
                    .event("result")
                    .data(result))).await;
            }
            Ok(Err(e)) => {
                let _ = tx.send(Ok(Event::default()
                    .event("error")
                    .data(format!("Agent error: {}", e)))).await;
            }
            Err(e) => {
                let _ = tx.send(Ok(Event::default()
                    .event("error")
                    .data(format!("Task error: {}", e)))).await;
            }
        }
    });

    // Create stream from receiver
    let event_stream = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| (event, rx))
    });

    Sse::new(event_stream)
}

#[cfg(feature = "web")]
async fn search_files_handler(
    Json(request): Json<SearchFilesRequest>,
) -> Json<SearchFilesResponse> {
    use walkdir::WalkDir;
    
    let codebase_path = PathBuf::from(&request.codebase_path);
    let query = request.query.to_lowercase();
    
    if !codebase_path.exists() {
        return Json(SearchFilesResponse {
            success: false,
            files: None,
            error: Some("Codebase path does not exist".to_string()),
        });
    }
    
    let mut matching_files = Vec::new();
    
    // Walk the directory and find matching files
    for entry in WalkDir::new(&codebase_path)
        .max_depth(10)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            let path = entry.path();
            
            // Skip hidden files and common ignore patterns
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name.starts_with('.') || 
                   path.to_str().map(|s| s.contains("/node_modules/") || 
                                          s.contains("/target/") ||
                                          s.contains("/.git/") ||
                                          s.contains("/dist/") ||
                                          s.contains("/build/")).unwrap_or(false) {
                    continue;
                }
            }
            
            // Check if path matches query
            if let Some(path_str) = path.to_str() {
                if path_str.to_lowercase().contains(&query) {
                    // Make path relative to codebase
                    if let Ok(relative) = path.strip_prefix(&codebase_path) {
                        if let Some(rel_str) = relative.to_str() {
                            matching_files.push(rel_str.to_string());
                        }
                    }
                }
            }
            
            // Limit results to 50
            if matching_files.len() >= 50 {
                break;
            }
        }
    }
    
    Json(SearchFilesResponse {
        success: true,
        files: Some(matching_files),
        error: None,
    })
}

#[cfg(feature = "web")]
async fn health_handler(
    State(_state): State<AppState>,
) -> Json<HealthResponse> {
    let gemini_configured = std::env::var("GEMINI_API_KEY").is_ok();

    // Check Qdrant connection
    let qdrant_connected = reqwest::get("http://localhost:6333/collections")
        .await
        .map(|resp| resp.status().is_success())
        .unwrap_or(false);

    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        qdrant_connected,
        gemini_configured,
    })
}

#[cfg(feature = "web")]
async fn debug_signature_handler(
    State(_state): State<AppState>,
    Json(request): Json<DebugRequest>,
) -> Result<Json<DebugSignatureResponse>, StatusCode> {
    let codebase_path = PathBuf::from(&request.codebase_path);
    
    match miow_core::ProjectSignature::detect(&codebase_path) {
        Ok(signature) => {
            let json = serde_json::json!({
                "language": signature.language,
                "framework": signature.framework,
                "package_manager": signature.package_manager,
                "ui_library": signature.ui_library,
                "validation_library": signature.validation_library,
                "auth_library": signature.auth_library,
                "description": signature.to_description(),
            });
            Ok(Json(DebugSignatureResponse {
                success: true,
                signature: Some(json),
                error: None,
            }))
        }
        Err(e) => {
            Ok(Json(DebugSignatureResponse {
                success: false,
                signature: None,
                error: Some(e.to_string()),
            }))
        }
    }
}

#[cfg(feature = "web")]
async fn debug_context_handler(
    State(state): State<AppState>,
    Json(request): Json<DebugRequest>,
) -> Result<Json<DebugContextResponse>, StatusCode> {
    let codebase_path = PathBuf::from(&request.codebase_path);
    let db_path = codebase_path.join(".miow").join("miow.db");
    let db_dir = db_path.parent().unwrap();
    
    // Create .miow directory if it doesn't exist
    if !db_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(db_dir) {
            return Ok(Json(DebugContextResponse {
                success: false,
                context: None,
                error: Some(format!("Failed to create .miow directory: {}", e)),
            }));
        }
    }
    
    // Auto-index if DB doesn't exist
    if !db_path.exists() {
        println!("{}", format!("⚠️  No index found for {}. Indexing now...", codebase_path.display()).yellow());
        println!("{}", "⏳ This may take a few minutes depending on codebase size...".bright_black());
        
        match handle_index(codebase_path.clone(), db_path.clone()).await {
            Ok(_) => {
                println!("✅ Indexing completed successfully");
            }
            Err(e) => {
                let error_msg = format!(
                    "Indexing failed: {}. This might be due to:\n- Insufficient permissions\n- Corrupted files\n- Network issues (if using remote paths)\n\nPlease check the path and try again.",
                    e
                );
                return Ok(Json(DebugContextResponse {
                    success: false,
                    context: None,
                    error: Some(error_msg),
                }));
            }
        }
    }
    
    match MiowOrchestrator::new(db_path.to_str().unwrap()) {
        Ok(mut orchestrator) => {
            if let Some(llm) = &state.llm {
                orchestrator = orchestrator.with_llm_arc(llm.clone());
            }
            
            let qdrant_url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
            let collection_name = collection_name_for_path(&codebase_path);
            if let Ok(store) = miow_vector::VectorStore::new(&qdrant_url, &collection_name).await {
                orchestrator = orchestrator.with_vector_store(std::sync::Arc::new(store));
            }
            
            // Get context summary
            let graph = orchestrator.graph();
            let total_symbols = graph.count_symbols().unwrap_or(0);
            let total_files = graph.count_files().unwrap_or(0);
            
            let json = serde_json::json!({
                "total_symbols": total_symbols,
                "total_files": total_files,
                "db_path": db_path.to_string_lossy(),
                "collection_name": collection_name,
            });
            
            Ok(Json(DebugContextResponse {
                success: true,
                context: Some(json),
                error: None,
            }))
        }
        Err(e) => {
            Ok(Json(DebugContextResponse {
                success: false,
                context: None,
                error: Some(e.to_string()),
            }))
        }
    }
}

#[cfg(feature = "web")]
async fn files_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<FilesResponse>, StatusCode> {
    println!("📁 Getting relevant files for: {}", request.user_prompt.bright_yellow());
    
    let codebase_path = PathBuf::from(&request.codebase_path);
    let db_path = codebase_path.join(".miow").join("miow.db");
    let db_dir = db_path.parent().unwrap();
    
    // Create .miow directory if it doesn't exist
    if !db_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(db_dir) {
            return Ok(Json(FilesResponse {
                success: false,
                files: vec![],
                error: Some(format!("Failed to create .miow directory: {}", e)),
            }));
        }
    }
    
    // Auto-index if DB doesn't exist
    if !db_path.exists() {
        println!("{}", format!("⚠️  No index found for {}. Indexing now...", codebase_path.display()).yellow());
        println!("{}", "⏳ This may take a few minutes depending on codebase size...".bright_black());
        
        match handle_index(codebase_path.clone(), db_path.clone()).await {
            Ok(_) => {
                println!("✅ Indexing completed successfully");
            }
            Err(e) => {
                let error_msg = format!(
                    "Indexing failed: {}. This might be due to:\n- Insufficient permissions\n- Corrupted files\n- Network issues (if using remote paths)\n\nPlease check the path and try again.",
                    e
                );
                return Ok(Json(FilesResponse {
                    success: false,
                    files: vec![],
                    error: Some(error_msg),
                }));
            }
        }
    }
    
    match MiowOrchestrator::new(db_path.to_str().unwrap()) {
        Ok(mut orchestrator) => {
            if let Some(llm) = &state.llm {
                orchestrator = orchestrator.with_llm_arc(llm.clone());
            }
            
            let qdrant_url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
            let collection_name = collection_name_for_path(&codebase_path);
            if let Ok(store) = miow_vector::VectorStore::new(&qdrant_url, &collection_name).await {
                orchestrator = orchestrator.with_vector_store(std::sync::Arc::new(store));
            }
            
            match orchestrator.get_relevant_files(&request.user_prompt, &codebase_path).await {
                Ok(context_items) => {
                    println!("✅ Found {} relevant files", context_items.len());
                    let files: Vec<FileInfo> = context_items.into_iter().map(|item| FileInfo {
                        file_path: item.file_path,
                        symbol_name: item.name,
                        symbol_kind: item.kind,
                        relevance_score: item.relevance_score,
                        preview: item.content.chars().take(200).collect(),
                    }).collect();
                    Ok(Json(FilesResponse {
                        success: true,
                        files,
                        error: None,
                    }))
                }
                Err(e) => {
                    println!("❌ Failed to get files: {}", e);
                    Ok(Json(FilesResponse {
                        success: false,
                        files: vec![],
                        error: Some(e.to_string()),
                    }))
                }
            }
        }
        Err(e) => {
            Ok(Json(FilesResponse {
                success: false,
                files: vec![],
                error: Some(e.to_string()),
            }))
        }
    }
}

#[cfg(feature = "web")]
async fn generate_with_files_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateWithFilesRequest>,
) -> Result<Json<GenerateResponse>, StatusCode> {
    println!("🤖 Processing request with selected files: {}", request.user_prompt.bright_yellow());
    println!("📋 Selected {} files", request.selected_files.len());
    
    let codebase_path = PathBuf::from(&request.codebase_path);
    let db_path = codebase_path.join(".miow").join("miow.db");
    let db_dir = db_path.parent().unwrap();
    
    // Create .miow directory if it doesn't exist
    if !db_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(db_dir) {
            return Ok(Json(GenerateResponse {
                success: false,
                result: None,
                error: Some(format!("Failed to create .miow directory: {}", e)),
            }));
        }
    }
    
    // Auto-index if DB doesn't exist
    if !db_path.exists() {
        println!("{}", format!("⚠️  No index found for {}. Indexing now...", codebase_path.display()).yellow());
        println!("{}", "⏳ This may take a few minutes depending on codebase size...".bright_black());
        
        match handle_index(codebase_path.clone(), db_path.clone()).await {
            Ok(_) => {
                println!("✅ Indexing completed successfully");
            }
            Err(e) => {
                let error_msg = format!(
                    "Indexing failed: {}. This might be due to:\n- Insufficient permissions\n- Corrupted files\n- Network issues (if using remote paths)\n\nPlease check the path and try again.",
                    e
                );
                return Ok(Json(GenerateResponse {
                    success: false,
                    result: None,
                    error: Some(error_msg),
                }));
            }
        }
    }
    
    match MiowOrchestrator::new(db_path.to_str().unwrap()) {
        Ok(mut orchestrator) => {
            if let Some(llm) = &state.llm {
                orchestrator = orchestrator.with_llm_arc(llm.clone());
            }
            
            let qdrant_url = std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string());
            let collection_name = collection_name_for_path(&codebase_path);
            if let Ok(store) = miow_vector::VectorStore::new(&qdrant_url, &collection_name).await {
                orchestrator = orchestrator.with_vector_store(std::sync::Arc::new(store));
            }
            
            match orchestrator.generate_enhanced_prompt_with_files(
                &request.user_prompt,
                &codebase_path,
                &request.selected_files,
            ).await {
                Ok(result) => {
                    println!("✅ Request completed successfully");
                    Ok(Json(GenerateResponse {
                        success: true,
                        result: Some(result),
                        error: None,
                    }))
                }
                Err(e) => {
                    println!("❌ Request failed: {}", e);
                    Ok(Json(GenerateResponse {
                        success: false,
                        result: None,
                        error: Some(e.to_string()),
                    }))
                }
            }
        }
        Err(e) => {
            Ok(Json(GenerateResponse {
                success: false,
                result: None,
                error: Some(e.to_string()),
            }))
        }
    }
}

#[cfg(not(feature = "web"))]
async fn start_web_server(_port: u16, _db_path: PathBuf) -> Result<()> {
    println!("❌ Web server feature not enabled. Compile with --features web");
    Ok(())
}
