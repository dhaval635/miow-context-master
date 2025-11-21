use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub mod query;
pub mod schema;
pub mod semantic_search;
pub mod relationship_inference;
pub mod query_expansion;

pub use query::*;
pub use schema::*;
pub use semantic_search::{SemanticGraphSearch, SemanticSearchResult};
pub use relationship_inference::{RelationshipInferencer, InferredRelationship, RelationshipType};
pub use query_expansion::{QueryExpander, ExpandedQuery};

use std::sync::Mutex;

/// Knowledge graph for storing and querying code symbols
pub struct KnowledgeGraph {
    conn: Mutex<Connection>,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph with the given database path
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        let graph = Self { conn: Mutex::new(conn) };
        graph.initialize_schema()?;
        Ok(graph)
    }

    /// Create an in-memory knowledge graph (useful for testing)
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let graph = Self { conn: Mutex::new(conn) };
        graph.initialize_schema()?;
        Ok(graph)
    }

    /// Initialize the database schema
    fn initialize_schema(&self) -> Result<()> {
        self.conn.lock().unwrap().execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                language TEXT NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                start_byte INTEGER NOT NULL,
                end_byte INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                parent_id INTEGER,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES symbols(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS symbol_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_symbol_id INTEGER NOT NULL,
                to_symbol_name TEXT NOT NULL,
                reference_type TEXT NOT NULL,
                FOREIGN KEY (from_symbol_id) REFERENCES symbols(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                names TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS design_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                token_type TEXT NOT NULL,
                name TEXT NOT NULL,
                value TEXT NOT NULL,
                context TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS type_definitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                definition TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS constants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                value TEXT NOT NULL,
                category TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS schemas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                schema_type TEXT NOT NULL,
                definition TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
            CREATE INDEX IF NOT EXISTS idx_references_from ON symbol_references(from_symbol_id);
            CREATE INDEX IF NOT EXISTS idx_references_to ON symbol_references(to_symbol_name);
            CREATE INDEX IF NOT EXISTS idx_design_tokens_name ON design_tokens(name);
            CREATE INDEX IF NOT EXISTS idx_type_definitions_name ON type_definitions(name);
            CREATE INDEX IF NOT EXISTS idx_constants_name ON constants(name);
            CREATE INDEX IF NOT EXISTS idx_schemas_name ON schemas(name);
            "#,
        )?;
        Ok(())
    }

    /// Insert a file and its symbols into the graph
    pub fn insert_file(&mut self, file_path: &str, parsed_file: &ParsedFileData) -> Result<i64> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;

        // Insert file
        tx.execute(
            "INSERT OR REPLACE INTO files (path, language) VALUES (?1, ?2)",
            params![file_path, parsed_file.language],
        )?;

        let file_id = tx.last_insert_rowid();

        // Insert symbols
        for symbol in &parsed_file.symbols {
            insert_symbol_recursive(&tx, file_id, symbol, None)?;
        }

        // Insert imports
        for import in &parsed_file.imports {
            tx.execute(
                "INSERT INTO imports (file_id, source, names, start_line, end_line) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    file_id,
                    import.source,
                    serde_json::to_string(&import.names)?,
                    import.start_line,
                    import.end_line
                ],
            )?;
        }

        // Insert design tokens
        for token in &parsed_file.design_tokens {
            tx.execute(
                "INSERT INTO design_tokens (file_id, token_type, name, value, context, start_line, end_line) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    file_id,
                    token.token_type,
                    token.name,
                    token.value,
                    token.context,
                    token.start_line,
                    token.end_line
                ],
            )?;
        }

        // Insert type definitions
        for type_def in &parsed_file.type_definitions {
            tx.execute(
                "INSERT INTO type_definitions (file_id, name, kind, definition, start_line, end_line) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    file_id,
                    type_def.name,
                    type_def.kind,
                    type_def.definition,
                    type_def.start_line,
                    type_def.end_line
                ],
            )?;
        }

        // Insert constants
        for constant in &parsed_file.constants {
            tx.execute(
                "INSERT INTO constants (file_id, name, value, category, start_line, end_line) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    file_id,
                    constant.name,
                    constant.value,
                    constant.category,
                    constant.start_line,
                    constant.end_line
                ],
            )?;
        }

        // Insert schemas
        for schema in &parsed_file.schemas {
            tx.execute(
                "INSERT INTO schemas (file_id, name, schema_type, definition, start_line, end_line) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    file_id,
                    schema.name,
                    schema.schema_type,
                    schema.definition,
                    schema.start_line,
                    schema.end_line
                ],
            )?;
        }

        tx.commit()?;
        Ok(file_id)
    }
}

fn insert_symbol_recursive(
    tx: &rusqlite::Transaction,
    file_id: i64,
    symbol: &SymbolData,
    parent_id: Option<i64>,
) -> Result<i64> {
    let metadata_json = serde_json::to_string(&symbol.metadata)?;

    tx.execute(
        "INSERT INTO symbols (file_id, name, kind, start_line, end_line, start_byte, end_byte, content, metadata, parent_id) 
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            file_id,
            symbol.name,
            symbol.kind,
            symbol.start_line,
            symbol.end_line,
            symbol.start_byte,
            symbol.end_byte,
            symbol.content,
            metadata_json,
            parent_id
        ],
    )?;

    let symbol_id = tx.last_insert_rowid();

    // Insert references
    for reference in &symbol.references {
        tx.execute(
            "INSERT INTO symbol_references (from_symbol_id, to_symbol_name, reference_type) VALUES (?1, ?2, ?3)",
            params![symbol_id, reference, "uses"],
        )?;
    }

    // Insert children recursively
    for child in &symbol.children {
        insert_symbol_recursive(tx, file_id, child, Some(symbol_id))?;
    }

    Ok(symbol_id)
}

impl KnowledgeGraph {
    /// Search for symbols by name (fuzzy match)
    pub fn search_symbols(&self, query: &str) -> Result<Vec<SymbolSearchResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT s.id, s.name, s.kind, s.content, f.path, s.start_line, s.end_line, s.metadata
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE s.name LIKE ?1
            ORDER BY s.name
            LIMIT 50
            "#,
        )?;

        let pattern = format!("%{}%", query);
        let results = stmt.query_map(params![pattern], |row| {
            Ok(SymbolSearchResult {
                id: row.get(0)?,
                name: row.get(1)?,
                kind: row.get(2)?,
                content: row.get(3)?,
                file_path: row.get(4)?,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
                metadata: row.get(7)?,
            })
        })?;

        let mut symbols = Vec::new();
        for result in results {
            symbols.push(result?);
        }
        Ok(symbols)
    }

    /// Find symbols by exact name
    pub fn find_symbols_by_name(&self, name: &str) -> Result<Vec<SymbolSearchResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT s.id, s.name, s.kind, s.content, f.path, s.start_line, s.end_line, s.metadata
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE s.name = ?1
            "#,
        )?;

        let results = stmt.query_map(params![name], |row| {
            Ok(SymbolSearchResult {
                id: row.get(0)?,
                name: row.get(1)?,
                kind: row.get(2)?,
                content: row.get(3)?,
                file_path: row.get(4)?,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
                metadata: row.get(7)?,
            })
        })?;

        let mut symbols = Vec::new();
        for result in results {
            symbols.push(result?);
        }
        Ok(symbols)
    }

    /// Find symbols by kind (e.g., "Component", "Function")
    pub fn find_symbols_by_kind(&self, kind: &str) -> Result<Vec<SymbolSearchResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT s.id, s.name, s.kind, s.content, f.path, s.start_line, s.end_line, s.metadata
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE s.kind = ?1
            ORDER BY s.name
            "#,
        )?;

        let results = stmt.query_map(params![kind], |row| {
            Ok(SymbolSearchResult {
                id: row.get(0)?,
                name: row.get(1)?,
                kind: row.get(2)?,
                content: row.get(3)?,
                file_path: row.get(4)?,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
                metadata: row.get(7)?,
            })
        })?;

        let mut symbols = Vec::new();
        for result in results {
            symbols.push(result?);
        }
        Ok(symbols)
    }

    /// Find design tokens by name
    pub fn find_design_tokens(&self, query: &str) -> Result<Vec<DesignTokenResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT dt.name, dt.value, dt.token_type, dt.context, f.path
            FROM design_tokens dt
            JOIN files f ON dt.file_id = f.id
            WHERE dt.name LIKE ?1
            "#,
        )?;

        let pattern = format!("%{}%", query);
        let results = stmt.query_map(params![pattern], |row| {
            Ok(DesignTokenResult {
                name: row.get(0)?,
                value: row.get(1)?,
                token_type: row.get(2)?,
                context: row.get(3)?,
                file_path: row.get(4)?,
            })
        })?;

        let mut tokens = Vec::new();
        for result in results {
            tokens.push(result?);
        }
        Ok(tokens)
    }

    /// Get symbols that reference a given symbol name
    pub fn find_references_to(&self, symbol_name: &str) -> Result<Vec<SymbolSearchResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT DISTINCT s.id, s.name, s.kind, s.content, f.path, s.start_line, s.end_line, s.metadata
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            JOIN symbol_references r ON r.from_symbol_id = s.id
            WHERE r.to_symbol_name = ?1
            "#,
        )?;

        let results = stmt.query_map(params![symbol_name], |row| {
            Ok(SymbolSearchResult {
                id: row.get(0)?,
                name: row.get(1)?,
                kind: row.get(2)?,
                content: row.get(3)?,
                file_path: row.get(4)?,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
                metadata: row.get(7)?,
            })
        })?;

        let mut symbols = Vec::new();
        for result in results {
            symbols.push(result?);
        }
        Ok(symbols)
    }

    /// Get names of symbols referenced by a given symbol
    pub fn get_symbol_dependencies(&self, symbol_id: i64) -> Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT to_symbol_name FROM symbol_references WHERE from_symbol_id = ?1"
        )?;
        
        let rows = stmt.query_map(params![symbol_id], |row| row.get(0))?;
        
        let mut refs = Vec::new();
        for row in rows {
            refs.push(row?);
        }
        Ok(refs)
    }

    /// Get all symbols in a file
    pub fn get_file_symbols(&self, file_path: &str) -> Result<Vec<SymbolSearchResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT s.id, s.name, s.kind, s.content, f.path, s.start_line, s.end_line, s.metadata
            FROM symbols s
            JOIN files f ON s.file_id = f.id
            WHERE f.path = ?1
            ORDER BY s.start_line
            "#,
        )?;

        let results = stmt.query_map(params![file_path], |row| {
            Ok(SymbolSearchResult {
                id: row.get(0)?,
                name: row.get(1)?,
                kind: row.get(2)?,
                content: row.get(3)?,
                file_path: row.get(4)?,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
                metadata: row.get(7)?,
            })
        })?;

        let mut symbols = Vec::new();
        for result in results {
            symbols.push(result?);
        }
        Ok(symbols)
    }

    /// Find type definitions by name
    pub fn find_type_definitions(&self, query: &str) -> Result<Vec<TypeDefinitionResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT td.name, td.kind, td.definition, f.path, td.start_line, td.end_line
            FROM type_definitions td
            JOIN files f ON td.file_id = f.id
            WHERE td.name LIKE ?1
            "#,
        )?;

        let pattern = format!("%{}%", query);
        let results = stmt.query_map(params![pattern], |row| {
            Ok(TypeDefinitionResult {
                name: row.get(0)?,
                kind: row.get(1)?,
                definition: row.get(2)?,
                file_path: row.get(3)?,
                start_line: row.get(4)?,
                end_line: row.get(5)?,
            })
        })?;

        let mut types = Vec::new();
        for result in results {
            types.push(result?);
        }
        Ok(types)
    }

    /// Find constants by name
    pub fn find_constants(&self, query: &str) -> Result<Vec<ConstantResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT c.name, c.value, c.category, f.path, c.start_line, c.end_line
            FROM constants c
            JOIN files f ON c.file_id = f.id
            WHERE c.name LIKE ?1
            "#,
        )?;

        let pattern = format!("%{}%", query);
        let results = stmt.query_map(params![pattern], |row| {
            Ok(ConstantResult {
                name: row.get(0)?,
                value: row.get(1)?,
                category: row.get(2)?,
                file_path: row.get(3)?,
                start_line: row.get(4)?,
                end_line: row.get(5)?,
            })
        })?;

        let mut constants = Vec::new();
        for result in results {
            constants.push(result?);
        }
        Ok(constants)
    }

    /// Count total symbols in the graph
    pub fn count_symbols(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM symbols",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Count total files in the graph
    pub fn count_files(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM files",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Find schemas by name
    pub fn find_schemas(&self, query: &str) -> Result<Vec<SchemaResult>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"
            SELECT s.name, s.schema_type, s.definition, f.path, s.start_line, s.end_line
            FROM schemas s
            JOIN files f ON s.file_id = f.id
            WHERE s.name LIKE ?1
            "#,
        )?;

        let pattern = format!("%{}%", query);
        let results = stmt.query_map(params![pattern], |row| {
            Ok(SchemaResult {
                name: row.get(0)?,
                schema_type: row.get(1)?,
                definition: row.get(2)?,
                file_path: row.get(3)?,
                start_line: row.get(4)?,
                end_line: row.get(5)?,
            })
        })?;

        let mut schemas = Vec::new();
        for result in results {
            schemas.push(result?);
        }
        Ok(schemas)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolSearchResult {
    pub id: i64,
    pub name: String,
    pub kind: String,
    pub content: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
    pub metadata: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignTokenResult {
    pub name: String,
    pub value: String,
    pub token_type: String,
    pub context: String,
    pub file_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefinitionResult {
    pub name: String,
    pub kind: String,
    pub definition: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantResult {
    pub name: String,
    pub value: String,
    pub category: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaResult {
    pub name: String,
    pub schema_type: String,
    pub definition: String,
    pub file_path: String,
    pub start_line: i64,
    pub end_line: i64,
}
