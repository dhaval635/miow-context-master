use crate::types::*;
use anyhow::{Context, Result};
use tree_sitter::{Node, Parser, Query, QueryCursor};

pub struct PythonParser {
    parser: Parser,
}

impl PythonParser {
    pub fn new() -> Self {
        let mut parser = Parser::new();
        let language = tree_sitter_python::language();
        parser
            .set_language(language)
            .expect("Error loading Python grammar");
        Self { parser }
    }

    pub fn parse(&self, content: &str) -> Result<ParsedFile> {
        let mut parser = Parser::new();
        parser
            .set_language(tree_sitter_python::language())
            .context("Failed to set Python language")?;

        let tree = parser
            .parse(content, None)
            .context("Failed to parse Python content")?;

        let root_node = tree.root_node();

        let symbols = self.extract_symbols(&root_node, content)?;
        let imports = self.extract_imports(&root_node, content)?;
        let type_definitions = self.extract_type_definitions(&root_node, content)?;
        let constants = self.extract_constants(&root_node, content)?;
        let schemas = self.extract_schemas(&root_node, content)?;

        Ok(ParsedFile {
            symbols,
            imports,
            exports: vec![], // Python exports are implicit (everything not starting with _)
            design_tokens: vec![],
            type_definitions,
            constants,
            schemas,
            language: "python".to_string(),
        })
    }

    fn extract_symbols(&self, node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut symbols = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if let Some(symbol) = self.process_node(&child, source)? {
                symbols.push(symbol);
            }
        }

        Ok(symbols)
    }

    fn process_node(&self, node: &Node, source: &str) -> Result<Option<Symbol>> {
        let kind = node.kind();
        let text = node.utf8_text(source.as_bytes())?;

        match kind {
            "class_definition" => {
                let name = self
                    .get_child_text(node, "name", source)
                    .unwrap_or_else(|| "Anonymous".to_string());
                let range = self.get_range(node);
                let metadata = self.extract_metadata(node, source)?;

                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Class,
                    range,
                    content: text.to_string(),
                    metadata,
                    children: self.extract_class_members(node, source)?,
                    references: vec![],
                }))
            }
            "function_definition" => {
                let name = self
                    .get_child_text(node, "name", source)
                    .unwrap_or_else(|| "anonymous".to_string());
                let range = self.get_range(node);
                let metadata = self.extract_function_metadata(node, source)?;

                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Function,
                    range,
                    content: text.to_string(),
                    metadata,
                    children: vec![],
                    references: vec![],
                }))
            }
            "assignment" => {
                // Global variables
                if let Some(left) = node.child_by_field_name("left") {
                    let name = left.utf8_text(source.as_bytes())?.to_string();
                    Ok(Some(Symbol {
                        name,
                        kind: SymbolType::Variable,
                        range: self.get_range(node),
                        content: text.to_string(),
                        metadata: SymbolMetadata::default(),
                        children: vec![],
                        references: vec![],
                    }))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn extract_class_members(&self, node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut members = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                match child.kind() {
                    "function_definition" | "decorated_definition" => {
                        // Handle both regular and decorated methods
                        let func_node = if child.kind() == "decorated_definition" {
                            child.child_by_field_name("definition")
                                .filter(|n| n.kind() == "function_definition")
                        } else {
                            Some(child)
                        };

                        if let Some(func) = func_node {
                            let name = self
                                .get_child_text(&func, "name", source)
                                .unwrap_or_else(|| "method".to_string());
                            let mut metadata = self.extract_function_metadata(&func, source)?;
                            
                            // Extract decorators from parent if decorated
                            if child.kind() == "decorated_definition" {
                                metadata.decorators = self.extract_decorators(&child, source)?;
                            }
                            
                            // Determine method type based on decorators
                            let symbol_kind = if metadata.decorators.iter().any(|d| d.contains("@property")) {
                                SymbolType::Property
                            } else if metadata.decorators.iter().any(|d| d.contains("@classmethod")) {
                                metadata.is_static = true;
                                SymbolType::Method
                            } else if metadata.decorators.iter().any(|d| d.contains("@staticmethod")) {
                                metadata.is_static = true;
                                SymbolType::Method
                            } else {
                                SymbolType::Method
                            };

                            members.push(Symbol {
                                name,
                                kind: symbol_kind,
                                range: self.get_range(&func),
                                content: child.utf8_text(source.as_bytes())?.to_string(),
                                metadata,
                                children: vec![],
                                references: vec![],
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(members)
    }

    fn extract_metadata(&self, node: &Node, source: &str) -> Result<SymbolMetadata> {
        let mut metadata = SymbolMetadata::default();

        // Extract decorators
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "decorator" {
                metadata.decorators.push(child.utf8_text(source.as_bytes())?.to_string());
            }
        }

        // Check inheritance
        if let Some(superclasses) = node.child_by_field_name("superclasses") {
            let mut cursor = superclasses.walk();
            for child in superclasses.children(&mut cursor) {
                if child.kind() == "identifier" || child.kind() == "attribute" {
                    metadata
                        .extends
                        .push(child.utf8_text(source.as_bytes())?.to_string());
                }
            }
        }

        Ok(metadata)
    }

    fn extract_function_metadata(&self, node: &Node, source: &str) -> Result<SymbolMetadata> {
        let mut metadata = self.extract_metadata(node, source)?;

        // Extract parameters
        if let Some(params_node) = node.child_by_field_name("parameters") {
            metadata.parameters = self.extract_parameters(&params_node, source)?;
        }

        // Extract return type (PEP 484 type hints)
        if let Some(return_type) = node.child_by_field_name("return_type") {
            metadata.return_type = Some(return_type.utf8_text(source.as_bytes())?.to_string());
        }

        // Extract docstring
        if let Some(body) = node.child_by_field_name("body") {
            metadata.documentation = self.extract_docstring(&body, source)?;
        }

        metadata.is_async = node.utf8_text(source.as_bytes())?.starts_with("async");

        Ok(metadata)
    }

    fn extract_parameters(&self, node: &Node, source: &str) -> Result<Vec<Parameter>> {
        let mut params = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            let kind = child.kind();
            if kind == "identifier"
                || kind == "typed_parameter"
                || kind == "default_parameter"
                || kind == "typed_default_parameter"
            {
                let mut name = String::new();
                let mut type_annotation = None;
                let mut default_value = None;

                if kind == "identifier" {
                    name = child.utf8_text(source.as_bytes())?.to_string();
                } else if kind == "typed_parameter" {
                    name = self
                        .get_child_text(&child, "name", source)
                        .unwrap_or_default();
                    type_annotation = self.get_child_text(&child, "type", source);
                } else if kind == "default_parameter" {
                    name = self
                        .get_child_text(&child, "name", source)
                        .unwrap_or_default();
                    default_value = self.get_child_text(&child, "value", source);
                } else if kind == "typed_default_parameter" {
                    name = self
                        .get_child_text(&child, "name", source)
                        .unwrap_or_default();
                    type_annotation = self.get_child_text(&child, "type", source);
                    default_value = self.get_child_text(&child, "value", source);
                }

                params.push(Parameter {
                    name,
                    type_annotation,
                    default_value,
                    is_optional: false,
                });
            }
        }
        Ok(params)
    }

    fn extract_imports(&self, node: &Node, source: &str) -> Result<Vec<Import>> {
        let mut imports = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "import_statement" {
                // import x, y
                let text = child.utf8_text(source.as_bytes())?;
                imports.push(Import {
                    source: text.to_string(), // Simplified
                    names: vec![],
                    range: self.get_range(&child),
                });
            } else if child.kind() == "import_from_statement" {
                // from x import y
                let module_name = self
                    .get_child_text(&child, "module_name", source)
                    .unwrap_or_default();
                imports.push(Import {
                    source: module_name,
                    names: vec![],
                    range: self.get_range(&child),
                });
            }
        }
        Ok(imports)
    }

    fn get_child_text(&self, node: &Node, field: &str, source: &str) -> Option<String> {
        node.child_by_field_name(field)
            .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string())
    }

    fn get_range(&self, node: &Node) -> Range {
        Range {
            start_line: node.start_position().row + 1,
            end_line: node.end_position().row + 1,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        }
    }

    fn extract_decorators(&self, node: &Node, source: &str) -> Result<Vec<String>> {
        let mut decorators = Vec::new();
        let mut cursor = node.walk();
        
        for child in node.children(&mut cursor) {
            if child.kind() == "decorator" {
                decorators.push(child.utf8_text(source.as_bytes())?.to_string());
            }
        }
        
        Ok(decorators)
    }

    fn extract_docstring(&self, body_node: &Node, source: &str) -> Result<Option<String>> {
        // First statement in body might be a docstring
        let mut cursor = body_node.walk();
        
        for child in body_node.children(&mut cursor) {
            if child.kind() == "expression_statement" {
                if let Some(string_node) = child.child(0) {
                    if string_node.kind() == "string" {
                        let docstring = string_node.utf8_text(source.as_bytes())?.to_string();
                        // Remove quotes
                        let cleaned = docstring
                            .trim_start_matches("\"\"\"")
                            .trim_start_matches("'''")
                            .trim_start_matches('"')
                            .trim_start_matches('\'')
                            .trim_end_matches("\"\"\"")
                            .trim_end_matches("'''")
                            .trim_end_matches('"')
                            .trim_end_matches('\'')
                            .trim()
                            .to_string();
                        return Ok(Some(cleaned));
                    }
                }
                break; // Only check first statement
            }
        }
        
        Ok(None)
    }

    fn extract_type_definitions(&self, node: &Node, source: &str) -> Result<Vec<TypeDefinition>> {
        let mut type_defs = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "class_definition" => {
                    if let Some(type_def) = self.extract_class_type_def(&child, source)? {
                        type_defs.push(type_def);
                    }
                }
                "type_alias_statement" => {
                    // Python 3.12+ type aliases: type MyType = SomeType
                    if let Some(type_def) = self.extract_type_alias(&child, source)? {
                        type_defs.push(type_def);
                    }
                }
                _ => {}
            }
        }

        Ok(type_defs)
    }

    fn extract_class_type_def(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "name", source)
            .unwrap_or_else(|| "Anonymous".to_string());
        
        let mut properties = Vec::new();
        let mut generic_params = Vec::new();

        // Extract type parameters (Python 3.12+)
        if let Some(type_params) = node.child_by_field_name("type_parameters") {
            generic_params = self.extract_generic_params(&type_params, source)?;
        }

        // Extract class attributes and methods
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                match child.kind() {
                    "expression_statement" => {
                        // Type-annotated class variables
                        if let Some(ann_assign) = child.child(0) {
                            if ann_assign.kind() == "assignment" {
                                if let Some(left) = ann_assign.child_by_field_name("left") {
                                    let prop_name = left.utf8_text(source.as_bytes())?.to_string();
                                    let type_annotation = ann_assign
                                        .child_by_field_name("type")
                                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string())
                                        .unwrap_or_else(|| "Any".to_string());

                                    properties.push(TypeProperty {
                                        name: prop_name,
                                        type_annotation,
                                        is_optional: false,
                                        description: None,
                                    });
                                }
                            }
                        }
                    }
                    "function_definition" => {
                        let method_name = self
                            .get_child_text(&child, "name", source)
                            .unwrap_or_else(|| "method".to_string());
                        
                        // Get method signature
                        let signature = child.utf8_text(source.as_bytes())?
                            .lines()
                            .next()
                            .unwrap_or("")
                            .to_string();

                        properties.push(TypeProperty {
                            name: method_name,
                            type_annotation: signature,
                            is_optional: false,
                            description: None,
                        });
                    }
                    _ => {}
                }
            }
        }

        Ok(Some(TypeDefinition {
            name,
            kind: TypeKind::Interface, // Python classes are like interfaces
            definition: node.utf8_text(source.as_bytes())?.to_string(),
            properties,
            generic_params,
            range: self.get_range(node),
        }))
    }

    fn extract_type_alias(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "name", source)
            .unwrap_or_else(|| "Anonymous".to_string());
        
        let mut generic_params = Vec::new();

        if let Some(type_params) = node.child_by_field_name("type_parameters") {
            generic_params = self.extract_generic_params(&type_params, source)?;
        }

        Ok(Some(TypeDefinition {
            name,
            kind: TypeKind::TypeAlias,
            definition: node.utf8_text(source.as_bytes())?.to_string(),
            properties: vec![],
            generic_params,
            range: self.get_range(node),
        }))
    }

    fn extract_generic_params(&self, node: &Node, source: &str) -> Result<Vec<String>> {
        let mut params = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "type_parameter" || child.kind() == "identifier" {
                params.push(child.utf8_text(source.as_bytes())?.to_string());
            }
        }

        Ok(params)
    }

    fn extract_constants(&self, node: &Node, source: &str) -> Result<Vec<Constant>> {
        let mut constants = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "assignment" {
                if let Some(left) = child.child_by_field_name("left") {
                    let name = left.utf8_text(source.as_bytes())?.to_string();
                    
                    // Python convention: UPPERCASE names are constants
                    if name.chars().all(|c| c.is_uppercase() || c == '_' || c.is_numeric()) {
                        let value = child
                            .child_by_field_name("right")
                            .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string())
                            .unwrap_or_else(|| "unknown".to_string());
                        
                        // Try to extract type annotation if present
                        let type_annotation = child
                            .child_by_field_name("type")
                            .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());

                        // Categorize based on name patterns
                        let category = if name.contains("URL") || name.contains("ENDPOINT") || name.contains("API") {
                            ConstantCategory::APIEndpoint
                        } else if name.contains("ERROR") || name.contains("MESSAGE") {
                            ConstantCategory::ErrorMessage
                        } else if name.contains("DEFAULT") {
                            ConstantCategory::DefaultValue
                        } else if name.contains("CONFIG") || name.contains("SETTINGS") {
                            ConstantCategory::Config
                        } else {
                            ConstantCategory::Other
                        };

                        constants.push(Constant {
                            name,
                            value,
                            type_annotation,
                            category,
                            range: self.get_range(&child),
                        });
                    }
                }
            }
        }

        Ok(constants)
    }

    fn extract_schemas(&self, node: &Node, source: &str) -> Result<Vec<ValidationSchema>> {
        let mut schemas = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "class_definition" {
                // Check if it's a Pydantic model
                if let Some(superclasses) = child.child_by_field_name("superclasses") {
                    let superclass_text = superclasses.utf8_text(source.as_bytes())?;
                    
                    if superclass_text.contains("BaseModel") || superclass_text.contains("pydantic") {
                        let name = self
                            .get_child_text(&child, "name", source)
                            .unwrap_or_else(|| "Model".to_string());
                        
                        let mut fields = Vec::new();
                        
                        // Extract Pydantic fields
                        if let Some(body) = child.child_by_field_name("body") {
                            let mut body_cursor = body.walk();
                            for body_child in body.children(&mut body_cursor) {
                                if body_child.kind() == "expression_statement" {
                                    if let Some(assignment) = body_child.child(0) {
                                        if assignment.kind() == "assignment" {
                                            if let Some(left) = assignment.child_by_field_name("left") {
                                                let field_name = left.utf8_text(source.as_bytes())?.to_string();
                                                
                                                let type_annotation = assignment
                                                    .child_by_field_name("type")
                                                    .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());
                                                
                                                let default_value = assignment
                                                    .child_by_field_name("right")
                                                    .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());

                                                fields.push(SchemaField {
                                                    name: field_name,
                                                    validation_rules: vec![],
                                                    is_required: default_value.is_none(),
                                                    default_value,
                                                    type_annotation,
                                                    is_optional: false,
                                                    validators: vec![],
                                                    description: None,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        schemas.push(ValidationSchema {
                            name,
                            schema_type: SchemaType::Other("Pydantic".to_string()),
                            definition: child.utf8_text(source.as_bytes())?.to_string(),
                            fields,
                            range: self.get_range(&child),
                        });
                    }
                }
            }
        }

        Ok(schemas)
    }
}

impl Default for PythonParser {
    fn default() -> Self {
        Self::new()
    }
}
