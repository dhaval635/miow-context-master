use crate::types::*;
use anyhow::{Context, Result};
use tree_sitter::{Node, Parser, Query, QueryCursor};

pub struct RustParser {
    parser: Parser,
}

impl RustParser {
    pub fn new() -> Self {
        let mut parser = Parser::new();
        let language = tree_sitter_rust::language();
        parser
            .set_language(language)
            .expect("Error loading Rust grammar");
        Self { parser }
    }

    pub fn parse(&self, content: &str) -> Result<ParsedFile> {
        let mut parser = Parser::new();
        parser
            .set_language(tree_sitter_rust::language())
            .context("Failed to set Rust language")?;

        let tree = parser
            .parse(content, None)
            .context("Failed to parse Rust content")?;

        let root_node = tree.root_node();

        let symbols = self.extract_symbols(&root_node, content)?;
        let imports = self.extract_imports(&root_node, content)?;
        let type_definitions = self.extract_type_definitions(&root_node, content)?;
        let constants = self.extract_constants(&root_node, content)?;

        Ok(ParsedFile {
            symbols,
            imports,
            exports: vec![], // Rust exports are usually pub, handled in symbols
            design_tokens: vec![],
            type_definitions,
            constants,
            schemas: vec![],          // Rust doesn't have runtime validation schemas like Zod
            language: "rust".to_string(),
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
            "struct_item" => {
                let name = self
                    .get_child_text(node, "type_identifier", source)
                    .unwrap_or_else(|| "Anonymous".to_string());
                let range = self.get_range(node);
                let metadata = self.extract_metadata(node, source)?;

                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Struct,
                    range,
                    content: text.to_string(),
                    metadata,
                    children: self.extract_struct_fields(node, source)?,
                    references: vec![],
                }))
            }
            "enum_item" => {
                let name = self
                    .get_child_text(node, "type_identifier", source)
                    .unwrap_or_else(|| "Anonymous".to_string());
                let range = self.get_range(node);
                let metadata = self.extract_metadata(node, source)?;

                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Enum,
                    range,
                    content: text.to_string(),
                    metadata,
                    children: self.extract_enum_variants(node, source)?,
                    references: vec![],
                }))
            }
            "function_item" => {
                let name = self
                    .get_child_text(node, "identifier", source)
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
            "impl_item" => {
                let type_name = self
                    .get_child_text(node, "type_identifier", source)
                    .unwrap_or_default();
                let trait_name = if let Some(trait_node) = node.child_by_field_name("trait") {
                    trait_node.utf8_text(source.as_bytes())?.to_string()
                } else {
                    String::new()
                };

                let name = if !trait_name.is_empty() {
                    format!("impl {} for {}", trait_name, type_name)
                } else {
                    format!("impl {}", type_name)
                };

                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Class, // Mapping impl to Class-like structure for now
                    range: self.get_range(node),
                    content: text.to_string(),
                    metadata: SymbolMetadata::default(),
                    children: self.extract_impl_members(node, source)?,
                    references: vec![],
                }))
            }
            "macro_definition" => {
                let name = self
                    .get_child_text(node, "identifier", source)
                    .unwrap_or_else(|| "anonymous".to_string());
                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Function, // Macros are function-like
                    range: self.get_range(node),
                    content: text.to_string(),
                    metadata: SymbolMetadata::default(),
                    children: vec![],
                    references: vec![],
                }))
            }
            "mod_item" => {
                let name = self
                    .get_child_text(node, "identifier", source)
                    .unwrap_or_else(|| "anonymous".to_string());
                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Module,
                    range: self.get_range(node),
                    content: text.to_string(),
                    metadata: SymbolMetadata::default(),
                    children: vec![], // Could recurse if inline mod
                    references: vec![],
                }))
            }
            "trait_item" => {
                let name = self
                    .get_child_text(node, "type_identifier", source)
                    .unwrap_or_else(|| "Anonymous".to_string());
                let range = self.get_range(node);
                let mut metadata = self.extract_metadata(node, source)?;
                
                // Extract generic parameters
                if let Some(type_params) = node.child_by_field_name("type_parameters") {
                    metadata.generic_params = self.extract_generic_params(&type_params, source)?;
                }

                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Interface, // Traits are similar to interfaces
                    range,
                    content: text.to_string(),
                    metadata,
                    children: self.extract_trait_members(node, source)?,
                    references: vec![],
                }))
            }
            "type_item" => {
                // Type alias: type MyType = SomeType;
                let name = self
                    .get_child_text(node, "type_identifier", source)
                    .unwrap_or_else(|| "Anonymous".to_string());
                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::TypeParameter,
                    range: self.get_range(node),
                    content: text.to_string(),
                    metadata: SymbolMetadata::default(),
                    children: vec![],
                    references: vec![],
                }))
            }
            "const_item" => {
                let name = self
                    .get_child_text(node, "identifier", source)
                    .unwrap_or_else(|| "CONST".to_string());
                let mut metadata = self.extract_metadata(node, source)?;
                
                // Extract type
                if let Some(type_node) = node.child_by_field_name("type") {
                    metadata.return_type = Some(type_node.utf8_text(source.as_bytes())?.to_string());
                }
                
                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Constant,
                    range: self.get_range(node),
                    content: text.to_string(),
                    metadata,
                    children: vec![],
                    references: vec![],
                }))
            }
            "static_item" => {
                let name = self
                    .get_child_text(node, "identifier", source)
                    .unwrap_or_else(|| "STATIC".to_string());
                let mut metadata = self.extract_metadata(node, source)?;
                metadata.is_static = true;
                
                if let Some(type_node) = node.child_by_field_name("type") {
                    metadata.return_type = Some(type_node.utf8_text(source.as_bytes())?.to_string());
                }
                
                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Variable,
                    range: self.get_range(node),
                    content: text.to_string(),
                    metadata,
                    children: vec![],
                    references: vec![],
                }))
            }
            _ => Ok(None),
        }
    }

    fn extract_struct_fields(&self, node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut fields = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "field_declaration" {
                    let name = self
                        .get_child_text(&child, "name", source)
                        .unwrap_or_else(|| "field".to_string());
                    let type_node = child.child_by_field_name("type");
                    let type_annotation =
                        type_node.map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());

                    let mut metadata = SymbolMetadata::default();
                    metadata.return_type = type_annotation;
                    metadata.access_modifier =
                        if child.utf8_text(source.as_bytes())?.starts_with("pub") {
                            Some("public".to_string())
                        } else {
                            Some("private".to_string())
                        };

                    fields.push(Symbol {
                        name,
                        kind: SymbolType::Field,
                        range: self.get_range(&child),
                        content: child.utf8_text(source.as_bytes())?.to_string(),
                        metadata,
                        children: vec![],
                        references: vec![],
                    });
                }
            }
        }
        Ok(fields)
    }

    fn extract_enum_variants(&self, node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut variants = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "enum_variant" {
                    let name = self
                        .get_child_text(&child, "name", source)
                        .unwrap_or_else(|| "variant".to_string());
                    variants.push(Symbol {
                        name,
                        kind: SymbolType::EnumMember,
                        range: self.get_range(&child),
                        content: child.utf8_text(source.as_bytes())?.to_string(),
                        metadata: SymbolMetadata::default(),
                        children: vec![],
                        references: vec![],
                    });
                }
            }
        }
        Ok(variants)
    }

    fn extract_impl_members(&self, node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut members = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "function_item" {
                    let name = self
                        .get_child_text(&child, "name", source)
                        .unwrap_or_else(|| "fn".to_string());
                    let metadata = self.extract_function_metadata(&child, source)?;

                    members.push(Symbol {
                        name,
                        kind: SymbolType::Method,
                        range: self.get_range(&child),
                        content: child.utf8_text(source.as_bytes())?.to_string(),
                        metadata,
                        children: vec![],
                        references: vec![],
                    });
                }
            }
        }
        Ok(members)
    }

    fn extract_metadata(&self, node: &Node, source: &str) -> Result<SymbolMetadata> {
        let mut metadata = SymbolMetadata::default();

        // Check visibility
        let text = node.utf8_text(source.as_bytes())?;
        metadata.access_modifier = if text.starts_with("pub") {
            Some("public".to_string())
        } else {
            Some("private".to_string())
        };

        Ok(metadata)
    }

    fn extract_function_metadata(&self, node: &Node, source: &str) -> Result<SymbolMetadata> {
        let mut metadata = self.extract_metadata(node, source)?;

        // Extract parameters
        if let Some(params_node) = node.child_by_field_name("parameters") {
            metadata.parameters = self.extract_parameters(&params_node, source)?;
        }

        // Extract return type
        if let Some(return_type) = node.child_by_field_name("return_type") {
            metadata.return_type = Some(return_type.utf8_text(source.as_bytes())?.to_string());
        }

        metadata.is_async = node.utf8_text(source.as_bytes())?.contains("async fn");

        Ok(metadata)
    }

    fn extract_parameters(&self, node: &Node, source: &str) -> Result<Vec<Parameter>> {
        let mut params = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "parameter" {
                let name = self
                    .get_child_text(&child, "pattern", source)
                    .unwrap_or_else(|| "_".to_string());
                let type_annotation = child
                    .child_by_field_name("type")
                    .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());

                params.push(Parameter {
                    name,
                    type_annotation,
                    default_value: None,
                    is_optional: false,
                });
            } else if child.kind() == "self_parameter" {
                params.push(Parameter {
                    name: "self".to_string(),
                    type_annotation: None,
                    default_value: None,
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
            if child.kind() == "use_declaration" {
                let text = child.utf8_text(source.as_bytes())?;
                let path = text.trim_start_matches("use ").trim_end_matches(';');

                imports.push(Import {
                    source: path.to_string(),
                    names: vec![], // Rust imports are complex, just storing the path for now
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

    fn extract_trait_members(&self, node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut members = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                match child.kind() {
                    "function_signature_item" => {
                        let name = self
                            .get_child_text(&child, "name", source)
                            .unwrap_or_else(|| "fn".to_string());
                        let metadata = self.extract_function_metadata(&child, source)?;

                        members.push(Symbol {
                            name,
                            kind: SymbolType::Method,
                            range: self.get_range(&child),
                            content: child.utf8_text(source.as_bytes())?.to_string(),
                            metadata,
                            children: vec![],
                            references: vec![],
                        });
                    }
                    "associated_type" => {
                        let name = self
                            .get_child_text(&child, "type_identifier", source)
                            .unwrap_or_else(|| "AssociatedType".to_string());
                        
                        members.push(Symbol {
                            name,
                            kind: SymbolType::TypeParameter,
                            range: self.get_range(&child),
                            content: child.utf8_text(source.as_bytes())?.to_string(),
                            metadata: SymbolMetadata::default(),
                            children: vec![],
                            references: vec![],
                        });
                    }
                    _ => {}
                }
            }
        }
        Ok(members)
    }

    fn extract_generic_params(&self, node: &Node, source: &str) -> Result<Vec<String>> {
        let mut params = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "type_identifier" || child.kind() == "lifetime" {
                params.push(child.utf8_text(source.as_bytes())?.to_string());
            }
        }

        Ok(params)
    }

    fn extract_type_definitions(&self, node: &Node, source: &str) -> Result<Vec<TypeDefinition>> {
        let mut type_defs = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "struct_item" => {
                    if let Some(type_def) = self.extract_struct_type_def(&child, source)? {
                        type_defs.push(type_def);
                    }
                }
                "enum_item" => {
                    if let Some(type_def) = self.extract_enum_type_def(&child, source)? {
                        type_defs.push(type_def);
                    }
                }
                "type_item" => {
                    if let Some(type_def) = self.extract_type_alias(&child, source)? {
                        type_defs.push(type_def);
                    }
                }
                "trait_item" => {
                    if let Some(type_def) = self.extract_trait_type_def(&child, source)? {
                        type_defs.push(type_def);
                    }
                }
                _ => {}
            }
        }

        Ok(type_defs)
    }

    fn extract_struct_type_def(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "type_identifier", source)
            .unwrap_or_else(|| "Anonymous".to_string());
        
        let mut properties = Vec::new();
        let mut generic_params = Vec::new();

        // Extract generic parameters
        if let Some(type_params) = node.child_by_field_name("type_parameters") {
            generic_params = self.extract_generic_params(&type_params, source)?;
        }

        // Extract fields
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "field_declaration" {
                    let field_name = self
                        .get_child_text(&child, "name", source)
                        .unwrap_or_else(|| "field".to_string());
                    let type_annotation = child
                        .child_by_field_name("type")
                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string())
                        .unwrap_or_else(|| "unknown".to_string());

                    properties.push(TypeProperty {
                        name: field_name,
                        type_annotation,
                        is_optional: false, // Rust doesn't have optional fields in the same way
                        description: None,
                    });
                }
            }
        }

        Ok(Some(TypeDefinition {
            name,
            kind: TypeKind::Interface, // Structs are like interfaces
            definition: node.utf8_text(source.as_bytes())?.to_string(),
            properties,
            generic_params,
            range: self.get_range(node),
        }))
    }

    fn extract_enum_type_def(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "type_identifier", source)
            .unwrap_or_else(|| "Anonymous".to_string());
        
        let mut properties = Vec::new();
        let mut generic_params = Vec::new();

        // Extract generic parameters
        if let Some(type_params) = node.child_by_field_name("type_parameters") {
            generic_params = self.extract_generic_params(&type_params, source)?;
        }

        // Extract variants
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "enum_variant" {
                    let variant_name = self
                        .get_child_text(&child, "name", source)
                        .unwrap_or_else(|| "variant".to_string());
                    
                    let variant_type = child.utf8_text(source.as_bytes())?.to_string();

                    properties.push(TypeProperty {
                        name: variant_name,
                        type_annotation: variant_type,
                        is_optional: false,
                        description: None,
                    });
                }
            }
        }

        Ok(Some(TypeDefinition {
            name,
            kind: TypeKind::Enum,
            definition: node.utf8_text(source.as_bytes())?.to_string(),
            properties,
            generic_params,
            range: self.get_range(node),
        }))
    }

    fn extract_type_alias(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "type_identifier", source)
            .unwrap_or_else(|| "Anonymous".to_string());
        
        let mut generic_params = Vec::new();

        // Extract generic parameters
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

    fn extract_trait_type_def(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "type_identifier", source)
            .unwrap_or_else(|| "Anonymous".to_string());
        
        let mut properties = Vec::new();
        let mut generic_params = Vec::new();

        // Extract generic parameters
        if let Some(type_params) = node.child_by_field_name("type_parameters") {
            generic_params = self.extract_generic_params(&type_params, source)?;
        }

        // Extract trait methods and associated types
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                match child.kind() {
                    "function_signature_item" => {
                        let method_name = self
                            .get_child_text(&child, "name", source)
                            .unwrap_or_else(|| "fn".to_string());
                        
                        let signature = child.utf8_text(source.as_bytes())?.to_string();

                        properties.push(TypeProperty {
                            name: method_name,
                            type_annotation: signature,
                            is_optional: false,
                            description: None,
                        });
                    }
                    "associated_type" => {
                        let type_name = self
                            .get_child_text(&child, "type_identifier", source)
                            .unwrap_or_else(|| "AssociatedType".to_string());
                        
                        properties.push(TypeProperty {
                            name: type_name,
                            type_annotation: "type".to_string(),
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
            kind: TypeKind::Interface, // Traits are like interfaces
            definition: node.utf8_text(source.as_bytes())?.to_string(),
            properties,
            generic_params,
            range: self.get_range(node),
        }))
    }

    fn extract_constants(&self, node: &Node, source: &str) -> Result<Vec<Constant>> {
        let mut constants = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "const_item" => {
                    let name = self
                        .get_child_text(&child, "identifier", source)
                        .unwrap_or_else(|| "CONST".to_string());
                    
                    let type_annotation = child
                        .child_by_field_name("type")
                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());
                    
                    let value = child
                        .child_by_field_name("value")
                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string())
                        .unwrap_or_else(|| "unknown".to_string());

                    constants.push(Constant {
                        name,
                        value,
                        type_annotation,
                        category: ConstantCategory::Other,
                        range: self.get_range(&child),
                    });
                }
                "static_item" => {
                    let name = self
                        .get_child_text(&child, "identifier", source)
                        .unwrap_or_else(|| "STATIC".to_string());
                    
                    let type_annotation = child
                        .child_by_field_name("type")
                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());
                    
                    let value = child
                        .child_by_field_name("value")
                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string())
                        .unwrap_or_else(|| "unknown".to_string());

                    constants.push(Constant {
                        name,
                        value,
                        type_annotation,
                        category: ConstantCategory::Config,
                        range: self.get_range(&child),
                    });
                }
                _ => {}
            }
        }

        Ok(constants)
    }
}

impl Default for RustParser {
    fn default() -> Self {
        Self::new()
    }
}
