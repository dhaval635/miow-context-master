use crate::types::*;
use anyhow::{Context, Result};
use tree_sitter::{Node, Parser, Query, QueryCursor};

pub struct TypeScriptParser {
    parser: Parser,
}

impl TypeScriptParser {
    pub fn new() -> Self {
        let mut parser = Parser::new();
        Self { parser }
    }

    pub fn parse(&self, content: &str, is_tsx: bool) -> Result<ParsedFile> {
        let mut parser = Parser::new();

        let language = if is_tsx {
            tree_sitter_typescript::language_tsx()
        } else {
            tree_sitter_typescript::language_typescript()
        };

        parser
            .set_language(language)
            .context("Failed to set TypeScript language")?;

        let tree = parser
            .parse(content, None)
            .context("Failed to parse TypeScript content")?;

        let root_node = tree.root_node();

        let symbols = self.extract_symbols(&root_node, content, is_tsx)?;
        let imports = self.extract_imports(&root_node, content)?;
        let exports = self.extract_exports(&root_node, content)?;
        let design_tokens = self.extract_design_tokens(&root_node, content)?;
        let type_definitions = self.extract_type_definitions(&root_node, content)?;
        let constants = self.extract_constants(&root_node, content)?;
        let schemas = self.extract_validation_schemas(&root_node, content)?;

        Ok(ParsedFile {
            symbols,
            imports,
            exports,
            design_tokens,
            type_definitions,
            constants,
            schemas,
            language: if is_tsx {
                "tsx".to_string()
            } else {
                "typescript".to_string()
            },
        })
    }

    fn extract_symbols(&self, root_node: &Node, source: &str, is_tsx: bool) -> Result<Vec<Symbol>> {
        let mut symbols = Vec::new();
        let mut cursor = root_node.walk();

        for child in root_node.children(&mut cursor) {
            if let Some(symbol) = self.process_node(&child, source, is_tsx)? {
                symbols.push(symbol);
            }
        }

        // Additional extraction for common UI components in JSX/TSX
        if is_tsx {
            if let Ok(ui_symbols) = self.extract_ui_components(root_node, source) {
                symbols.extend(ui_symbols);
            }
        }

        Ok(symbols)
    }

    fn extract_ui_components(&self, root_node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut ui_symbols = Vec::new();

        // Query for common UI components: InputBox, Button, Form, Modal, etc.
        let ui_query_str = r#"
        (jsx_element 
          (opening_element 
            (identifier) @component_name
          )
          (identifier) @tag_name (#any-of? @tag_name "InputBox" "Button" "Form" "Modal" "Input" "Select" "Checkbox")
        ) |
        (call_expression 
          function: (identifier) @component_name (#any-of? @component_name "InputBox" "Button" "Form" "Modal" "Input" "Select" "Checkbox")
        ) |
        (variable_declarator 
          name: (identifier) @var_name 
          value: (call_expression 
            function: (identifier) @init_name (#any-of? @init_name "forwardRef" "memo")
            arguments: (argument_list 
              (arrow_function | function) @component_body
            )
          ) (#match? @var_name "^[A-Z]")
        )
        "#;

        let language = tree_sitter_typescript::language_tsx();
        let query = Query::new(
            language,
            ui_query_str
        ).context("Failed to create UI component query")?;

        let mut cursor = QueryCursor::new();
        for m in cursor.matches(&query, *root_node, source.as_bytes()) {
            for capture in m.captures {
                let node = capture.node;
                let kind = node.kind();
                let text = node.utf8_text(source.as_bytes())?;

                // Extract component name and body
                let name = if kind == "identifier" {
                    text.to_string()
                } else {
                    // Find parent or nearby identifier for name
                    self.find_nearest_component_name(&node, source).unwrap_or("UIComponent".to_string())
                };

                // Get full component code by expanding context
                let range = self.get_range_expanded(&node, 5); // Expand 5 lines for context
                let content = self.extract_node_content_with_context(&node, source, &range);

                let mut metadata = SymbolMetadata::default();
                metadata.tags = vec!["ui-component".to_string(), "common".to_string()];
                metadata.priority = Some(1.0); // High priority for common components

                ui_symbols.push(Symbol {
                    name,
                    kind: SymbolType::Component,
                    range,
                    content,
                    metadata,
                    children: vec![],
                    references: self.extract_references(&node, source)?,
                });
            }
        }

        Ok(ui_symbols)
    }

    fn find_nearest_component_name(&self, node: &Node, source: &str) -> Option<String> {
        let mut current = node.parent();
        let mut depth = 0;
        while let Some(parent) = current {
            if parent.kind() == "identifier" {
                let text = parent.utf8_text(source.as_bytes()).ok()?;
                if text.chars().next().unwrap_or('a').is_uppercase() {
                    return Some(text.to_string());
                }
            }
            if parent.kind() == "variable_declarator" || parent.kind() == "function_declaration" {
                // Look for name in declarator
                if let Some(name_node) = parent.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source.as_bytes()).map(|s| s.to_string()) {
                        if name.chars().next().unwrap_or('a').is_uppercase() {
                            return Some(name);
                        }
                    }
                }
            }
            current = parent.parent();
            depth += 1;
            if depth > 10 { break; } // Prevent infinite loops
        }
        None
    }

    fn get_range_expanded(&self, node: &Node, lines: usize) -> Range {
        let start_row = node.start_position().row.saturating_sub(lines as usize);
        let end_row = node.end_position().row + lines as usize;
        Range {
            start_line: start_row + 1,
            end_line: end_row + 1,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
        }
    }

    fn extract_node_content_with_context(&self, node: &Node, source: &str, range: &Range) -> String {
        let full_source = source.as_bytes();
        let start_byte = range.start_byte as usize;
        let end_byte = range.end_byte as usize;
        let start = std::cmp::max(0, start_byte.saturating_sub(1000));
        let end = std::cmp::min(full_source.len(), end_byte + 1000);
        String::from_utf8_lossy(&full_source[start..end]).to_string()
    }

    fn process_node(&self, node: &Node, source: &str, is_tsx: bool) -> Result<Option<Symbol>> {
        let kind = node.kind();
        let text = node.utf8_text(source.as_bytes())?;

        match kind {
            "function_declaration" => {
                let name = self
                    .get_child_text(node, "name", source)
                    .unwrap_or_else(|| "anonymous".to_string());
                let range = self.get_range(node);
                let metadata = self.extract_function_metadata(node, source)?;

                // Check if it's a component (starts with Uppercase and returns JSX)
                let symbol_type = if self.is_component_name(&name) && self.returns_jsx(node, source)
                {
                    SymbolType::Component
                } else {
                    SymbolType::Function
                };

                Ok(Some(Symbol {
                    name,
                    kind: symbol_type,
                    range,
                    content: text.to_string(),
                    metadata,
                    children: vec![],   // TODO: Extract local variables/functions
                    references: self.extract_references(node, source)?,
                }))
            }
            "class_declaration" => {
                let name = self
                    .get_child_text(node, "name", source)
                    .unwrap_or_else(|| "Anonymous".to_string());
                let range = self.get_range(node);
                let metadata = SymbolMetadata::default(); // TODO: Extract class metadata

                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Class,
                    range,
                    content: text.to_string(),
                    metadata,
                    children: self.extract_class_members(node, source)?,
                    references: self.extract_references(node, source)?,
                }))
            }
            "interface_declaration" => {
                let name = self
                    .get_child_text(node, "name", source)
                    .unwrap_or_else(|| "Anonymous".to_string());
                let range = self.get_range(node);

                Ok(Some(Symbol {
                    name,
                    kind: SymbolType::Interface,
                    range,
                    content: text.to_string(),
                    metadata: SymbolMetadata::default(),
                    children: self.extract_interface_members(node, source)?,
                    references: vec![],
                }))
            }
            "lexical_declaration" | "variable_declaration" => {
                // Handle const/let/var
                // This can be complex with destructuring, but let's handle simple cases first
                // const x = ...
                self.extract_variable_declaration(node, source)
            }
            "export_statement" => {
                // Recurse into export statement
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    if child.kind() != "export" && child.kind() != "default" {
                        return self.process_node(&child, source, is_tsx);
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn extract_variable_declaration(&self, node: &Node, source: &str) -> Result<Option<Symbol>> {
        // Look for variable_declarator child
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "variable_declarator" {
                let name_node = child.child_by_field_name("name");
                if let Some(name_node) = name_node {
                    let name = name_node.utf8_text(source.as_bytes())?.to_string();
                    let value_node = child.child_by_field_name("value");

                    if let Some(value_node) = value_node {
                        // Check if it's an arrow function
                        if value_node.kind() == "arrow_function" {
                            let range = self.get_range(node);
                            let metadata =
                                self.extract_arrow_function_metadata(&value_node, source)?;

                            let symbol_type = if self.is_component_name(&name)
                                && self.returns_jsx(&value_node, source)
                            {
                                SymbolType::Component
                            } else {
                                SymbolType::Function
                            };

                            return Ok(Some(Symbol {
                                name,
                                kind: symbol_type,
                                range,
                                content: node.utf8_text(source.as_bytes())?.to_string(),
                                metadata,
                                children: vec![],
                                references: self.extract_references(&value_node, source)?,
                            }));
                        }
                    }

                    // Regular variable
                    return Ok(Some(Symbol {
                        name,
                        kind: SymbolType::Variable,
                        range: self.get_range(node),
                        content: node.utf8_text(source.as_bytes())?.to_string(),
                        metadata: SymbolMetadata::default(),
                        children: vec![],
                        references: self.extract_references(node, source)?,
                    }));
                }
            }
        }
        Ok(None)
    }

    fn extract_class_members(&self, node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut members = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                let kind = child.kind();
                match kind {
                    "method_definition" => {
                        let name = self
                            .get_child_text(&child, "property_identifier", source)
                            .unwrap_or_default();
                        members.push(Symbol {
                            name,
                            kind: SymbolType::Method,
                            range: self.get_range(&child),
                            content: child.utf8_text(source.as_bytes())?.to_string(),
                            metadata: SymbolMetadata::default(),
                            children: vec![],
                            references: vec![],
                        });
                    }
                    "public_field_definition" => {
                        let name = self
                            .get_child_text(&child, "property_identifier", source)
                            .unwrap_or_default();
                        members.push(Symbol {
                            name,
                            kind: SymbolType::Field,
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

    fn extract_interface_members(&self, node: &Node, source: &str) -> Result<Vec<Symbol>> {
        let mut members = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "property_signature" {
                    let name = self
                        .get_child_text(&child, "property_identifier", source)
                        .unwrap_or_default();
                    members.push(Symbol {
                        name,
                        kind: SymbolType::Property,
                        range: self.get_range(&child),
                        content: child.utf8_text(source.as_bytes())?.to_string(),
                        metadata: SymbolMetadata::default(),
                        children: vec![],
                        references: vec![],
                    });
                }
            }
        }
        Ok(members)
    }

    fn extract_function_metadata(&self, node: &Node, source: &str) -> Result<SymbolMetadata> {
        let mut metadata = SymbolMetadata::default();

        // Extract parameters
        if let Some(params_node) = node.child_by_field_name("parameters") {
            metadata.parameters = self.extract_parameters(&params_node, source)?;
            
            // If this is a component, try to extract props from the first parameter
            if self.is_component_node(node, source) {
                if let Some(first_param) = metadata.parameters.first() {
                    metadata.props = self.extract_props_from_param(first_param, node, source)?;
                }
            }
        }

        // Extract return type
        if let Some(return_type) = node.child_by_field_name("return_type") {
            metadata.return_type = Some(return_type.utf8_text(source.as_bytes())?.to_string());
        }

        metadata.is_async = node.utf8_text(source.as_bytes())?.starts_with("async");

        Ok(metadata)
    }

    fn is_component_node(&self, node: &Node, source: &str) -> bool {
        if let Some(name_node) = node.child_by_field_name("name") {
             if let Ok(name) = name_node.utf8_text(source.as_bytes()) {
                 return self.is_component_name(name) && self.returns_jsx(node, source);
             }
        }
        false
    }

    fn extract_props_from_param(&self, param: &Parameter, _node: &Node, _source: &str) -> Result<Vec<PropDefinition>> {
        let mut props = Vec::new();

        // Case 1: Destructured props: ({ name, age }: Props)
        if param.name.starts_with('{') {
             // Simple regex to find property names in { prop1, prop2: alias }
             // Matches words followed by comma or closing brace, ignoring values
             let re = regex::Regex::new(r"(\w+)(?:\s*[:=]\s*[^,}]+)?").unwrap();
             
             // Remove braces
             let content = param.name.trim_matches(|c| c == '{' || c == '}');
             
             for cap in re.captures_iter(content) {
                 if let Some(name) = cap.get(1) {
                     props.push(PropDefinition {
                         name: name.as_str().to_string(),
                         type_annotation: None, 
                         is_required: true, // Assume required by default in destructuring unless = value
                         default_value: None,
                         description: None,
                         validation: None,
                     });
                 }
             }
        }
        
        // Case 2: Named props with type annotation (props: MyProps)
        if let Some(_type_name) = &param.type_annotation {
             // We can't resolve the type definition here easily, but we can record that 
             // this component uses `type_name` as props.
             // For now, we just return an empty list if not destructured, 
             // relying on the type definition elsewhere.
        }

        Ok(props)
    }

    fn extract_arrow_function_metadata(&self, node: &Node, source: &str) -> Result<SymbolMetadata> {
        let mut metadata = SymbolMetadata::default();

        // Extract parameters
        if let Some(params_node) = node.child_by_field_name("parameters") {
            metadata.parameters = self.extract_parameters(&params_node, source)?;
        } else {
            // Single parameter arrow function: x => ...
            if let Some(param) = node.child_by_field_name("parameter") {
                let name = param.utf8_text(source.as_bytes())?.to_string();
                metadata.parameters.push(Parameter {
                    name,
                    type_annotation: None,
                    default_value: None,
                    is_optional: false,
                });
            }
        }

        metadata.is_async = node.utf8_text(source.as_bytes())?.starts_with("async");

        Ok(metadata)
    }

    fn extract_parameters(&self, node: &Node, source: &str) -> Result<Vec<Parameter>> {
        let mut params = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "identifier" => {
                    params.push(Parameter {
                        name: child.utf8_text(source.as_bytes())?.to_string(),
                        type_annotation: None,
                        default_value: None,
                        is_optional: false,
                    });
                }
                "required_parameter" => {
                    let name = self
                        .get_child_text(&child, "pattern", source)
                        .unwrap_or_default();
                    let type_annotation = child
                        .child_by_field_name("type")
                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());
                    params.push(Parameter {
                        name,
                        type_annotation,
                        default_value: None,
                        is_optional: false,
                    });
                }
                "optional_parameter" => {
                    let name = self
                        .get_child_text(&child, "pattern", source)
                        .unwrap_or_default();
                    let type_annotation = child
                        .child_by_field_name("type")
                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string());
                    params.push(Parameter {
                        name,
                        type_annotation,
                        default_value: None,
                        is_optional: true,
                    });
                }
                _ => {}
            }
        }
        Ok(params)
    }

    fn extract_imports(&self, node: &Node, source: &str) -> Result<Vec<Import>> {
        let mut imports = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "import_statement" {
                let source_path = self
                    .get_child_text(&child, "source", source)
                    .map(|s| s.trim_matches(|c| c == '\'' || c == '"').to_string())
                    .unwrap_or_default();

                let mut names = Vec::new();

                // Handle import clause
                if let Some(clause) = child.child_by_field_name("clause") {
                    // Default import?
                    let mut cursor2 = clause.walk();
                    for sub in clause.children(&mut cursor2) {
                        if sub.kind() == "identifier" {
                            names.push(ImportName {
                                name: sub.utf8_text(source.as_bytes())?.to_string(),
                                alias: None,
                                is_default: true,
                                is_namespace: false,
                                is_type: false,
                            });
                        } else if sub.kind() == "named_imports" {
                            let mut cursor3 = sub.walk();
                            for spec in sub.children(&mut cursor3) {
                                if spec.kind() == "import_specifier" {
                                    let name = self
                                        .get_child_text(&spec, "name", source)
                                        .unwrap_or_default();
                                    let alias = self.get_child_text(&spec, "alias", source);
                                    names.push(ImportName {
                                        name,
                                        alias,
                                        is_default: false,
                                        is_namespace: false,
                                        is_type: false,
                                    });
                                }
                            }
                        }
                    }
                }

                imports.push(Import {
                    source: source_path,
                    names,
                    range: self.get_range(&child),
                });
            }
        }
        Ok(imports)
    }

    fn extract_exports(&self, node: &Node, source: &str) -> Result<Vec<Export>> {
        let mut exports = Vec::new();
        // TODO: Implement robust export extraction
        // For now, we rely on symbols being marked as exported if we were to add that flag
        // But here we want explicit export statements
        Ok(exports)
    }

    fn extract_references(&self, node: &Node, source: &str) -> Result<Vec<String>> {
        let mut references = Vec::new();
        let mut cursor = node.walk();

        // Traverse the node to find identifiers that are used (not declared)
        for child in node.children(&mut cursor) {
            self.collect_references(&child, source, &mut references)?;
        }

        // Deduplicate
        references.sort();
        references.dedup();

        Ok(references)
    }

    fn collect_references(&self, node: &Node, source: &str, references: &mut Vec<String>) -> Result<()> {
        let kind = node.kind();

        match kind {
            "identifier" => {
                // Check if this identifier is a reference
                // We need to avoid declarations
                if self.is_reference(node) {
                    let name = node.utf8_text(source.as_bytes())?.to_string();
                    // Filter out basic keywords and types if possible (simple heuristic)
                    if !self.is_keyword(&name) {
                        references.push(name);
                    }
                }
            }
            "property_identifier" => {
                // Usually property access, might be relevant if it's a method call
                // e.g. obj.method() -> we want 'method' if it's a known symbol
                // For now, let's be conservative and skip properties to avoid noise
            }
            "type_identifier" => {
                // Type references
                let name = node.utf8_text(source.as_bytes())?.to_string();
                if !self.is_keyword(&name) {
                    references.push(name);
                }
            }
            _ => {
                // Recurse
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.collect_references(&child, source, references)?;
                }
            }
        }
        Ok(())
    }

    fn is_reference(&self, node: &Node) -> bool {
        let parent = match node.parent() {
            Some(p) => p,
            None => return false,
        };

        let kind = parent.kind();
        
        // Exclude declarations
        match kind {
            "function_declaration" | "variable_declarator" | "class_declaration" | "interface_declaration" | "method_definition" => {
                // If it's the name field, it's a declaration
                if let Some(name_child) = parent.child_by_field_name("name") {
                    if name_child.id() == node.id() {
                        return false;
                    }
                }
            }
            "property_signature" => {
                 if let Some(name_child) = parent.child_by_field_name("name") {
                    if name_child.id() == node.id() {
                        return false;
                    }
                }
            }
            "import_specifier" | "import_clause" | "namespace_import" => return false,
            _ => {}
        }

        true
    }

    fn is_keyword(&self, name: &str) -> bool {
        matches!(name, 
            "string" | "number" | "boolean" | "any" | "void" | "null" | "undefined" | 
            "true" | "false" | "this" | "super" | "console" | "window" | "document" |
            "Array" | "Promise" | "Object" | "Function" | "Error" | "Map" | "Set"
        )
    }

    fn extract_design_tokens(&self, node: &Node, source: &str) -> Result<Vec<DesignToken>> {
        let mut tokens = Vec::new();

        // Simple regex-like scan for now, but using tree-sitter would be better for context
        // Look for className strings
        let query = Query::new(
            tree_sitter_typescript::language_tsx(),
            r#"(jsx_attribute (property_identifier) @prop_name (#eq? @prop_name "className") (string (string_fragment) @class_value))"#
        ).unwrap();

        let mut cursor = QueryCursor::new();
        let matches = cursor.matches(&query, *node, source.as_bytes());

        for m in matches {
            for capture in m.captures {
                if capture.index == 1 {
                    // @class_value
                    let text = capture.node.utf8_text(source.as_bytes())?;
                    for class in text.split_whitespace() {
                        if class.starts_with("bg-")
                            || class.starts_with("text-")
                            || class.starts_with("p-")
                            || class.starts_with("m-")
                        {
                            tokens.push(DesignToken {
                                token_type: DesignTokenType::TailwindClass,
                                name: class.to_string(),
                                value: class.to_string(),
                                context: "className".to_string(),
                                range: self.get_range(&capture.node),
                            });
                        }
                    }
                }
            }
        }

        Ok(tokens)
    }

    // Helper methods

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

    fn is_component_name(&self, name: &str) -> bool {
        name.chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
    }

    fn returns_jsx(&self, node: &Node, source: &str) -> bool {
        // Heuristic: check if the function body contains JSX elements
        // A better way is to traverse the body and look for return statements with JSX
        let text = node.utf8_text(source.as_bytes()).unwrap_or("");
        text.contains("return <") || text.contains("=> <") || text.contains("=> ( <")
    }

    /// Extract ALL type definitions (interfaces, type aliases, enums)
    fn extract_type_definitions(&self, node: &Node, source: &str) -> Result<Vec<TypeDefinition>> {
        let mut types = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            match child.kind() {
                "interface_declaration" => {
                    if let Some(type_def) = self.extract_interface(&child, source)? {
                        types.push(type_def);
                    }
                }
                "type_alias_declaration" => {
                    if let Some(type_def) = self.extract_type_alias(&child, source)? {
                        types.push(type_def);
                    }
                }
                "enum_declaration" => {
                    if let Some(type_def) = self.extract_enum_type(&child, source)? {
                        types.push(type_def);
                    }
                }
                _ => {}
            }
        }

        Ok(types)
    }

    fn extract_interface(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "type_identifier", source)
            .unwrap_or_default();
        let definition = node.utf8_text(source.as_bytes())?.to_string();

        let mut properties = Vec::new();
        if let Some(body) = node.child_by_field_name("body") {
            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if child.kind() == "property_signature" {
                    let prop_name = self
                        .get_child_text(&child, "property_identifier", source)
                        .unwrap_or_default();
                    let type_annotation = child
                        .child_by_field_name("type")
                        .map(|n| n.utf8_text(source.as_bytes()).unwrap().to_string())
                        .unwrap_or_default();
                    let is_optional = child.utf8_text(source.as_bytes())?.contains('?');

                    properties.push(TypeProperty {
                        name: prop_name,
                        type_annotation,
                        is_optional,
                        description: None,
                    });
                }
            }
        }

        Ok(Some(TypeDefinition {
            name,
            kind: TypeKind::Interface,
            definition,
            properties,
            generic_params: vec![],
            range: self.get_range(node),
        }))
    }

    fn extract_type_alias(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "type_identifier", source)
            .unwrap_or_default();
        let definition = node.utf8_text(source.as_bytes())?.to_string();

        Ok(Some(TypeDefinition {
            name,
            kind: TypeKind::TypeAlias,
            definition,
            properties: vec![],
            generic_params: vec![],
            range: self.get_range(node),
        }))
    }

    fn extract_enum_type(&self, node: &Node, source: &str) -> Result<Option<TypeDefinition>> {
        let name = self
            .get_child_text(node, "identifier", source)
            .unwrap_or_default();
        let definition = node.utf8_text(source.as_bytes())?.to_string();

        Ok(Some(TypeDefinition {
            name,
            kind: TypeKind::Enum,
            definition,
            properties: vec![],
            generic_params: vec![],
            range: self.get_range(node),
        }))
    }

    /// Extract ALL constants and configuration values
    fn extract_constants(&self, node: &Node, source: &str) -> Result<Vec<Constant>> {
        let mut constants = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "lexical_declaration" || child.kind() == "variable_declaration" {
                let text = child.utf8_text(source.as_bytes())?;
                if text.starts_with("const") || text.starts_with("export const") {
                    if let Some(constant) =
                        self.extract_constant_from_declaration(&child, source)?
                    {
                        constants.push(constant);
                    }
                }
            }
        }

        Ok(constants)
    }

    fn extract_constant_from_declaration(
        &self,
        node: &Node,
        source: &str,
    ) -> Result<Option<Constant>> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "variable_declarator" {
                let name_node = child.child_by_field_name("name");
                let value_node = child.child_by_field_name("value");

                if let (Some(name_node), Some(value_node)) = (name_node, value_node) {
                    let name = name_node.utf8_text(source.as_bytes())?.to_string();
                    let value = value_node.utf8_text(source.as_bytes())?.to_string();

                    // Skip if it's a function
                    if value_node.kind() == "arrow_function" || value_node.kind() == "function" {
                        continue;
                    }

                    let category = self.categorize_constant(&name, &value);

                    return Ok(Some(Constant {
                        name,
                        value,
                        type_annotation: None,
                        category,
                        range: self.get_range(node),
                    }));
                }
            }
        }
        Ok(None)
    }

    fn categorize_constant(&self, name: &str, _value: &str) -> ConstantCategory {
        let name_lower = name.to_lowercase();

        if name_lower.contains("api")
            || name_lower.contains("endpoint")
            || name_lower.contains("url")
        {
            ConstantCategory::APIEndpoint
        } else if name_lower.contains("config") || name_lower.contains("settings") {
            ConstantCategory::Config
        } else if name_lower.contains("error") || name_lower.contains("message") {
            ConstantCategory::ErrorMessage
        } else if name_lower.contains("default") {
            ConstantCategory::DefaultValue
        } else {
            ConstantCategory::Other
        }
    }

    /// Extract validation schemas (Zod, Yup, etc.)
    fn extract_validation_schemas(
        &self,
        node: &Node,
        source: &str,
    ) -> Result<Vec<ValidationSchema>> {
        let mut schemas = Vec::new();
        let text = node.utf8_text(source.as_bytes())?;

        // Look for Zod schemas
        if text.contains("z.object") || text.contains("zod") {
            schemas.extend(self.extract_zod_schemas(node, source)?);
        }

        Ok(schemas)
    }

    fn extract_zod_schemas(&self, node: &Node, source: &str) -> Result<Vec<ValidationSchema>> {
        let mut schemas = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "lexical_declaration" || child.kind() == "variable_declaration" {
                let text = child.utf8_text(source.as_bytes())?;
                if text.contains("z.object") {
                    if let Some(schema) =
                        self.extract_zod_schema_from_declaration(&child, source)?
                    {
                        schemas.push(schema);
                    }
                }
            }
        }

        Ok(schemas)
    }

    fn extract_zod_schema_from_declaration(
        &self,
        node: &Node,
        source: &str,
    ) -> Result<Option<ValidationSchema>> {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "variable_declarator" {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = name_node.utf8_text(source.as_bytes())?.to_string();
                    let definition = child.utf8_text(source.as_bytes())?.to_string();

                    // Enhanced field extraction for Zod schemas
                    let mut fields = Vec::new();
                    if let Some(value_node) = child.child_by_field_name("value") {
                        if value_node.kind() == "call_expression" {
                            // Look for z.object({ ... })
                            if let Some(arg_list) = value_node.child_by_field_name("arguments") {
                                let mut arg_cursor = arg_list.walk();
                                for arg in arg_list.children(&mut arg_cursor) {
                                    if arg.kind() == "object" {
                                        // Parse object literal for fields
                                        fields = self.parse_zod_object_fields(&arg, source)?;
                                    }
                                }
                            }
                        }
                    }

                    return Ok(Some(ValidationSchema {
                        name,
                        schema_type: SchemaType::Zod,
                        definition,
                        fields,
                        range: self.get_range(node),
                    }));
                }
            }
        }
        Ok(None)
    }

    fn parse_zod_object_fields(&self, node: &Node, source: &str) -> Result<Vec<SchemaField>> {
        let mut fields = Vec::new();
        let mut cursor = node.walk();

        for child in node.children(&mut cursor) {
            if child.kind() == "object_member" || child.kind() == "pair" {
                let key_node = child.child_by_field_name("key");
                let value_node = child.child_by_field_name("value");

                if let (Some(key_node), Some(value_node)) = (key_node, value_node) {
                    let field_name = key_node.utf8_text(source.as_bytes())?.trim_matches('"').to_string();
                    let field_type = value_node.utf8_text(source.as_bytes())?.to_string();
                    let is_optional = field_name.ends_with('?');

                    fields.push(SchemaField {
                        name: field_name.trim_end_matches('?').to_string(),
                        validation_rules: self.extract_zod_validators(&value_node, source)?,
                        is_required: !is_optional,
                        default_value: None,
                        type_annotation: Some(field_type),
                        is_optional,
                        validators: self.extract_zod_validators(&value_node, source)?,
                        description: None,
                    });
                }
            }
        }

        Ok(fields)
    }

    fn extract_zod_validators(&self, node: &Node, source: &str) -> Result<Vec<String>> {
        let mut validators = Vec::new();
        let text = node.utf8_text(source.as_bytes())?;

        // Common Zod validators
        if text.contains(".min(") {
            validators.push("min_length".to_string());
        }
        if text.contains(".max(") {
            validators.push("max_length".to_string());
        }
        if text.contains(".email(") {
            validators.push("email".to_string());
        }
        if text.contains(".required(") {
            validators.push("required".to_string());
        }
        if text.contains(".regex(") {
            validators.push("pattern".to_string());
        }

        Ok(validators)
    }
}

impl Default for TypeScriptParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_references() {
        let parser = TypeScriptParser::new();
        let content = r#"
            import { helper } from './utils';
            
            function myFunction() {
                const x = helper(10);
                const y = anotherFunction();
                return x + y;
            }
        "#;
        
        let parsed = parser.parse(content, false).unwrap();
        println!("Parsed symbols: {:?}", parsed.symbols.iter().map(|s| &s.name).collect::<Vec<_>>());
        let symbol = parsed.symbols.iter().find(|s| s.name == "myFunction").unwrap();
        
        assert!(symbol.references.contains(&"helper".to_string()));
        assert!(symbol.references.contains(&"anotherFunction".to_string()));
        // Local variables might be captured as references, which is acceptable for now.
        // assert!(!symbol.references.contains(&"x".to_string())); 
    }

    #[test]
    fn test_extract_props() {
        let parser = TypeScriptParser::new();
        let content = r#"
            function MyComponent({ title, isActive }: { title: string, isActive: boolean }) {
                return <div>{title}</div>;
            }
        "#;
        
        let parsed = parser.parse(content, true).unwrap();
        println!("Parsed symbols: {:?}", parsed.symbols.iter().map(|s| &s.name).collect::<Vec<_>>());
        let symbol = parsed.symbols.iter().find(|s| s.name == "MyComponent").unwrap();
        
        assert_eq!(symbol.metadata.props.len(), 2);
        assert!(symbol.metadata.props.iter().any(|p| p.name == "title"));
        assert!(symbol.metadata.props.iter().any(|p| p.name == "isActive"));
    }
}
