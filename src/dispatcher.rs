//! Tool dispatcher implementations for zeroclaw-lite.

use crate::traits::{
    ChatMessage, ChatResponse, ConversationMessage, ParsedToolCall, ToolDispatcher,
    ToolExecutionResult,
};
use std::fmt::Write as FmtWrite;

/// XML-based tool dispatcher that parses `<tool_call>` tags from plain text responses.
/// Used for providers that don't support native function calling.
pub struct XmlToolDispatcher;

impl XmlToolDispatcher {
    /// Parse tool calls from XML-style tags in text.
    fn parse_xml_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>) {
        let cleaned = Self::strip_think_tags(text);
        let mut text_parts = Vec::new();
        let mut calls = Vec::new();
        let mut remaining = cleaned.as_str();

        while let Some(start) = remaining.find("<tool_call>") {
            let before = &remaining[..start];
            if !before.trim().is_empty() {
                text_parts.push(before.trim().to_string());
            }

            if let Some(end) = remaining[start..].find("</tool_call>") {
                let inner = &remaining[start + 11..start + end];
                match serde_json::from_str::<serde_json::Value>(inner.trim()) {
                    Ok(parsed) => {
                        let name = parsed
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if !name.is_empty() {
                            let arguments = parsed
                                .get("arguments")
                                .cloned()
                                .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
                            calls.push(ParsedToolCall {
                                name,
                                arguments,
                                tool_call_id: None,
                            });
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Malformed <tool_call> JSON: {e}");
                    }
                }
                remaining = &remaining[start + end + 12..];
            } else {
                break;
            }
        }

        if !remaining.trim().is_empty() {
            text_parts.push(remaining.trim().to_string());
        }

        (text_parts.join("\n"), calls)
    }

    /// Remove `<think>...</think>` blocks from model output.
    fn strip_think_tags(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        let mut rest = s;
        loop {
            if let Some(start) = rest.find("<think>") {
                result.push_str(&rest[..start]);
                if let Some(end) = rest[start..].find("</think>") {
                    rest = &rest[start + end + 6..];
                } else {
                    break;
                }
            } else {
                result.push_str(rest);
                break;
            }
        }
        result
    }

    pub fn tool_specs(tools: &[Box<dyn crate::traits::Tool>]) -> Vec<crate::traits::ToolSpec> {
        tools.iter().map(|tool| tool.spec()).collect()
    }
}

impl ToolDispatcher for XmlToolDispatcher {
    fn parse_response(&self, response: &ChatResponse) -> (String, Vec<ParsedToolCall>) {
        let text = response.text_or_empty();
        Self::parse_xml_tool_calls(text)
    }

    fn format_results(&self, results: &[ToolExecutionResult]) -> ConversationMessage {
        let mut content = String::new();
        for result in results {
            let status = if result.success { "ok" } else { "error" };
            let _ = writeln!(
                content,
                "<tool_result name=\"{}\" status=\"{}\">\n{}\n</tool_result>",
                result.name, status, result.output
            );
        }
        ConversationMessage::Chat(ChatMessage::user(format!("[Tool results]\n{content}")))
    }

    fn prompt_instructions(&self, _tools: &[Box<dyn crate::traits::Tool>]) -> String {
        let mut instructions = String::new();
        instructions.push_str("## Tool Use Protocol\n\n");
        instructions.push_str("To use a tool, respond with JSON in <tool_call></tool_call> tags:\n\n");
        instructions.push_str("<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n</tool_call>\n\n");
        instructions.push_str("You may call multiple tools in one response.\n");
        instructions
    }
}

/// Native tool dispatcher for providers with native function calling support.
pub struct NativeToolDispatcher;

impl ToolDispatcher for NativeToolDispatcher {
    fn parse_response(&self, response: &ChatResponse) -> (String, Vec<ParsedToolCall>) {
        let text = response.text_or_empty().to_string();
        let calls = response.tool_calls.iter().map(|tc| ParsedToolCall {
            name: tc.name.clone(),
            arguments: serde_json::from_str(&tc.arguments).unwrap_or_default(),
            tool_call_id: Some(tc.id.clone()),
        }).collect();
        (text, calls)
    }

    fn format_results(&self, results: &[ToolExecutionResult]) -> ConversationMessage {
        let content = results.iter()
            .map(|r| {
                if r.success {
                    format!("{}: {}", r.name, r.output)
                } else {
                    format!("{}: ERROR: {}", r.name, r.output)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        ConversationMessage::Chat(ChatMessage::user(format!("[Tool results]\n{content}")))
    }

    fn prompt_instructions(&self, _tools: &[Box<dyn crate::traits::Tool>]) -> String {
        String::new()
    }
}

/// Get the default tools payload format instructions for the dispatcher.
pub fn default_prompt_addenda(dispatcher: &dyn ToolDispatcher, tools: &[Box<dyn crate::traits::Tool>]) -> String {
    dispatcher.prompt_instructions(tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xml_dispatcher_parses_single_tool_call() {
        let dispatcher = XmlToolDispatcher;
        let text = "Let me check that for you.\n<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>\nI'll run the command.";

        let response = ChatResponse::text_response(text);
        let (clean_text, calls) = dispatcher.parse_response(&response);

        assert!(clean_text.contains("Let me check"));
        assert!(clean_text.contains("I'll run the command"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn xml_dispatcher_parses_multiple_tool_calls() {
        let dispatcher = XmlToolDispatcher;
        let text = "<tool_call>{\"name\": \"tool1\", \"arguments\": {}}</tool_call>\n<tool_call>{\"name\": \"tool2\", \"arguments\": {}}</tool_call>";

        let response = ChatResponse::text_response(text);
        let (_, calls) = dispatcher.parse_response(&response);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "tool1");
        assert_eq!(calls[1].name, "tool2");
    }

    #[test]
    fn xml_dispatcher_strips_think_tags() {
        let dispatcher = XmlToolDispatcher;
        let text = "<think> I should use a tool here</think>Some thought<tool_call>{\"name\": \"test\", \"arguments\": {}}</tool_call>";

        let response = ChatResponse::text_response(text);
        let (_, calls) = dispatcher.parse_response(&response);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "test");
    }

    #[test]
    fn xml_dispatcher_handles_malformed_json() {
        let dispatcher = XmlToolDispatcher;
        let text = "<tool_call>not valid json</tool_call>";

        let response = ChatResponse::text_response(text);
        let (_, calls) = dispatcher.parse_response(&response);

        assert!(calls.is_empty());
    }

    #[test]
    fn xml_dispatcher_handles_empty_name() {
        let dispatcher = XmlToolDispatcher;
        let text = "<tool_call>{\"name\": \"\", \"arguments\": {}}</tool_call>";

        let response = ChatResponse::text_response(text);
        let (_, calls) = dispatcher.parse_response(&response);

        assert!(calls.is_empty());
    }

    #[test]
    fn xml_dispatcher_no_tool_calls() {
        let dispatcher = XmlToolDispatcher;
        let text = "Just a regular response without any tools.";

        let response = ChatResponse::text_response(text);
        let (clean_text, calls) = dispatcher.parse_response(&response);

        assert_eq!(clean_text, text);
        assert!(calls.is_empty());
    }

    #[test]
    fn xml_dispatcher_format_results() {
        let dispatcher = XmlToolDispatcher;
        let results = vec![
            ToolExecutionResult { name: "shell".into(), output: "files listed".into(), success: true },
            ToolExecutionResult { name: "file_read".into(), output: "file contents".into(), success: true },
        ];

        let msg = dispatcher.format_results(&results);
        let content = match msg {
            ConversationMessage::Chat(c) => c.content,
            _ => panic!("expected Chat variant"),
        };

        assert!(content.contains("shell"));
        assert!(content.contains("files listed"));
        assert!(content.contains("file_read"));
        assert!(content.contains("file contents"));
    }

    #[test]
    fn xml_dispatcher_format_results_with_error() {
        let dispatcher = XmlToolDispatcher;
        let results = vec![
            ToolExecutionResult { name: "shell".into(), output: "command failed".into(), success: false },
        ];

        let msg = dispatcher.format_results(&results);
        let content = match msg {
            ConversationMessage::Chat(c) => c.content,
            _ => panic!("expected Chat variant"),
        };

        assert!(content.contains("error"));
    }

    #[test]
    fn native_dispatcher_parses_tool_calls() {
        let dispatcher = NativeToolDispatcher;
        let response = ChatResponse {
            text: Some("I'll check that.".into()),
            tool_calls: vec![
                crate::traits::ToolCall {
                    id: "call_1".into(),
                    name: "shell".into(),
                    arguments: r#"{"command": "ls"}"#.into(),
                },
            ],
            reasoning_content: None,
        };

        let (text, calls) = dispatcher.parse_response(&response);

        assert_eq!(text, "I'll check that.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls");
        assert_eq!(calls[0].tool_call_id, Some("call_1".into()));
    }

    #[test]
    fn native_dispatcher_format_results() {
        let dispatcher = NativeToolDispatcher;
        let results = vec![
            ToolExecutionResult { name: "test".into(), output: "ok".into(), success: true },
        ];

        let msg = dispatcher.format_results(&results);
        let content = match msg {
            ConversationMessage::Chat(c) => c.content,
            _ => panic!("expected Chat variant"),
        };

        assert!(content.contains("test: ok"));
    }
}
