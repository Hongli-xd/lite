//! Ollama provider for zeroclaw-lite.
//!
//! Local LLM provider using the Ollama API. No API key required.

use crate::traits::{
    build_tool_instructions_text, ChatMessage, ChatRequest, ChatResponse, Provider,
    ProviderCapabilities,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Ollama provider configuration
#[derive(Clone)]
pub struct OllamaConfig {
    pub base_url: String,
    pub model: String,
    pub timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
            timeout_secs: 120,
        }
    }
}

/// Ollama provider using local LLM with no API key.
pub struct OllamaProvider {
    config: OllamaConfig,
    client: Client,
}

impl OllamaProvider {
    pub fn new(base_url: Option<&str>, model: Option<&str>) -> Self {
        let timeout_secs = 120u64;
        let config = OllamaConfig {
            base_url: base_url.unwrap_or("http://localhost:11434").to_string(),
            model: model.unwrap_or("llama3").to_string(),
            timeout_secs,
        };
        Self {
            config,
            client: Client::builder()
                .timeout(Duration::from_secs(timeout_secs))
                .build()
                .unwrap_or_default(),
        }
    }

    pub fn with_config(config: OllamaConfig) -> Self {
        Self {
            config: config.clone(),
            client: Client::builder()
                .timeout(Duration::from_secs(config.timeout_secs))
                .build()
                .unwrap_or_default(),
        }
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: &'a [OllamaMessage],
    stream: bool,
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: OllamaMessageResponse,
}

#[derive(Deserialize)]
struct OllamaMessageResponse {
    content: String,
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Ollama supports native tool calling only for specific models
        // that have been fine-tuned for function calling. Default to false
        // to use XML-based tool calling for broad compatibility.
        ProviderCapabilities {
            native_tool_calling: false,
            vision: false,
            prompt_caching: false,
        }
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        _temperature: f64,
    ) -> anyhow::Result<String> {
        let mut messages = Vec::new();

        if let Some(sys) = system_prompt {
            messages.push(OllamaMessage {
                role: "system".to_string(),
                content: sys.to_string(),
            });
        }

        messages.push(OllamaMessage {
            role: "user".to_string(),
            content: message.to_string(),
        });

        let request = OllamaChatRequest {
            model,
            messages: &messages,
            stream: false,
        };

        let url = format!("{}/api/chat", self.config.base_url);
        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("Ollama API error ({status}): {text}");
        }

        let ollama_resp: OllamaResponse = response.json().await?;
        Ok(ollama_resp.message.content)
    }

    async fn chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        // Ollama doesn't have native tool calling, so inject tools into system prompt
        let system_prompt = if let Some(tools) = request.tools {
            if !tools.is_empty() {
                Some(build_tool_instructions_text(tools))
            } else {
                None
            }
        } else {
            None
        };

        let last_user = request
            .messages
            .iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let text = self.chat_with_system(system_prompt.as_deref(), last_user, model, temperature).await?;

        Ok(ChatResponse {
            text: Some(text),
            tool_calls: Vec::new(),
            reasoning_content: None,
        })
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        // Check if Ollama is reachable
        let url = format!("{}/api/tags", self.config.base_url);
        self.client.get(&url).send().await?;
        Ok(())
    }
}

/// Mock provider for testing without Ollama
pub struct MockProvider {
    responses: parking_lot::Mutex<std::collections::VecDeque<String>>,
}

impl MockProvider {
    pub fn new(responses: Vec<String>) -> Self {
        Self { responses: parking_lot::Mutex::new(responses.into()) }
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: false,
            prompt_caching: false,
        }
    }

    async fn chat_with_system(
        &self,
        _system_prompt: Option<&str>,
        _message: &str,
        _model: &str,
        _temperature: f64,
    ) -> anyhow::Result<String> {
        if let Some(response) = self.responses.lock().pop_front() {
            Ok(response)
        } else {
            Ok("done".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_provider_returns_responses() {
        let provider = MockProvider::new(vec!["hello".to_string(), "world".to_string()]);

        let resp1 = provider.chat_with_system(None, "hi", "llama3", 0.7).await.unwrap();
        assert_eq!(resp1, "hello");

        let resp2 = provider.chat_with_system(None, "hi", "llama3", 0.7).await.unwrap();
        assert_eq!(resp2, "world");
    }

    #[tokio::test]
    async fn mock_provider_returns_done_when_empty() {
        let provider = MockProvider::new(vec![]);
        let resp = provider.chat_with_system(None, "hi", "llama3", 0.7).await.unwrap();
        assert_eq!(resp, "done");
    }

    #[test]
    fn ollama_config_default() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama3");
        assert_eq!(config.timeout_secs, 120);
    }

    #[test]
    fn ollama_provider_name() {
        let provider = OllamaProvider::new(None, None);
        assert_eq!(provider.name(), "ollama");
    }

    #[test]
    fn ollama_provider_capabilities() {
        let provider = OllamaProvider::new(None, None);
        let caps = provider.capabilities();
        assert!(!caps.native_tool_calling); // Ollama uses XML dispatcher by default
    }

    #[test]
    fn mock_provider_capabilities() {
        let provider = MockProvider::new(vec![]);
        let caps = provider.capabilities();
        assert!(caps.native_tool_calling);
    }
}

// ─── MiniMax Provider ─────────────────────────────────────────────────────────

/// MiniMax API type: anthropic-messages (Bedeutung) or openai-completions
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub enum MiniMaxApiType {
    /// Anthropic /v1/messages endpoint (Anthropic-compatible)
    AnthropicMessages,
    /// OpenAI /chat/completions endpoint (OpenAI-compatible)
    #[default]
    OpenAiCompletions,
}

/// MiniMax API provider configuration
#[derive(Clone)]
pub struct MiniMaxConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub api_type: MiniMaxApiType,
    pub timeout_secs: u64,
}

impl MiniMaxConfig {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.minimaxi.com/anthropic".to_string(),
            model: "MiniMax-M2.7".to_string(),
            api_type: MiniMaxApiType::AnthropicMessages,
            timeout_secs: 120,
        }
    }

    pub fn with_url_and_model(api_key: &str, base_url: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
            model: model.to_string(),
            api_type: MiniMaxApiType::AnthropicMessages,
            timeout_secs: 120,
        }
    }

    /// Return the API endpoint path based on api_type
    fn api_path(&self) -> &str {
        match self.api_type {
            MiniMaxApiType::AnthropicMessages => "/v1/messages",
            MiniMaxApiType::OpenAiCompletions => "/chat/completions",
        }
    }
}

/// MiniMax API provider
pub struct MiniMaxProvider {
    config: MiniMaxConfig,
    client: Client,
}

impl MiniMaxProvider {
    pub fn new(api_key: &str) -> Self {
        let config = MiniMaxConfig::new(api_key);
        Self::with_config(config)
    }

    pub fn with_config(config: MiniMaxConfig) -> Self {
        Self {
            config: config.clone(),
            client: Client::builder()
                .timeout(Duration::from_secs(config.timeout_secs))
                .build()
                .unwrap_or_default(),
        }
    }
}

// ── Anthropic messages API types ────────────────────────────────────────────────

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: AnthropicContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    /// Content as array of blocks: [{"type": "text", "text": "..."}]
    Blocks(Vec<AnthropicContentBlock>),
    /// Content as plain string: "Hello..."
    PlainText(String),
    /// Content as object with text field: {"type": "text", "text": "..."}
    ContentObj { text: String },
}

#[derive(Deserialize)]
#[serde(tag = "type", content = "text")]
enum AnthropicContentBlock {
    Text { text: String },
    #[serde(other)]
    Other,
}

/// Extract text from AnthropicContent, handling multiple response formats
fn extract_anthropic_text(content: &AnthropicContent) -> String {
    match content {
        AnthropicContent::Blocks(blocks) => {
            blocks.iter()
                .filter_map(|b| match b {
                    AnthropicContentBlock::Text { text } => Some(text.clone()),
                    AnthropicContentBlock::Other => None,
                })
                .collect::<Vec<_>>()
                .join("")
        }
        AnthropicContent::PlainText(s) => s.clone(),
        AnthropicContent::ContentObj { text } => text.clone(),
    }
}

// ── OpenAI completions API types ───────────────────────────────────────────────

#[derive(Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

#[async_trait]
impl Provider for MiniMaxProvider {
    fn name(&self) -> &str {
        "minimax"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            vision: false,
            prompt_caching: false,
        }
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let temperature = if temperature == 0.0 { None } else { Some(temperature) };

        match self.config.api_type {
            MiniMaxApiType::AnthropicMessages => {
                let mut messages = Vec::new();
                messages.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: message.to_string(),
                });

                let request = AnthropicRequest {
                    model: model.to_string(),
                    messages,
                    max_tokens: 8192,
                    temperature,
                    system: system_prompt.map(String::from),
                };

                let url = format!("{}{}", self.config.base_url.trim_end_matches('/'), self.config.api_path());
                let response = self.client
                    .post(&url)
                    .header("x-api-key", &self.config.api_key)
                    .header("Content-Type", "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .json(&request)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    anyhow::bail!("MiniMax API error ({status}): {text}");
                }

                let body = response.text().await?;
                eprintln!("[DEBUG] raw response ({} chars): {}", body.len(), &body[..body.len().min(500)]);

                let resp: AnthropicResponse = serde_json::from_str(&body)?;
                let text = extract_anthropic_text(&resp.content);
                Ok(text)
            }
            MiniMaxApiType::OpenAiCompletions => {
                let mut messages = Vec::new();
                if let Some(sys) = system_prompt {
                    messages.push(OpenAiMessage { role: "system".to_string(), content: sys.to_string() });
                }
                messages.push(OpenAiMessage { role: "user".to_string(), content: message.to_string() });

                let request = OpenAiRequest {
                    model: model.to_string(),
                    messages,
                    stream: false,
                    max_tokens: Some(8192),
                    temperature,
                };

                let url = format!("{}{}", self.config.base_url.trim_end_matches('/'), self.config.api_path());
                let response = self.client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", self.config.api_key))
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    anyhow::bail!("MiniMax API error ({status}): {text}");
                }

                let resp: OpenAiResponse = response.json().await?;
                if let Some(choice) = resp.choices.first() {
                    Ok(choice.message.content.clone())
                } else {
                    anyhow::bail!("MiniMax API returned no choices")
                }
            }
        }
    }

    async fn chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<ChatResponse> {
        let temperature = if temperature == 0.0 { None } else { Some(temperature) };

        match self.config.api_type {
            MiniMaxApiType::AnthropicMessages => {
                let mut messages = Vec::new();
                let system_prompt: Option<String>;

                if let Some(tools) = request.tools {
                    if !tools.is_empty() && !self.supports_native_tools() {
                        let tool_instructions = build_tool_instructions_text(tools);
                        let mut msgs = request.messages.to_vec();
                        if let Some(sys) = msgs.iter_mut().find(|m| m.role == "system") {
                            if !sys.content.is_empty() { sys.content.push_str("\n\n"); }
                            sys.content.push_str(&tool_instructions);
                        } else {
                            msgs.insert(0, ChatMessage::system(tool_instructions));
                        }
                        system_prompt = None;
                        for msg in msgs {
                            messages.push(AnthropicMessage { role: msg.role.clone(), content: msg.content.clone() });
                        }
                    } else {
                        system_prompt = None;
                        for msg in request.messages {
                            messages.push(AnthropicMessage { role: msg.role.clone(), content: msg.content.clone() });
                        }
                    }
                } else {
                    system_prompt = None;
                    for msg in request.messages {
                        messages.push(AnthropicMessage { role: msg.role.clone(), content: msg.content.clone() });
                    }
                }

                let anthropic_request = AnthropicRequest {
                    model: model.to_string(),
                    messages,
                    max_tokens: 8192,
                    temperature,
                    system: system_prompt,
                };

                let url = format!("{}{}", self.config.base_url.trim_end_matches('/'), self.config.api_path());
                let response = self.client
                    .post(&url)
                    .header("x-api-key", &self.config.api_key)
                    .header("Content-Type", "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .json(&anthropic_request)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    anyhow::bail!("MiniMax API error ({status}): {text}");
                }

                let resp: AnthropicResponse = response.json().await?;
                let text = extract_anthropic_text(&resp.content);

                Ok(ChatResponse {
                    text: Some(text),
                    tool_calls: Vec::new(),
                    reasoning_content: None,
                })
            }
            MiniMaxApiType::OpenAiCompletions => {
                let mut messages = Vec::new();
                for msg in request.messages {
                    messages.push(OpenAiMessage { role: msg.role.clone(), content: msg.content.clone() });
                }

                let openai_request = OpenAiRequest {
                    model: model.to_string(),
                    messages,
                    stream: false,
                    max_tokens: Some(8192),
                    temperature,
                };

                let url = format!("{}{}", self.config.base_url.trim_end_matches('/'), self.config.api_path());
                let response = self.client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", self.config.api_key))
                    .header("Content-Type", "application/json")
                    .json(&openai_request)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    anyhow::bail!("MiniMax API error ({status}): {text}");
                }

                let resp: OpenAiResponse = response.json().await?;
                if let Some(choice) = resp.choices.first() {
                    Ok(ChatResponse {
                        text: Some(choice.message.content.clone()),
                        tool_calls: Vec::new(),
                        reasoning_content: None,
                    })
                } else {
                    Ok(ChatResponse { text: Some(String::new()), tool_calls: Vec::new(), reasoning_content: None })
                }
            }
        }
    }

    async fn warmup(&self) -> anyhow::Result<()> {
        // Try models endpoint; gracefully ignore 404 (some providers don't expose it)
        let url = format!("{}/v1/models", self.config.base_url.trim_end_matches('/'));
        match self.client.get(&url)
            .header("x-api-key", &self.config.api_key)
            .send()
            .await {
            Ok(resp) if resp.status() == reqwest::StatusCode::NOT_FOUND => {}
            Ok(_) => {}
            Err(e) => {
                // Log but don't fail warmup
                eprintln!("warmup warning: {}", e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod minimax_tests {
    use super::*;

    #[test]
    fn minimax_config_default() {
        let config = MiniMaxConfig::new("test-key");
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.minimaxi.com/anthropic");
        assert_eq!(config.model, "MiniMax-M2.7");
        assert_eq!(config.api_type, MiniMaxApiType::AnthropicMessages);
        assert_eq!(config.timeout_secs, 120);
    }

    #[test]
    fn minimax_provider_name() {
        let provider = MiniMaxProvider::new("test-key");
        assert_eq!(provider.name(), "minimax");
    }

    #[test]
    fn minimax_provider_capabilities() {
        let provider = MiniMaxProvider::new("test-key");
        let caps = provider.capabilities();
        assert!(caps.native_tool_calling);
        assert!(!caps.vision);
    }
}
