//! Configuration for zeroclaw-lite.
//!
//! Supports loading from TOML config file and environment variables.
//! Config file location: `~/.zeroclaw-lite.toml` or `./zeroclaw-lite.toml`

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Autonomy level for agent operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomyLevel {
    /// Read-only mode - no tool execution
    ReadOnly,
    /// Supervised mode - requires approval for destructive operations
    Supervised,
    /// Full autonomy - executes all approved tools
    Full,
}

impl Default for AutonomyLevel {
    fn default() -> Self {
        Self::Supervised
    }
}

impl std::fmt::Display for AutonomyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnly => write!(f, "read_only"),
            Self::Supervised => write!(f, "supervised"),
            Self::Full => write!(f, "full"),
        }
    }
}

/// Configuration for zeroclaw-lite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteConfig {
    /// Ollama base URL (default: "http://localhost:11434")
    pub ollama_url: Option<String>,
    /// Model name to use (default: "llama3")
    pub model: Option<String>,
    /// Temperature for LLM responses (default: 0.7)
    pub temperature: Option<f64>,
    /// Workspace directory for file operations (default: ".")
    pub workspace_dir: Option<String>,
    /// Maximum tool-call iterations per turn (default: 10)
    pub max_tool_iterations: Option<usize>,
    /// Use XmlToolDispatcher instead of NativeToolDispatcher (default: false)
    pub use_xml_dispatcher: Option<bool>,
    /// Autonomy level (default: supervised)
    pub autonomy_level: Option<String>,
    /// Allowed directories for file operations (default: ["."])
    pub allowed_dirs: Option<Vec<String>>,
    /// Security: allow script execution in skills (default: false)
    pub allow_scripts: Option<bool>,

    // MiniMax API 配置
    /// MiniMax API Key (若设置则使用 MiniMax provider)
    pub minimax_api_key: Option<String>,
    /// MiniMax base URL
    pub minimax_url: Option<String>,
    /// MiniMax 模型名称
    pub minimax_model: Option<String>,
}

impl Default for LiteConfig {
    fn default() -> Self {
        Self {
            ollama_url: None,
            model: None,
            temperature: None,
            workspace_dir: None,
            max_tool_iterations: None,
            use_xml_dispatcher: None,
            autonomy_level: None,
            allowed_dirs: None,
            allow_scripts: None,
            minimax_api_key: None,
            minimax_url: None,
            minimax_model: None,
        }
    }
}

/// Resolved configuration with all defaults applied
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub ollama_url: String,
    pub model: String,
    pub temperature: f64,
    pub workspace_dir: String,
    pub max_tool_iterations: usize,
    pub use_xml_dispatcher: bool,
    pub autonomy_level: AutonomyLevel,
    pub allowed_dirs: Vec<String>,
    // MiniMax
    pub minimax_api_key: Option<String>,
    pub minimax_url: String,
    pub minimax_model: String,
}

impl Default for ResolvedConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".into(),
            model: "llama3".into(),
            temperature: 0.7,
            workspace_dir: ".".into(),
            max_tool_iterations: 10,
            use_xml_dispatcher: false,
            autonomy_level: AutonomyLevel::Supervised,
            allowed_dirs: vec![".".to_string()],
            minimax_api_key: None,
            minimax_url: "https://api.minimaxi.com/anthropic".into(),
            minimax_model: "MiniMax-M2.7".into(),
        }
    }
}

impl ResolvedConfig {
    /// Merge config file values with CLI overrides
    pub fn merge(&self, file_config: &LiteConfig, cli_ollama_url: Option<&str>, cli_model: Option<&str>, cli_temp: Option<f64>, cli_workspace: Option<&str>, cli_max_iter: Option<usize>, cli_xml: bool) -> Self {
        Self {
            ollama_url: cli_ollama_url.map(String::from)
                .or_else(|| file_config.ollama_url.clone())
                .unwrap_or_else(|| self.ollama_url.clone()),
            model: cli_model.map(String::from)
                .or_else(|| file_config.model.clone())
                .unwrap_or_else(|| self.model.clone()),
            temperature: cli_temp
                .or(file_config.temperature)
                .unwrap_or(self.temperature),
            workspace_dir: cli_workspace.map(String::from)
                .or_else(|| file_config.workspace_dir.clone())
                .unwrap_or_else(|| self.workspace_dir.clone()),
            max_tool_iterations: cli_max_iter
                .or(file_config.max_tool_iterations)
                .unwrap_or(self.max_tool_iterations),
            use_xml_dispatcher: if cli_xml {
                true
            } else {
                file_config.use_xml_dispatcher.unwrap_or(self.use_xml_dispatcher)
            },
            autonomy_level: file_config.autonomy_level
                .as_ref()
                .and_then(|s| match s.to_lowercase().as_str() {
                    "read_only" | "readonly" => Some(AutonomyLevel::ReadOnly),
                    "full" => Some(AutonomyLevel::Full),
                    _ => Some(AutonomyLevel::Supervised),
                })
                .unwrap_or(self.autonomy_level),
            allowed_dirs: file_config.allowed_dirs.clone()
                .unwrap_or_else(|| self.allowed_dirs.clone()),
            minimax_api_key: file_config.minimax_api_key.clone(),
            minimax_url: file_config.minimax_url.clone()
                .unwrap_or_else(|| self.minimax_url.clone()),
            minimax_model: file_config.minimax_model.clone()
                .unwrap_or_else(|| self.minimax_model.clone()),
        }
    }
}

/// Agent runtime configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_tool_iterations: usize,
    pub autonomy_level: AutonomyLevel,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_iterations: 10,
            autonomy_level: AutonomyLevel::Supervised,
        }
    }
}

/// Find the config file path
pub fn find_config_file() -> Option<PathBuf> {
    // Check ./zeroclaw-lite.toml first
    let local = PathBuf::from("./zeroclaw-lite.toml");
    if local.exists() {
        return Some(local);
    }

    // Check ~/.zeroclaw-lite.toml
    if let Some(dirs) = directories::UserDirs::new() {
        let home_config = dirs.home_dir().join(".zeroclaw-lite.toml");
        if home_config.exists() {
            return Some(home_config);
        }
    }

    None
}

/// Load config from file
pub fn load_config_file(path: &PathBuf) -> anyhow::Result<LiteConfig> {
    let content = std::fs::read_to_string(path)?;
    let config: LiteConfig = toml::from_str(&content)?;
    Ok(config)
}

/// Create a default config file at the given path
pub fn create_default_config(path: &PathBuf) -> anyhow::Result<()> {
    let default = LiteConfig {
        ollama_url: Some("http://localhost:11434".into()),
        model: Some("llama3".into()),
        temperature: Some(0.7),
        workspace_dir: Some(".".into()),
        max_tool_iterations: Some(10),
        use_xml_dispatcher: Some(false),
        autonomy_level: Some("supervised".into()),
        allowed_dirs: Some(vec![".".to_string()]),
        allow_scripts: Some(false),
        minimax_api_key: None,
        minimax_url: Some("https://api.minimaxi.com/anthropic".into()),
        minimax_model: Some("MiniMax-M2.7".into()),
    };

    let content = toml::to_string_pretty(&default)?;
    std::fs::write(path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lite_config_default() {
        let config = LiteConfig::default();
        assert!(config.ollama_url.is_none());
        assert!(config.model.is_none());
    }

    #[test]
    fn autonomy_level_default() {
        assert_eq!(AutonomyLevel::default(), AutonomyLevel::Supervised);
    }

    #[test]
    fn autonomy_level_display() {
        assert_eq!(AutonomyLevel::ReadOnly.to_string(), "read_only");
        assert_eq!(AutonomyLevel::Supervised.to_string(), "supervised");
        assert_eq!(AutonomyLevel::Full.to_string(), "full");
    }

    #[test]
    fn resolved_config_default() {
        let config = ResolvedConfig::default();
        assert_eq!(config.ollama_url, "http://localhost:11434");
        assert_eq!(config.model, "llama3");
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.max_tool_iterations, 10);
    }

    #[test]
    fn resolved_config_merge_prefers_cli_over_file() {
        let default = ResolvedConfig::default();
        let file = LiteConfig {
            ollama_url: Some("http://file:11434".into()),
            model: Some("file-model".into()),
            ..Default::default()
        };

        let merged = default.merge(&file,
            Some("http://cli:11434"),
            Some("cli-model"),
            Some(0.9),
            None, None, false
        );

        assert_eq!(merged.ollama_url, "http://cli:11434");
        assert_eq!(merged.model, "cli-model");
        assert_eq!(merged.temperature, 0.9);
    }

    #[test]
    fn resolved_config_merges_file_when_no_cli_override() {
        let default = ResolvedConfig::default();
        let file = LiteConfig {
            ollama_url: Some("http://file:11434".into()),
            model: Some("file-model".into()),
            temperature: Some(0.5),
            ..Default::default()
        };

        let merged = default.merge(&file, None, None, None, None, None, false);

        assert_eq!(merged.ollama_url, "http://file:11434");
        assert_eq!(merged.model, "file-model");
        assert_eq!(merged.temperature, 0.5);
    }

    #[test]
    fn lite_config_serde_roundtrip() {
        let config = LiteConfig {
            ollama_url: Some("http://localhost:11434".into()),
            model: Some("llama3".into()),
            temperature: Some(0.7),
            workspace_dir: Some(".".into()),
            max_tool_iterations: Some(10),
            use_xml_dispatcher: Some(false),
            autonomy_level: Some("supervised".into()),
            allowed_dirs: Some(vec![".".to_string()]),
            allow_scripts: Some(false),
            minimax_api_key: None,
            minimax_url: None,
            minimax_model: None,
        };

        let toml_str = toml::to_string(&config).unwrap();
        let parsed: LiteConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.ollama_url, config.ollama_url);
        assert_eq!(parsed.model, config.model);
    }
}
