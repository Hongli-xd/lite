//! ZeroClaw Lite — Minimal tool-calling agent for constrained hardware.
//!
//! This binary provides a stripped-down agent suitable for resource-constrained
//! environments. It uses:
//! - **OllamaProvider** — local LLM, no external API calls
//! - **default_tools** (3 tools) — shell, file read, file write
//! - **NoneMemory** — zero persistent storage
//!
//! No gateway, channels, skills, cron, or MCP integrations are loaded.
//!
//! # Configuration
//!
//! Configuration is loaded from `zeroclaw-lite.toml` in the following order:
//! 1. `./zeroclaw-lite.toml` (local, highest priority)
//! 2. `~/.zeroclaw-lite.toml` (home directory)
//! 3. CLI arguments (highest priority)
//!
//! # Example Config File
//!
//! ```toml
//! ollama_url = "http://localhost:11434"
//! model = "llama3"
//! temperature = 0.7
//! workspace_dir = "."
//! max_tool_iterations = 10
//! use_xml_dispatcher = false
//! autonomy_level = "supervised"
//! allowed_dirs = ["."]
//! ```

use anyhow::Result;
use clap::Parser;
use zeroclaw_lite::{
    config::{find_config_file, load_config_file, ResolvedConfig},
    LiteAgentBuilder, MiniMaxProvider, OllamaProvider,
};

// ─── CLI Interface ────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "zeroclaw-lite")]
#[command(about = "Minimal ZeroClaw agent for constrained hardware")]
struct Cli {
    /// Ollama base URL
    #[arg(long)]
    ollama_url: Option<String>,

    /// Model name
    #[arg(long)]
    model: Option<String>,

    /// Temperature for responses
    #[arg(long)]
    temperature: Option<f64>,

    /// Workspace directory
    #[arg(long)]
    workspace_dir: Option<String>,

    /// Maximum tool iterations per turn
    #[arg(long)]
    max_tool_iterations: Option<usize>,

    /// Use XML dispatcher instead of native (for models without function calling)
    #[arg(long)]
    use_xml_dispatcher: bool,

    /// Single message to process (if not set, runs interactive mode)
    #[arg(long)]
    message: Option<String>,

    /// Generate a default config file at ./zeroclaw-lite.toml
    #[arg(long)]
    init_config: bool,

    /// Path to config file (default: ./zeroclaw-lite.toml or ~/.zeroclaw-lite.toml)
    #[arg(long)]
    config: Option<String>,
}

async fn run_interactive(agent: &mut zeroclaw_lite::LiteAgent) -> Result<()> {
    println!("ZeroClaw Lite — Interactive Mode");
    println!("================================");
    println!("Type '/quit' or Ctrl-D to exit.\n");

    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        let bytes_read = std::io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" {
            break;
        }

        match agent.turn(input).await {
            Ok(response) => {
                println!("\n{response}\n");
            }
            Err(e) => {
                eprintln!("\nError: {e}\n");
            }
        }
    }

    Ok(())
}

fn load_config(cli: &Cli) -> Result<ResolvedConfig> {
    let default = ResolvedConfig::default();

    // Try to find config file
    let file_config = if let Some(ref path) = cli.config {
        Some(load_config_file(&std::path::PathBuf::from(path))?)
    } else if let Some(path) = find_config_file() {
        Some(load_config_file(&path)?)
    } else {
        None
    };

    let file_config = file_config.unwrap_or_default();

    // Merge with CLI overrides
    let resolved = default.merge(
        &file_config,
        cli.ollama_url.as_deref(),
        cli.model.as_deref(),
        cli.temperature,
        cli.workspace_dir.as_deref(),
        cli.max_tool_iterations,
        cli.use_xml_dispatcher,
    );

    Ok(resolved)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .init();

    let cli = Cli::parse();

    // Handle --init-config
    if cli.init_config {
        let path = if let Some(ref p) = cli.config {
            std::path::PathBuf::from(p)
        } else {
            std::path::PathBuf::from("./zeroclaw-lite.toml")
        };

        if path.exists() {
            println!("Config file already exists: {}", path.display());
        } else {
            zeroclaw_lite::config::create_default_config(&path)?;
            println!("Created default config: {}", path.display());
        }
        return Ok(());
    }

    // Load merged config
    let config = load_config(&cli)?;

    println!("ZeroClaw Lite");
    println!("=============");

    // Determine provider: MiniMax if API key is set, otherwise Ollama
    let provider_name = if let Some(ref api_key) = config.minimax_api_key {
        println!("Provider: MiniMax ({})", config.minimax_model);
        println!("API: {}", config.minimax_url);
        println!("Model: {}", config.minimax_model);
        "minimax"
    } else {
        println!("Provider: Ollama ({}):", config.ollama_url);
        println!("Model: {}", config.model);
        "ollama"
    };
    println!("Workspace: {}", config.workspace_dir);
    println!();

    // Build the agent with appropriate provider
    let mut agent = match provider_name {
        "minimax" => {
            let api_key = config.minimax_api_key.as_ref().unwrap();
            let minimax_config = zeroclaw_lite::MiniMaxConfig::with_url_and_model(
                api_key,
                &config.minimax_url,
                &config.minimax_model,
            );
            LiteAgentBuilder::new()
                .provider(Box::new(MiniMaxProvider::with_config(minimax_config)))
                .model_name(&config.minimax_model)
                .temperature(config.temperature)
                .workspace_dir(&config.workspace_dir)
                .config(zeroclaw_lite::AgentConfig {
                    max_tool_iterations: config.max_tool_iterations,
                    autonomy_level: config.autonomy_level,
                })
                .build()?
        }
        _ => {
            if config.use_xml_dispatcher {
                LiteAgentBuilder::new()
                    .provider(Box::new(OllamaProvider::new(
                        Some(&config.ollama_url),
                        Some(&config.model),
                    )))
                    .model_name(&config.model)
                    .temperature(config.temperature)
                    .workspace_dir(&config.workspace_dir)
                    .config(zeroclaw_lite::AgentConfig {
                        max_tool_iterations: config.max_tool_iterations,
                        autonomy_level: config.autonomy_level,
                    })
                    .use_xml_dispatcher()
                    .build()?
            } else {
                LiteAgentBuilder::new()
                    .provider(Box::new(OllamaProvider::new(
                        Some(&config.ollama_url),
                        Some(&config.model),
                    )))
                    .model_name(&config.model)
                    .temperature(config.temperature)
                    .workspace_dir(&config.workspace_dir)
                    .config(zeroclaw_lite::AgentConfig {
                        max_tool_iterations: config.max_tool_iterations,
                        autonomy_level: config.autonomy_level,
                    })
                    .build()?
            }
        }
    };

    if let Some(msg) = cli.message {
        // Single message mode
        let response = agent.turn(&msg).await?;
        println!("{response}");
    } else {
        // Interactive mode
        run_interactive(&mut agent).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_from_empty() {
        // When no config file exists and no CLI args, should use defaults
        let cli = Cli::parse_from(["zeroclaw-lite"]);
        assert!(cli.ollama_url.is_none());
        assert!(cli.model.is_none());
        assert!(cli.message.is_none());
        assert!(!cli.init_config);
    }

    #[test]
    fn cli_with_custom_values() {
        let cli = Cli::parse_from([
            "zeroclaw-lite",
            "--ollama-url", "http://custom:11434",
            "--model", "codellama",
            "--temperature", "0.5",
            "--workspace-dir", "/tmp",
            "--max-tool-iterations", "5",
            "--use-xml-dispatcher",
            "--message", "Hello world",
        ]);
        assert_eq!(cli.ollama_url, Some("http://custom:11434".to_string()));
        assert_eq!(cli.model, Some("codellama".to_string()));
        assert_eq!(cli.temperature, Some(0.5));
        assert_eq!(cli.workspace_dir, Some("/tmp".to_string()));
        assert_eq!(cli.max_tool_iterations, Some(5));
        assert_eq!(cli.use_xml_dispatcher, true);
        assert_eq!(cli.message, Some("Hello world".to_string()));
    }
}
