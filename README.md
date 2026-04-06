# zeroclaw-lite

Minimal tool-calling agent for constrained hardware. <50MB overhead, no persistent storage.

## Features

- **Local LLM via Ollama** or **MiniMax API**
- **3 tools**: shell, file read, file write
- **Security policy**: path/command validation
- **Two dispatcher modes**: native function calling or XML tag parsing
- **No persistent memory** — single-session only

## Quick Start

```bash
# Build
cargo build --release

# Generate default config
./target/release/zeroclaw-lite --init-config

# Run (interactive)
./target/release/zeroclaw-lite

# Run (single message)
./target/release/zeroclaw-lite --message "Hello"
```

## Configuration

Edit `zeroclaw-lite.toml`:

```toml
# Ollama (default)
ollama_url = "http://localhost:11434"
model = "llama3"

# Or MiniMax (uncomment and add API key)
# minimax_api_key = "your-api-key"
# minimax_model = "MiniMax-M2.7"

temperature = 0.7
workspace_dir = "."
max_tool_iterations = 10
use_xml_dispatcher = false
autonomy_level = "supervised"
allowed_dirs = ["."]
```

## Architecture

```
Message → LiteAgent → Provider → Dispatcher → Tools
                                            ↓
                                      Tool Results → (loop)
```

See [CLAUDE.md](CLAUDE.md) for full architecture details.

## Development

```bash
cargo test                # All tests
cargo test --lib          # Unit tests only
cargo fmt --all           # Format
cargo clippy --all-targets -- -D warnings  # Lint
```
