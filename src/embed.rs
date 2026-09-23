//! Embedding providers behind one trait.
//!
//! [`FastembedProvider`] runs a tiny ONNX model in-process (default path,
//! no server needed). [`HttpProvider`] speaks OpenAI-compatible
//! `/v1/embeddings` for a future MLX embedding lane; the current
//! `mlx-lm.server` lanes have no such endpoint (verified 404).

use std::sync::Mutex;

use rmcp::schemars;

use crate::Error;

/// Maps texts to dense vectors. Implementations must be thread-safe.
pub trait EmbedProvider: Send + Sync {
    /// Embed each text, preserving order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the backend fails.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error>;

    /// Vector dimensionality.
    fn dims(&self) -> usize;

    /// Stable model label recorded in the vector store.
    fn name(&self) -> &str;

    /// Query-side prefix (documents are embedded raw). Snowflake arctic
    /// models expect the BGE-style retrieval prefix; others use none.
    fn query_prefix(&self) -> Option<&str> {
        None
    }
}

/// In-process ONNX embeddings (fastembed).
pub struct FastembedProvider {
    model: Mutex<fastembed::TextEmbedding>,
    which: OnnxModel,
    name: String,
    dims: usize,
}

/// Selectable ONNX embedding model. Ordered by quality/cost trade-off.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OnnxModel {
    /// `all-MiniLM-L6-v2`, 22M params, 384 dims. Fastest, weakest.
    #[default]
    MiniLM,
    /// `snowflake-arctic-embed-m`, 109M params, 768 dims. Better
    /// separation, ~5x compute. Apache-2.0.
    ArcticM,
    /// `embeddinggemma-300m`, 300M params, 768 dims. Best retrieval
    /// under 500M params, slowest CPU per-query. Gemma license.
    Gemma300M,
    /// Bring-your-own directory: `model.onnx` (+`.onnx.data`), tokenizer
    /// files, mean pooling. From `embed --model <dir>`.
    Custom(std::path::PathBuf),
}

impl OnnxModel {
    fn embedding_model(self: &OnnxModel) -> Option<fastembed::EmbeddingModel> {
        match self {
            Self::MiniLM => Some(fastembed::EmbeddingModel::AllMiniLML6V2),
            Self::ArcticM => Some(fastembed::EmbeddingModel::SnowflakeArcticEmbedM),
            Self::Gemma300M => Some(fastembed::EmbeddingModel::EmbeddingGemma300M),
            Self::Custom(_) => None,
        }
    }

    /// Parse a CLI label or model directory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] for unknown labels.
    pub fn parse(label: &str) -> Result<Self, Error> {
        match label {
            "minilm" => Ok(Self::MiniLM),
            "arctic-m" => Ok(Self::ArcticM),
            "gemma-300m" => Ok(Self::Gemma300M),
            other => {
                let dir = std::path::PathBuf::from(other);
                if dir.join("model.onnx").is_file() {
                    Ok(Self::Custom(dir))
                } else {
                    Err(Error::InvalidInput(format!(
                        "unknown embedding model: {other} (minilm|arctic-m|gemma-300m|<dir>)"
                    )))
                }
            }
        }
    }

    /// Parse a stored provider name back, for `--hybrid` auto-detect.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the store was built by an unknown model.
    pub fn parse_stored(name: &str) -> Result<Self, Error> {
        match name {
            "fastembed/all-MiniLM-L6-v2" => Ok(Self::MiniLM),
            "fastembed/snowflake-arctic-embed-m" => Ok(Self::ArcticM),
            "fastembed/embeddinggemma-300m" => Ok(Self::Gemma300M),
            custom => custom
                .strip_prefix("user:")
                .map(|d| Self::Custom(std::path::PathBuf::from(d)))
                .ok_or_else(|| {
                    Error::Embed(format!(
                        "vector store uses unknown model {name}; re-run `embed`"
                    ))
                }),
        }
    }

    fn label(&self) -> String {
        match self {
            Self::MiniLM => "fastembed/all-MiniLM-L6-v2".to_owned(),
            Self::ArcticM => "fastembed/snowflake-arctic-embed-m".to_owned(),
            Self::Gemma300M => "fastembed/embeddinggemma-300m".to_owned(),
            Self::Custom(dir) => format!("user:{}", dir.display()),
        }
    }
}

impl FastembedProvider {
    /// Load the default model, downloading to the global cache on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the model cannot be fetched or started.
    pub fn load() -> Result<Self, Error> {
        Self::load_model(OnnxModel::default())
    }

    /// Load a specific model.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the model cannot be fetched or started.
    pub fn load_model(which: OnnxModel) -> Result<Self, Error> {
        if let OnnxModel::Custom(dir) = &which {
            return Self::load_user(dir, &which);
        }
        let cache = global_cache_dir();
        std::fs::create_dir_all(&cache).map_err(|e| Error::Embed(e.to_string()))?;
        let embedding_model = which
            .embedding_model()
            .ok_or_else(|| Error::Embed("custom model took the wrong path".to_owned()))?;
        let dims = fastembed::TextEmbedding::get_model_info(&embedding_model)
            .map_err(|e| Error::Embed(e.to_string()))?
            .dim;
        let model = fastembed::TextEmbedding::try_new(
            fastembed::TextInitOptions::new(embedding_model)
                .with_cache_dir(cache)
                .with_show_download_progress(true),
        )
        .map_err(|e| Error::Embed(e.to_string()))?;
        Ok(Self {
            model: Mutex::new(model),
            name: which.label(),
            which,
            dims,
        })
    }

    /// Load a bring-your-own ONNX directory (mean pooling).
    fn load_user(dir: &std::path::Path, which: &OnnxModel) -> Result<Self, Error> {
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .map_err(|e| Error::Embed(format!("{}: {e}", dir.join(name).display())))
        };
        // Prefer a single-file export; fall back to model.onnx.
        let onnx = read("model_single.onnx").or_else(|_| read("model.onnx"))?;
        let tokenizer_files = fastembed::TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        let mut user = fastembed::UserDefinedEmbeddingModel::new(onnx, tokenizer_files);
        user.pooling = Some(fastembed::Pooling::Mean);
        let mut model = fastembed::TextEmbedding::try_new_from_user_defined(
            user,
            fastembed::InitOptionsUserDefined::new(),
        )
        .map_err(|e| Error::Embed(e.to_string()))?;
        let dims = model
            .embed(vec!["probe"], None)
            .map_err(|e| Error::Embed(e.to_string()))?
            .into_iter()
            .next()
            .map(|v| v.len())
            .ok_or_else(|| Error::Embed("user model returned no vector".to_owned()))?;
        Ok(Self {
            model: Mutex::new(model),
            which: which.clone(),
            name: which.label(),
            dims,
        })
    }
}

impl EmbedProvider for FastembedProvider {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = self.model.lock().map_err(|e| Error::Embed(e.to_string()))?;
        model
            .embed(texts, None)
            .map_err(|e| Error::Embed(e.to_string()))
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn query_prefix(&self) -> Option<&str> {
        match self.which {
            OnnxModel::ArcticM => Some("Represent this sentence for searching relevant passages: "),
            OnnxModel::MiniLM | OnnxModel::Gemma300M | OnnxModel::Custom(_) => None,
        }
    }
}

/// OpenAI-compatible HTTP embeddings (`POST {base}/v1/embeddings`).
/// Placeholder for a future MLX embedding lane.
pub struct HttpProvider {
    base_url: String,
    model: String,
    dims: usize,
    client: reqwest::blocking::Client,
}

impl HttpProvider {
    /// Point at `base_url` (e.g. `http://127.0.0.1:8081`) with `model` id.
    #[must_use]
    pub fn new(base_url: &str, model: &str, dims: usize) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            model: model.to_owned(),
            dims,
            client: reqwest::blocking::Client::new(),
        }
    }
}

#[derive(serde::Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(serde::Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(serde::Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
}

impl EmbedProvider for HttpProvider {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let response: EmbedResponse = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(&EmbedRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.json())
            .map_err(|e| Error::Embed(e.to_string()))?;
        response.data.into_iter().map(|d| Ok(d.embedding)).collect()
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn name(&self) -> &str {
        &self.model
    }
}

fn global_cache_dir() -> std::path::PathBuf {
    std::env::var_os("HOME").map_or_else(
        || std::path::PathBuf::from(".one-grep-cache"),
        |home| {
            std::path::PathBuf::from(home)
                .join(".one-grep")
                .join("models")
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// std-only fake llama-server: serves one canned `/v1/rerank`
    /// response, then exits. Returns the base URL.
    fn fake_rerank_server(body: &'static str) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = vec![0u8; 65536];
            let _ = stream.read(&mut buf);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).expect("write");
        });
        format!("http://{addr}")
    }

    #[test]
    fn llama_reranker_orders_by_relevance_score() {
        let url = fake_rerank_server(
            r#"{"results":[{"index":1,"relevance_score":0.9},{"index":0,"relevance_score":0.2}]}"#,
        );
        let r = LlamaReranker::from_url(&url).expect("client");
        let order = r.rerank("q", &["alpha", "bravo"]).expect("rerank");
        assert_eq!(order, vec![(1, 0.9), (0, 0.2)]);
    }

    #[test]
    fn llama_reranker_accepts_plain_score_field() {
        let url = fake_rerank_server(r#"{"results":[{"index":0,"score":0.5}]}"#);
        let r = LlamaReranker::from_url(&url).expect("client");
        let order = r.rerank("q", &["only"]).expect("rerank");
        assert_eq!(order, vec![(0, 0.5)]);
    }

    #[test]
    fn llama_reranker_empty_docs_short_circuits() {
        // No server needed: empty input never touches the network.
        let r = LlamaReranker::from_url("http://127.0.0.1:9").expect("client");
        assert!(r.rerank("q", &[]).expect("rerank").is_empty());
    }

    #[test]
    fn rank_kind_parses_lowercase_labels() {
        let parse =
            |s: &str| serde_json::from_value::<RankKind>(serde_json::Value::String(s.into()));
        assert_eq!(parse("jev").expect("jev"), RankKind::Jev);
        assert_eq!(parse("jina").expect("jina"), RankKind::Jina);
        assert_eq!(parse("llama").expect("llama"), RankKind::Llama);
        assert!(parse("bert").is_err());
    }

    #[test]
    fn model_labels_roundtrip() {
        for (label, model) in [
            ("minilm", OnnxModel::MiniLM),
            ("arctic-m", OnnxModel::ArcticM),
            ("gemma-300m", OnnxModel::Gemma300M),
        ] {
            assert_eq!(OnnxModel::parse(label).expect("parse"), model);
        }
        assert!(OnnxModel::parse("bert").is_err());
    }
}

/// Cross-encoder reranking over fused candidates.
pub trait Rerank: Send + Sync {
    /// Score `docs` against `query`, best first as `(doc_index, score)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the backend fails.
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<(usize, f32)>, Error>;
}

/// Ranking backend selector, shared by CLI (`--rank`) and MCP
/// (`search_ranked rank`). `jev` is hosted, `jina` is in-process ONNX,
/// `llama` is a llama.cpp server (`--rerank`/`--embedding --pooling rank`,
/// e.g. `bge-reranker-v2-m3` on Vulkan) over HTTP.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum RankKind {
    Jev,
    Jina,
    Llama,
}

/// In-process cross-encoder (`jina-reranker-v1-turbo-en`, ~38M params).
pub struct JinaReranker {
    model: Mutex<fastembed::TextRerank>,
}

impl JinaReranker {
    /// Load the model, downloading to the global cache on first use.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the model cannot be fetched or started.
    pub fn load() -> Result<Self, Error> {
        let cache = global_cache_dir();
        std::fs::create_dir_all(&cache).map_err(|e| Error::Embed(e.to_string()))?;
        let model = fastembed::TextRerank::try_new(
            fastembed::RerankInitOptions::new(fastembed::RerankerModel::JINARerankerV1TurboEn)
                .with_cache_dir(cache)
                .with_show_download_progress(true),
        )
        .map_err(|e| Error::Embed(e.to_string()))?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }
}

impl Rerank for JinaReranker {
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<(usize, f32)>, Error> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = self.model.lock().map_err(|e| Error::Embed(e.to_string()))?;
        Ok(model
            .rerank(query, docs, false, None)
            .map_err(|e| Error::Embed(e.to_string()))?
            .into_iter()
            .map(|r| (r.index, r.score))
            .collect())
    }
}

/// Env override for the reranker server URL.
pub const ENV_RERANK_URL: &str = "ONE_GREP_RERANK_URL";
/// llama-server default port.
pub const DEFAULT_RERANK_URL: &str = "http://127.0.0.1:8080";
/// HTTP budget per rerank call.
const RERANK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Cross-encoder over HTTP: a llama.cpp server started with `--rerank`
/// (e.g. `bge-reranker-v2-m3`, ideally Vulkan-offloaded). Speaks the
/// documented `POST /v1/rerank {query, documents, top_n}` endpoint and
/// returns best-first `(doc_index, score)` like every [`Rerank`] backend.
pub struct LlamaReranker {
    client: reqwest::blocking::Client,
    base_url: String,
}

impl std::fmt::Debug for LlamaReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaReranker")
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl LlamaReranker {
    /// Build for an explicit base URL (`http://127.0.0.1:8080` shape).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the HTTP client cannot be built.
    pub fn from_url(base_url: &str) -> Result<Self, Error> {
        let client = reqwest::blocking::Client::builder()
            .timeout(RERANK_TIMEOUT)
            .build()
            .map_err(|e| Error::Embed(e.to_string()))?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_owned(),
        })
    }

    /// Build from `ONE_GREP_RERANK_URL`, defaulting to the llama-server
    /// loopback default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embed`] when the HTTP client cannot be built.
    pub fn from_env() -> Result<Self, Error> {
        let base_url =
            std::env::var(ENV_RERANK_URL).unwrap_or_else(|_| DEFAULT_RERANK_URL.to_owned());
        Self::from_url(&base_url)
    }

    fn post(&self, query: &str, docs: &[&str]) -> Result<serde_json::Value, Error> {
        let url = format!("{}/v1/rerank", self.base_url);
        let response = self
            .client
            .post(url)
            .json(&serde_json::json!({
                "query": query,
                "documents": docs,
                "top_n": docs.len(),
            }))
            .send()
            .map_err(|e| Error::Embed(e.to_string()))?;
        let status = response.status();
        let text = response.text().map_err(|e| Error::Embed(e.to_string()))?;
        if !status.is_success() {
            let mut err = text.trim().to_owned();
            if err.len() > 180 {
                err.truncate(180);
            }
            return Err(Error::Embed(format!("rerank HTTP {status}: {err}")));
        }
        serde_json::from_str(&text).map_err(|e| Error::Embed(e.to_string()))
    }
}

impl Rerank for LlamaReranker {
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<(usize, f32)>, Error> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let body = self.post(query, docs)?;
        let mut scored: Vec<(usize, f32)> = Vec::new();
        if let Some(results) = body.get("results").and_then(|r| r.as_array()) {
            for item in results {
                let (Some(index), Some(score)) = (
                    item.get("index").and_then(serde_json::Value::as_u64),
                    item.get("relevance_score")
                        .or_else(|| item.get("score"))
                        .and_then(serde_json::Value::as_f64),
                ) else {
                    continue;
                };
                if (index as usize) < docs.len() {
                    scored.push((index as usize, score as f32));
                }
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(scored)
    }
}
