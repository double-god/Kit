//! Text embedding module using FastEmbed.
//!
//! This module provides a wrapper around FastEmbed's TextEmbedding to generate
//! 384-dimensional vectors from text using the BGE-small-en-v1.5 model.
//!
//! # Thread Safety
//!
//! [`EmbeddingModel`] implements `Send + Sync` and can be safely shared across threads.
//! Note that [`EmbeddingModel::embed_text`] requires `&mut self`, so for concurrent
//! use you may need to wrap it in a `Mutex` or `RwLock` when using `Arc`.
//!
//! # Performance
//!
//! Model initialization is expensive (downloads and loads ONNX model) and should
//! be done once at application startup. Subsequent `embed_text` calls are fast
//! (< 100ms for single text).
//!
//! # Environment Variables
//!
//! - `FASTEMBED_CACHE_DIR`: Optional path to store downloaded ONNX models.
//!   If not set, FastEmbed uses the system default cache directory.
//!   Set this to share model files across multiple worktrees or CI environments.
//!
//! # Example
//!
//! ```rust
//! use contextfy_core::embeddings::EmbeddingModel;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Initialize once at startup
//! let mut model = EmbeddingModel::new()?;
//!
//! // Generate embedding for text
//! let vector = model.embed_text("Hello, world!")?;
//! assert_eq!(vector.len(), 384);
//! # Ok(())
//! # }
//! ```

use anyhow::Context;
use fastembed::{EmbeddingModel as FastEmbedModel, InitOptions, TextEmbedding};

/// Text embedding model wrapper.
///
/// Wraps FastEmbed's `TextEmbedding` with a simplified API optimized for
/// single-text embedding generation. Uses BGE-small-en-v1.5 model by default,
/// producing 384-dimensional float vectors.
///
/// # Thread Safety
///
/// This type is `Send + Sync` and can be safely shared across threads.
/// Note that [`embed_text()`][Self::embed_text] requires `&mut self`, so for
/// concurrent use you may need to wrap it in a `Mutex` or `RwLock` when using `Arc`.
///
/// # Performance
///
/// - **Initialization**: Expensive (model download + ONNX loading), do once at startup
/// - **Per-query**: < 100ms for single text (after first call)
pub struct EmbeddingModel {
    inner: TextEmbedding,
}

impl EmbeddingModel {
    /// Initializes a new embedding model with BGE-small-en-v1.5.
    ///
    /// This method downloads the ONNX model on first run (cached locally) and
    /// initializes the inference runtime. Subsequent calls load from cache.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model download fails (network issue or disk full)
    /// - ONNX runtime initialization fails
    /// - Model file is corrupted
    ///
    /// # Example
    ///
    /// ```rust
    /// use anyhow::Context;
    /// use contextfy_core::embeddings::EmbeddingModel;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let model = EmbeddingModel::new()
    ///     .context("Failed to initialize embedding model")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> anyhow::Result<Self> {
        let inner = TextEmbedding::try_new(
            InitOptions::new(FastEmbedModel::BGESmallENV15).with_show_download_progress(true),
        )
        .context("Failed to initialize FastEmbed TextEmbedding with BGE-small-en-v1.5")?;

        Ok(Self { inner })
    }

    /// Generates a 384-dimensional embedding vector for the given text.
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to embed (can be empty string)
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` of length 384 containing the embedding vector.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Text encoding fails
    /// - ONNX inference fails
    /// - Returned embedding has unexpected dimension
    ///
    /// # Safety Guarantees
    ///
    /// This method:
    /// - Never panics on valid UTF-8 input
    /// - Returns descriptive errors via `anyhow::Context`
    /// - Validates output dimension (must be 384)
    ///
    /// # Example
    ///
    /// ```rust
    /// use contextfy_core::embeddings::EmbeddingModel;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// # let mut model = EmbeddingModel::new()?;
    /// let vector = model.embed_text("Hello, world!")?;
    /// assert_eq!(vector.len(), 384);
    /// # Ok(())
    /// # }
    /// ```
    pub fn embed_text(&mut self, text: &str) -> anyhow::Result<Vec<f32>> {
        // FastEmbed's embed method accepts Vec<&str> and returns Vec<Vec<f32>>
        // We wrap the single text in an array and extract the first result safely
        let embeddings = self
            .inner
            .embed(vec![text], None)
            .context("Failed to generate embedding for text")?;

        // Safely extract the first (and only) embedding from the batch result
        let embedding = embeddings
            .into_iter()
            .next()
            .context("Embedding batch returned empty results")?;

        // Validate dimension (BGE-small-en-v1.5 always produces 384-dimensional vectors)
        if embedding.len() != 384 {
            anyhow::bail!(
                "Expected 384-dimensional embedding, got {} dimensions",
                embedding.len()
            );
        }

        Ok(embedding)
    }
}

// SAFETY: TextEmbedding is Send + Sync, so our wrapper is too
// This is explicitly documented for users who need Arc<EmbeddingModel>
unsafe impl Send for EmbeddingModel {}
unsafe impl Sync for EmbeddingModel {}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_model_loading() {
        // Test that model initialization succeeds
        let model = EmbeddingModel::new();
        assert!(model.is_ok(), "Model initialization should succeed");
    }

    #[test]
    #[serial]
    fn test_embedding_dimension() {
        // Test that embeddings have exactly 384 dimensions
        let mut model = EmbeddingModel::new().expect("Model should initialize");
        let vector = model
            .embed_text("Test text for dimension validation")
            .expect("Embedding generation should succeed");

        assert_eq!(
            vector.len(),
            384,
            "Embedding vector must have exactly 384 dimensions"
        );
    }

    #[test]
    #[serial]
    fn test_embedding_determinism() {
        // Test that identical text produces identical embeddings
        let mut model = EmbeddingModel::new().expect("Model should initialize");
        let text = "Deterministic test text";

        let vector1 = model
            .embed_text(text)
            .expect("First embedding should succeed");
        let vector2 = model
            .embed_text(text)
            .expect("Second embedding should succeed");

        // All dimensions should match (floating-point equality is fine here)
        assert_eq!(
            vector1.len(),
            vector2.len(),
            "Embedding dimensions should match"
        );

        for (i, (v1, v2)) in vector1.iter().zip(vector2.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-6,
                "Dimension {} differs: {} vs {}",
                i,
                v1,
                v2
            );
        }
    }

    #[test]
    #[serial]
    fn test_empty_text_embedding() {
        // Test that empty text produces valid embedding
        let mut model = EmbeddingModel::new().expect("Model should initialize");
        let vector = model
            .embed_text("")
            .expect("Empty text embedding should succeed");

        assert_eq!(
            vector.len(),
            384,
            "Empty text embedding should be 384-dimensional"
        );
    }

    #[test]
    #[serial]
    fn test_unicode_text_embedding() {
        // Test that Unicode text (Chinese, emojis) works correctly
        let mut model = EmbeddingModel::new().expect("Model should initialize");
        let text = "你好世界 🚀 Hello 世界!";

        let vector = model
            .embed_text(text)
            .expect("Unicode text embedding should succeed");

        assert_eq!(
            vector.len(),
            384,
            "Unicode text embedding should be 384-dimensional"
        );
    }

    #[test]
    #[serial]
    #[ignore]
    fn bench_embedding_speed() {
        // Performance benchmark: single text embedding should be < 100ms
        // Run with: cargo test --package contextfy-core bench_embedding_speed -- --ignored
        let mut model = EmbeddingModel::new().expect("Model should initialize");

        // Warm-up call (model loading overhead)
        let _ = model
            .embed_text("Warm-up text")
            .expect("Warm-up should succeed");

        // Measure single text embedding (typical use case)
        let test_text = "This is a typical text document that might be embedded in a production environment. It contains enough content to represent a realistic scenario.";
        let start = std::time::Instant::now();
        let vector = model
            .embed_text(test_text)
            .expect("Embedding should succeed");
        let duration = start.elapsed();

        println!("Embedding generation took: {:?}", duration);
        println!("Generated vector of {} dimensions", vector.len());

        // Assert performance requirement: < 100ms
        assert!(
            duration.as_millis() < 100,
            "Embedding generation should be < 100ms, took {}ms",
            duration.as_millis()
        );

        // Also verify dimension
        assert_eq!(vector.len(), 384, "Vector must be 384-dimensional");
    }
}
