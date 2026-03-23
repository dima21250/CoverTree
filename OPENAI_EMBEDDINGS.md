# Using CoverTree with OpenAI Embeddings

This guide explains how to use the CoverTree package with OpenAI embeddings for semantic search, clustering, and hierarchical analysis of text data.

## Overview

The CoverTree package now supports direct integration with OpenAI's embedding API, allowing you to:
- Build hierarchical search indexes from text data
- Perform semantic nearest neighbor search
- Create hierarchical clusters of semantically similar texts
- Compare CoverTree performance with FAISS on real-world embeddings

## Use Cases

- **Semantic Search**: Find documents/texts similar to a query based on meaning
- **Document Clustering**: Automatically group similar documents at multiple granularities
- **Anomaly Detection**: Identify outlier texts that don't fit common patterns
- **Hierarchical Topic Analysis**: Explore document collections at coarse-to-fine levels

## Installation

### 1. Install System Dependencies

**macOS:**
```bash
# Install build tools and dependencies
brew install cmake eigen

# Install Python package manager
brew install uv
```

**Linux:**
```bash
sudo apt-get install cmake libeigen3-dev python3-pip
pip install uv
```

### 2. Install Python Dependencies

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install core dependencies
uv pip install numpy scipy scikit-learn

# Install OpenAI integration dependencies
uv pip install openai tenacity

# Optional: Install FAISS for comparison
uv pip install faiss-cpu  # or faiss-gpu for GPU support
uv pip install torch  # Required for GPU support
```

### 3. Build CoverTree with OpenAI Support

**Option A: Quick Build with OpenAI support enabled**
```bash
./build.sh --python -DUSE_OPENAI_EMBEDDINGS=ON
```

**Option B: Manual CMake Configuration**
```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_OPENAI_EMBEDDINGS=ON \
  -DUSE_GPU=ON  # Optional: Enable GPU support for FAISS
make -j$(sysctl -n hw.ncpu)  # macOS
# or
make -j$(nproc)  # Linux
cd ..
```

**Option C: Install Python package**
```bash
# After building, install the package
uv pip install -e .
```

## Configuration

### Setting Up API Credentials

The package reads OpenAI API configuration from environment variables:

```bash
# Required: Your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Optional: Custom endpoint (defaults to OpenAI's official endpoint)
export OPENAI_ENDPOINT="https://api.openai.com/v1/embeddings"

# Optional: Model selection (defaults to text-embedding-3-small)
export OPENAI_MODEL="text-embedding-3-small"
```

**Recommended API Keys Storage:**

Create a `.env` file in your project directory (DO NOT commit to git):
```bash
# .env file
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=text-embedding-3-small
```

Load it in your Python script:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Available OpenAI Embedding Models

| Model | Dimensions | Performance | Cost |
|-------|-----------|-------------|------|
| `text-embedding-3-small` | 1536 | Fast, efficient | Low |
| `text-embedding-3-large` | 3072 | Best quality | Medium |
| `text-embedding-ada-002` | 1536 | Legacy model | Low |

You can also specify custom dimensions for text-embedding-3 models:
```bash
export OPENAI_DIMENSIONS=512  # Reduce dimensionality
```

## Basic Usage

### Example 1: Simple Text Embedding and Search

```python
import os
import numpy as np
from covertree import CoverTree

# Import the helper function from sample_events.py
from sample_events import get_openai_embeddings

# Your text data
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming technology.",
    "Natural language processing enables computers to understand text.",
    "Deep learning models require large datasets.",
    "Python is a popular programming language for AI."
]

# Get embeddings from OpenAI
api_key = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1/embeddings")
model = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

embeddings = get_openai_embeddings(texts, api_key, endpoint, model)

# Normalize embeddings (OpenAI embeddings are already normalized, but ensure it)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = np.float32(embeddings)

# Build CoverTree index
ct = CoverTree.from_matrix(embeddings)

# Verify tree structure
print(f"Tree is valid: {ct.check_covering()}")
print(f"Number of points: {len(texts)}")

# Query: Find nearest neighbor
query_text = ["Python programming for machine learning"]
query_embedding = get_openai_embeddings(query_text, api_key, endpoint, model)
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
query_embedding = np.float32(query_embedding[0])

# Search
result = ct.NearestNeighbour(query_embedding)
nearest_idx = result[0].point_idx  # Get the index of nearest neighbor
print(f"\nQuery: {query_text[0]}")
print(f"Nearest match: {texts[nearest_idx]}")
```

### Example 2: K-Nearest Neighbors

```python
# Find top 3 most similar texts
k = 3
results = ct.kNearestNeighbours(query_embedding, k=k)

print(f"\nTop {k} matches for: {query_text[0]}\n")
for i, (node, distance) in enumerate(results, 1):
    idx = node.point_idx
    print(f"{i}. {texts[idx]} (distance: {distance:.4f})")
```

### Example 3: Hierarchical Clustering

```python
# Get hierarchical clustering information
ct_json = ct.dumps()
import json
ct_data = json.loads(ct_json)

# Analyze tree structure
print("\nTree Hierarchy:")
print(f"Minimum level: {ct_data['min_level']}")
print(f"Maximum level: {ct_data['max_level']}")

# Count nodes per level
from collections import defaultdict
level_counts = defaultdict(int)
for node in ct_data['nodes']:
    level_counts[node['level']] += 1

print("\nNodes per level:")
for level in sorted(level_counts.keys(), reverse=True):
    print(f"  Level {level}: {level_counts[level]} nodes")
```

### Example 4: Using the Modern Clustering API

```python
from covertree2 import CoverTree as CoverTree2

# Build tree with modern API
tree = CoverTree2.from_matrix(embeddings)

# Get statistics
stats = tree.get_level_stats()
print(f"\nTree Statistics:")
print(f"  Min level: {stats['min_level']}")
print(f"  Max level: {stats['max_level']}")
print(f"  Total points: {stats['num_points']}")

# Get clusters at different granularities
coarse_level = stats['min_level'] + 2
fine_level = stats['max_level'] - 2

coarse_clusters = tree.get_clusters_at_level(coarse_level)
fine_clusters = tree.get_clusters_at_level(fine_level)

print(f"\nCoarse clustering (level {coarse_level}): {len(coarse_clusters)} clusters")
print(f"Fine clustering (level {fine_level}): {len(fine_clusters)} clusters")

# Examine a cluster
for i, cluster in enumerate(coarse_clusters[:3]):
    print(f"\nCluster {i}:")
    print(f"  Size: {cluster.size} points")
    print(f"  Covering distance: {cluster.covering_distance:.4f}")
    # Get the texts in this cluster
    point_indices = cluster.point_ids
    cluster_texts = [texts[idx] for idx in point_indices if idx < len(texts)]
    for text in cluster_texts[:3]:  # Show first 3
        print(f"    - {text}")
```

## Advanced Features

### GPU Acceleration with FAISS

If you built with `-DUSE_GPU=ON`, you can enable GPU acceleration for FAISS comparisons:

```python
import torch
import faiss

# Check GPU availability
if torch.cuda.is_available():
    print("GPU is available for acceleration")

    # FAISS GPU index will be automatically used in sample_events.py
    # Or create manually:
    gpu_id = 0
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu_id

    d = embeddings.shape[1]
    index = faiss.GpuIndexFlatIP(res, d, cfg)  # Inner product (for normalized vectors)
    index.add(embeddings)
else:
    # Fallback to CPU
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
```

### Batch Processing Large Document Collections

For large text collections, process in batches:

```python
from sample_events import get_openai_embeddings

# Large document collection
documents = [...]  # Your large list of texts

# The get_openai_embeddings function automatically batches requests
# Default batch size: 1000 (safe for most APIs)
embeddings = get_openai_embeddings(
    documents,
    api_key=api_key,
    endpoint=endpoint,
    model=model,
    max_retries=6  # Retry on rate limits
)

# Build tree
ct = CoverTree.from_matrix(embeddings)
```

### Custom Retry Logic

The `get_openai_embeddings` function includes automatic retry with exponential backoff:

```python
embeddings = get_openai_embeddings(
    texts,
    api_key=api_key,
    endpoint=endpoint,
    model=model,
    dimensions=1024,      # Optional: custom dimensions
    max_retries=10        # Increase for unreliable connections
)
```

Retries are triggered for:
- Rate limit errors (429)
- Connection errors
- Timeout errors
- Internal server errors (500+)

### Using Alternative OpenAI-Compatible APIs

Many services provide OpenAI-compatible embedding APIs:

```bash
# Example: Azure OpenAI
export OPENAI_API_KEY="your-azure-key"
export OPENAI_ENDPOINT="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
export OPENAI_MODEL="text-embedding-ada-002"

# Example: Local embedding server
export OPENAI_ENDPOINT="http://localhost:8080/v1/embeddings"
export OPENAI_MODEL="your-model-name"
```

## Running the Sample Script

The package includes `sample_events.py` which demonstrates the integration:

```bash
# Ensure environment is configured
export OPENAI_API_KEY="sk-..."

# Run the sample
python sample_events.py
```

The script will:
1. Get embeddings from OpenAI for sample texts
2. Build a CoverTree index
3. Build a FAISS index (with GPU if available)
4. Create hierarchical clusters
5. Perform outlier detection
6. Generate visualization data

## Performance Considerations

### Embedding Costs

OpenAI charges per token:
- `text-embedding-3-small`: ~$0.02 per 1M tokens
- `text-embedding-3-large`: ~$0.13 per 1M tokens

**Cost Optimization:**
- Cache embeddings locally
- Use smaller models for development
- Reduce dimensions for text-embedding-3 models
- Batch requests (already handled by `get_openai_embeddings`)

### Memory Usage

For N documents with D dimensions:
- CoverTree memory: ~O(N * D * 8) bytes for float64
- FAISS memory: ~O(N * D * 4) bytes for float32
- Total: ~12 bytes per dimension per document

**Example:** 100,000 documents with 1536 dimensions:
- CoverTree: ~1.2 GB
- FAISS: ~600 MB
- Total: ~1.8 GB

### Query Performance

CoverTree provides sublinear query time:
- **Construction**: O(N log N) for N points
- **Query**: O(log N) on average
- **Space**: O(N) nodes

For comparison with FAISS:
- FAISS exact search: O(N) - linear scan
- FAISS approximate search (IVF): O(√N) with quantization
- CoverTree: O(log N) average case

## Troubleshooting

### "ImportError: cannot import name 'USE_OPENAI_EMBEDDINGS'"

**Cause:** Package was not built with OpenAI support enabled.

**Solution:**
```bash
./build.sh --clean --python -DUSE_OPENAI_EMBEDDINGS=ON
uv pip install -e .
```

### "ValueError: Please set the OPENAI_API_KEY environment variable"

**Cause:** API key not configured.

**Solution:**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or use a `.env` file:
```python
from dotenv import load_dotenv
load_dotenv()
```

### "ImportError: OpenAI and tenacity packages are required"

**Cause:** Optional dependencies not installed.

**Solution:**
```bash
uv pip install openai tenacity
```

### Rate Limit Errors (429)

**Cause:** Exceeding OpenAI API rate limits.

**Solution:**
- The retry logic handles this automatically with exponential backoff
- Reduce batch size if needed
- Upgrade your OpenAI account tier
- Add delays between batches:

```python
import time
batches = [texts[i:i+100] for i in range(0, len(texts), 100)]
all_embeddings = []
for batch in batches:
    emb = get_openai_embeddings(batch, api_key, endpoint, model)
    all_embeddings.append(emb)
    time.sleep(1)  # Rate limiting
embeddings = np.vstack(all_embeddings)
```

### GPU Not Available

**Cause:** PyTorch/FAISS GPU not installed or no CUDA device.

**Solution:**
```bash
# Install GPU versions
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
uv pip install faiss-gpu

# Rebuild with GPU support
./build.sh --clean --python -DUSE_GPU=ON -DUSE_OPENAI_EMBEDDINGS=ON
```

The code automatically falls back to CPU if GPU is not available.

### "RuntimeError: CoverTree construction failed"

**Cause:** Invalid input data (NaN, Inf, or non-normalized vectors).

**Solution:**
```python
# Check for invalid values
assert not np.any(np.isnan(embeddings))
assert not np.any(np.isinf(embeddings))

# Ensure normalization
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Convert to float32 for consistency
embeddings = np.float32(embeddings)
```

## Configuration Reference

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_OPENAI_EMBEDDINGS` | `OFF` | Enable OpenAI API integration |
| `USE_GPU` | `OFF` | Enable GPU support for FAISS |
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Release/Debug) |

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `OPENAI_ENDPOINT` | No | `https://api.openai.com/v1/embeddings` | API endpoint |
| `OPENAI_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `OPENAI_DIMENSIONS` | No | (model default) | Custom dimensions |

## Examples Repository

For more examples, see:
- `sample_events.py` - Full integration example
- `test_pybind11.py` - Modern API examples
- `test_clustering.py` - Clustering API examples

## Additional Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [CoverTree Paper (Izbicki & Shelton 2015)](https://arxiv.org/abs/1502.06833)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

## License

This project follows the same license as the main CoverTree package.
