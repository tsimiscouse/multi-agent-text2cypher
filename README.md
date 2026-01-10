# Multi-Agent Text-to-Cypher

Penelitian evaluasi sistem multi-agent untuk meningkatkan akurasi Text-to-Cypher pada Knowledge Graph kurikulum berbahasa Indonesia.

## Deskripsi

Penelitian ini mengimplementasikan dan mengevaluasi sistem multi-agent iterative refinement untuk tugas Text-to-Cypher, dengan membandingkan performanya terhadap baseline single-shot generation (kg-axel).

**Baseline:** kg-axel (CoT + only_paths)
**Proposed:** Multi-Agent Iterative Refinement System

## Struktur Project

```
multi-agent-text2cypher/
├── agents/                  # Multi-agent components
│   ├── query_generator.py   # Agent 1: Query generation & refinement
│   ├── query_evaluator.py   # Agent 2: LLM-based query evaluation
│   ├── entity_extractor.py  # Agent 3: Named entity extraction
│   ├── entity_verifier.py   # Agent 4: Entity verification & correction
│   ├── instructions_generator.py  # Agent 5: Correction instructions
│   └── feedback_aggregator.py     # Agent 6: Feedback synthesis
│
├── baseline/                # Baseline system (kg-axel)
│   ├── kg_axel_generator.py
│   └── prompts.py
│
├── system/                  # Multi-agent orchestration
│   ├── orchestrator.py      # Main multi-agent coordinator
│   ├── graph_executor.py    # Neo4j query executor
│   └── agent_state.py       # State management
│
├── notebooks/               # Experiment notebooks
│   ├── 01_inference_baseline.ipynb
│   ├── 02_inference_multiagent.ipynb
│   ├── 03_metrics_evaluation.ipynb
│   └── 04_efficiency_analysis.ipynb
│
├── data/                    # Dataset
│   ├── ground_truth/
│   │   ├── ground_truth_52.csv     # Original from kg-axel
│   │   └── ground_truth_100.csv    # Expanded dataset
│   └── schemas/
│       ├── only_paths.txt
│       ├── nodes_and_paths.txt
│       └── full_schema.txt
│
├── prompts/                 # Prompt templates
│   ├── templates/
│   │   ├── query_generator.txt
│   │   ├── query_evaluator.txt
│   │   ├── entity_verifier.txt
│   │   └── feedback_aggregator.txt
│   └── examples/
│
├── output/                  # Experiment outputs
│   ├── baseline_kg_axel.csv
│   ├── multiagent_k3.csv
│   └── multiagent_k3_states.json
│
├── outputMetrics/          # Evaluation metrics
│   ├── comparison_metrics.csv
│   ├── ablation_study.csv
│   └── cost_benefit_analysis.csv
│
├── evaluation/             # Evaluation modules
│   ├── metrics.py
│   ├── comparator.py
│   └── analyzer.py
│
├── config/                 # Configuration files
│   ├── model_config.yaml
│   └── experiment_config.yaml
│
├── docs/                   # Documentation
│   ├── architecture.md
│   ├── agent_specifications.md
│   └── experiment_protocol.md
│
└── utils/                  # Utility functions
    ├── prompt_loader.py
    ├── schema_loader.py
    └── logger.py
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and Neo4j credentials
```

## Quick Start

### 1. Run Baseline Experiment (kg-axel)

```bash
jupyter notebook notebooks/01_inference_baseline.ipynb
```

### 2. Run Multi-Agent Experiment

```bash
jupyter notebook notebooks/02_inference_multiagent.ipynb
```

### 3. Evaluate Metrics

```bash
jupyter notebook notebooks/03_metrics_evaluation.ipynb
```

### 4. Analyze Efficiency

```bash
jupyter notebook notebooks/04_efficiency_analysis.ipynb
```

## Research Questions

**RQ1:** Seberapa efektif multi-agent refinement dalam meningkatkan Pass@k dibandingkan single-shot generation (kg-axel baseline)?

**RQ2:** Pada jenis query apa multi-agent memberikan improvement terbesar?

**RQ3:** Apa trade-off antara peningkatan akurasi dengan computational cost?

## Metrics

- **Pass@k**: Query output matches ground truth
- **KG Valid@k**: Structurally valid query
- **Execution Success Rate**: Query executes without error
- **Recovery Rate**: Queries fixed through refinement
- **Avg Iterations**: Average refinement iterations
- **Token Cost**: Total tokens used
- **Latency**: Query generation time

## Experiment Configuration

```yaml
Baseline:
  System: kg-axel
  Config: CoT + only_paths
  Max k: 1 (single-shot)

Multi-Agent:
  System: Multi-Agent Iterative Refinement
  Max k: 3
  Agents: 6 (Generator, Evaluator, Extractor, Verifier, Instructions, Aggregator)

Dataset:
  Original: 52 questions (kg-axel)
  Expanded: 100 questions
  Complexity: Easy, Medium, Hard
```

## Results

Results akan tersimpan di:
- `output/`: Raw experiment outputs
- `outputMetrics/`: Computed metrics and comparisons
- `notebooks/`: Analysis notebooks with visualizations

## Citation

```bibtex
@thesis{multiagent2025,
  title={Multi-Agent Iterative Refinement untuk Text-to-Cypher pada Knowledge Graph Kurikulum},
  author={[Your Name]},
  year={2025},
  school={[Your University]}
}
```

## License

MIT License

## Contact

[Your Contact Information]
