.PHONY: eval test

# Run the RAGAS evaluation harness then assert CI thresholds.
# Requires: pgvector container running (docker-compose up -d)
#           GROQ_API_KEY set in .env
eval:
	python tests/eval/run_eval.py && python -m pytest tests/eval/test_ci_gate.py -v

# Run the full test suite (unit + integration, excludes the slow eval harness).
test:
	python -m pytest tests/ --ignore=tests/eval/run_eval.py -v
