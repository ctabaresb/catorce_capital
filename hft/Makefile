# =============================================================================
# Makefile — Bitso HFT Strategy Lab
# Path: /Users/carlos/Documents/GitHub/catorce_capital/hft
# =============================================================================

PYTHON      = python3
PIP         = pip3
DATA_DIR    = ./data
RESULTS_DIR = ./results
ASSET       = btc

.PHONY: help install test synthetic real load-only backtest clean sync-data \
        deploy-ec2 watch-logs kill-all

# =============================================================================
# Help
# =============================================================================
help:
	@echo ""
	@echo "Bitso HFT Strategy Lab"
	@echo "======================"
	@echo ""
	@echo "SETUP"
	@echo "  make install        Install all Python dependencies"
	@echo "  make env            Copy .env.example to .env (edit before use)"
	@echo ""
	@echo "LOCAL RESEARCH (Mac)"
	@echo "  make test           Run pytest suite (no data needed)"
	@echo "  make synthetic      Generate synthetic data + run full lab"
	@echo "  make real           Load real parquet + run lab (requires data/)"
	@echo "  make load-only      Build features.parquet from raw data (cache step)"
	@echo "  make backtest       Run backtest on cached features.parquet"
	@echo ""
	@echo "DATA"
	@echo "  make sync-data      Sync parquet files from S3"
	@echo "  make clean-data     Delete legacy parquet files from S3 (see HANDOFF.md)"
	@echo ""
	@echo "ENVIRONMENT"
	@echo "  make clean          Remove generated files"
	@echo ""

# =============================================================================
# Setup
# =============================================================================
install:
	$(PIP) install -r requirements.txt

env:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env — edit it before use"; \
	else echo ".env already exists"; fi

# =============================================================================
# Local research commands
# =============================================================================
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

synthetic:
	$(PYTHON) -m research.run_research --mode synthetic \
	    --data-dir $(DATA_DIR) --out-dir $(RESULTS_DIR)

real:
	$(PYTHON) -m research.run_research --mode real \
	    --data-dir $(DATA_DIR) --out-dir $(RESULTS_DIR) --asset $(ASSET)

load-only:
	$(PYTHON) -m research.run_research --mode load-only \
	    --data-dir $(DATA_DIR) --out-dir $(RESULTS_DIR) --asset $(ASSET)

backtest:
	$(PYTHON) -m research.run_research --mode backtest \
	    --features $(RESULTS_DIR)/features_$(ASSET).parquet \
	    --out-dir $(RESULTS_DIR) --asset $(ASSET)

# Run all three assets (after syncing data)
all-assets:
	$(PYTHON) -m research.run_research --mode real --asset btc \
	    --data-dir $(DATA_DIR) --out-dir $(RESULTS_DIR)
	$(PYTHON) -m research.run_research --mode real --asset eth \
	    --data-dir $(DATA_DIR) --out-dir $(RESULTS_DIR)
	$(PYTHON) -m research.run_research --mode real --asset sol \
	    --data-dir $(DATA_DIR) --out-dir $(RESULTS_DIR)

# =============================================================================
# S3 data sync
# =============================================================================
S3_BUCKET = s3://bitso-orderbook/data/

sync-data:
	@echo "Syncing parquet files from S3..."
	aws s3 sync $(S3_BUCKET) $(DATA_DIR)/ \
	    --exclude "*" \
	    --include "btc_bitso_*.parquet" \
	    --include "eth_bitso_*.parquet" \
	    --include "sol_bitso_*.parquet" \
	    --include "trades_*.parquet" \
	    --include "book_20260307_211515.parquet"
	@echo "Sync complete. Files in $(DATA_DIR):"
	@ls -lh $(DATA_DIR)/*.parquet 2>/dev/null || echo "  (no parquet files found)"

# Delete legacy files from S3 (see HANDOFF.md for list)
clean-data-s3:
	@echo "Deleting legacy S3 files..."
	aws s3 rm $(S3_BUCKET)book_20260307_053702.parquet
	aws s3 rm $(S3_BUCKET)book_20260307_054100.parquet
	aws s3 rm $(S3_BUCKET)book_20260307_054245.parquet
	aws s3 rm $(S3_BUCKET)binance_20260307_190559.parquet
	aws s3 rm $(S3_BUCKET)binance_20260307_211520.parquet
	aws s3 rm $(S3_BUCKET)binance_20260308_174204.parquet
	aws s3 rm $(S3_BUCKET)coinbase_20260307_190559.parquet
	aws s3 rm $(S3_BUCKET)coinbase_20260307_211520.parquet
	aws s3 rm $(S3_BUCKET)coinbase_20260308_174204.parquet
	aws s3 rm $(S3_BUCKET)trades_20260307_054100.parquet
	@echo "Done."

# =============================================================================
# Cleanup
# =============================================================================
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f $(RESULTS_DIR)/features_*.parquet
	rm -f $(DATA_DIR)/*.parquet
	@echo "Cleaned generated files. S3 data not touched."
