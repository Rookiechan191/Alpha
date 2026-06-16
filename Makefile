.PHONY: setup scrape chunk index eval run clean ingest

setup:
	pip install -r requirements.txt
	cp -n .env.example .env || true

scrape:
	python ingest/scrape.py --limit 300

chunk:
	python ingest/chunk.py

index:
	python ingest/build_index.py

ingest: scrape chunk index

eval:
	python eval/run_eval.py

run:
	python main.py --interactive

clean:
	rm -rf data/raw data/index data/chunks.jsonl
