.PHONY: dev ui

dev:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

ui:
	streamlit run app/app.py
