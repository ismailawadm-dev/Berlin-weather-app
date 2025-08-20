# Berlin Rain Planning App ☔

A simple app for **week‑2 rain risk** in Berlin + an **imminent 10‑minute rain check** for last‑mile ops.

## Deploy on Streamlit Cloud
1) Create a GitHub repo and upload this folder (include `apt.txt`, `requirements.txt`, `config.yaml`, `streamlit_app.py`, and `src/`).  
2) In Streamlit Cloud → **New app** → choose this repo → set file to `streamlit_app.py`.  
3) First run: click **Train models (3 years)** inside the app. Then pick dates.

## Local run (optional)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `config.yaml` : Berlin coords + settings
- `apt.txt` : system packages for Streamlit Cloud (eccodes etc.)
- `requirements.txt` : python deps
- `streamlit_app.py` : web app
- `src/` : data ingestion, features, models, alerts
