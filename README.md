# Dashboard de Inteligencia Económica 🇵🇪

App Streamlit para consultar y analizar indicadores macroeconómicos del
**BCRP**, **INEI** y **Banco Mundial**, con procesamiento de series, modelos
econométricos, reportes PDF y análisis asistido por LLM.

## Estructura

- `app.py` — Aplicación Streamlit (punto de entrada).
- `src/data/` — Conectores a APIs (BCRP, INEI, Banco Mundial).
- `src/processing/` — Limpieza y transformación de series.
- `src/models/` — Modelos econométricos y de ML.
- `src/reports/` — Generación de reportes PDF.
- `src/llm/` — Analista económico asistido por LLM.
- `assets/` — Recursos gráficos.

## Puesta en marcha (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

## Configuración

Copia `.env.example` a `.env` y completa las credenciales de las APIs.
