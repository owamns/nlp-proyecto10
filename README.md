# NLP Proyecto 10: Arquitecturas Seq2Seq Jerárquicas
## Descripción
Proyecto para procesar documentos largos (>2000 tokens) usando una arquitectura Seq2Seq jerárquica. Implementa un encoder local (BERT) para procesar chunks y un encoder global (Transformer) para capturar dependencias entre secciones. Se enfoca en resumen de documentos y generación de informes técnicos.

## Dataset
CNN/DailyMail (Hugging Face `datasets`), usado para resumen de documentos largos.

## Estructura
- `src/`: Módulos de código (e.g., encoder, decoder).
- `notebooks/`: Notebooks para exploración y pruebas.
- `requirements.txt`: Dependencias del proyecto.

## Instalación
```bash
python3.9 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Estado
En progreso. Consultar commits diarios en la rama `develop`.

