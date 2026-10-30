# How to run PERO OCR and evaluate

1. Download the public PERO engine (layout + European printed OCR): https://nextcloud.fit.vutbr.cz/s/NtAbHTNkZFpapdJ  -> unzip it, e.g. into ./pero_engine/

2. Create a virtual environment using Python (3.10–3.12):
```bash
python -m venv ~/pero-env && source ~/pero-env/bin/activate
pip install pero-ocr opencv-python
```

3. Run from this folder (use the CPU config inside the engine folder; name may be config_cpu.ini or config.ini):
```bash
python run_pero.py --config pero_engine/<engine-folder>/config_cpu.ini
# python run_pero.py --config ../../../../pero_engine/pero_eu_cz_print_newspapers_2022-09-26/config_cpu.ini
```
The script writes `depechetoulouse-*/pero.txt` (plain text) and `depechetoulouse-*/pero_alto/rXX.xml` (page layout details).

4. Compare the OCR quality between PERO OCR and Tesseract OCR:
In each subfolder starting with `depechetoulouse`, two OCRed texts are generated and available : `pero.txt` (PERO OCR) and `text.txt` (Tesseract OCR). Run from this folder to evaluate them with the Impresso OCR-quality metric :
```bash
python eval_ocrqa.py  
```
(`pip install huggingface_hub pybloomfiltermmap3 matplotlib` in the same environment.)
This writes `ocrqa_scores.csv` (per issue), `ocrqa_summary.csv` (per engine: mean, median, ...),
`ocrqa_report.md` (results + explanation of every column) and `ocrqa_comparison.png` (per-issue chart).
To rebuild the report and chart without rescoring: `python eval_ocrqa.py --report-only`.
