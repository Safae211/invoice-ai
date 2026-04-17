"""
pipeline.py — version corrigée
Correction : full_text_fr passé à extract_entities
"""

from ocr_engine import load_file, run_ocr, get_full_text, get_avg_confidence
from nlp_extractor import extract_entities, validate
from exporter import build_result, to_json, to_csv


def process_invoice(file_path: str, export_json=True, export_csv=True) -> dict:
    """
    Pipeline complet sur un fichier PDF ou image.
    Retourne le dict résultat.
    """
    # 1. Charger
    images = load_file(file_path)

    # 2. OCR sur toutes les pages
    all_blocks = []
    for img in images:
        blocks = run_ocr(img)
        all_blocks.extend(blocks)

    # 3. Textes — complet ET français uniquement
    full_text    = get_full_text(all_blocks)                    # pour export brut
    full_text_fr = get_full_text(all_blocks, lang_filter="fr")  # pour NLP
    avg_conf     = get_avg_confidence(all_blocks)

    # 4. Extraction NLP — texte FR uniquement
    extracted = extract_entities(
        ocr_blocks   = all_blocks,
        full_text    = full_text,
        full_text_fr = full_text_fr,
    )

    # 5. Validation métier
    report = validate(extracted, avg_conf)

    # 6. Construire résultat
    result = build_result(
        source_file    = file_path,
        ocr_blocks     = all_blocks,
        full_text      = full_text,
        extracted_data = extracted,
        validation     = report,
        nb_pages       = len(images),
    )

    # 7. Export
    base = file_path.rsplit(".", 1)[0]
    if export_json:
        to_json(result, base + "_ocr.json")
    if export_csv:
        to_csv(result, base + "_ocr.csv")

    return result