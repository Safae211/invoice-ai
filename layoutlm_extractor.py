"""
layoutlm_extractor.py
Extraction par LayoutLMv3 — couche de compréhension documentaire.

LayoutLMv3 comprend le LAYOUT (position spatiale) + le TEXTE en même temps.
C'est pour ça qu'il est bien meilleur que du regex pur sur des factures
avec des structures variables (ONEE, factures privées, tickets, etc.)

Installation :
    pip install transformers torch Pillow

Modèle utilisé : microsoft/layoutlmv3-base (fine-tuned FUNSD/CORD)
Pour un meilleur résultat sur factures marocaines :
    → fine-tuner sur tes propres données après la compétition
"""

from __future__ import annotations
import re
import numpy as np
from PIL import Image

# ── Import optionnel ─────────────────────────────────────────
try:
    import torch
    from transformers import (
        LayoutLMv3Processor,
        LayoutLMv3ForTokenClassification,
    )
    HAS_LAYOUTLM = True
except ImportError:
    HAS_LAYOUTLM = False


# ── Modèle en singleton ───────────────────────────────────────
_processor = None
_model     = None
_LABELS    = None

# Labels CORD (modèle pré-entraîné sur reçus/factures)
CORD_LABELS = [
    "O",
    "B-MENU.NM", "I-MENU.NM",           # nom produit
    "B-MENU.NUM", "I-MENU.NUM",          # numéro produit
    "B-MENU.UNITPRICE", "I-MENU.UNITPRICE",  # prix unitaire
    "B-MENU.CNT", "I-MENU.CNT",          # quantité
    "B-MENU.DISCOUNTPRICE", "I-MENU.DISCOUNTPRICE",
    "B-MENU.PRICE", "I-MENU.PRICE",      # prix ligne
    "B-MENU.ITEMSUBTOTAL", "I-MENU.ITEMSUBTOTAL",
    "B-MENU.VATYN", "I-MENU.VATYN",
    "B-MENU.ETC", "I-MENU.ETC",
    "B-MENU.SUB_CNT", "I-MENU.SUB_CNT",
    "B-MENU.SUB_NM", "I-MENU.SUB_NM",
    "B-MENU.SUB_PRICE", "I-MENU.SUB_PRICE",
    "B-MENU.SUB_ETC", "I-MENU.SUB_ETC",
    "B-SUBTOTAL.SUBTOTAL_PRICE", "I-SUBTOTAL.SUBTOTAL_PRICE",  # sous-total
    "B-SUBTOTAL.DISCOUNT_PRICE", "I-SUBTOTAL.DISCOUNT_PRICE",
    "B-SUBTOTAL.SERVICE_PRICE", "I-SUBTOTAL.SERVICE_PRICE",
    "B-SUBTOTAL.OTHERSVC_PRICE", "I-SUBTOTAL.OTHERSVC_PRICE",
    "B-SUBTOTAL.TAX_PRICE", "I-SUBTOTAL.TAX_PRICE",           # TVA
    "B-SUBTOTAL.TIPS_PRICE", "I-SUBTOTAL.TIPS_PRICE",
    "B-SUBTOTAL.TOTAL_PRICE", "I-SUBTOTAL.TOTAL_PRICE",       # total
    "B-SUBTOTAL.PNT", "I-SUBTOTAL.PNT",
    "B-SUBTOTAL.ETC", "I-SUBTOTAL.ETC",
    "B-TOTAL.TOTAL_PRICE", "I-TOTAL.TOTAL_PRICE",             # TOTAL FINAL
    "B-TOTAL.TOTAL_ETC", "I-TOTAL.TOTAL_ETC",
    "B-TOTAL.CASHPRICE", "I-TOTAL.CASHPRICE",
    "B-TOTAL.CHANGEPRICE", "I-TOTAL.CHANGEPRICE",
    "B-TOTAL.CREDITCARDPRICE", "I-TOTAL.CREDITCARDPRICE",
    "B-TOTAL.EMONEYPRICE", "I-TOTAL.EMONEYPRICE",
    "B-TOTAL.MENUTYPE_CNT", "I-TOTAL.MENUTYPE_CNT",
    "B-TOTAL.MENU_CNT", "I-TOTAL.MENU_CNT",
    "B-TOTAL.VOID_MENU_PRICE", "I-TOTAL.VOID_MENU_PRICE",
    "B-TOTAL.DISCOUNT_PRICE", "I-TOTAL.DISCOUNT_PRICE",
]

# Mapping labels → champs InvoiceAI
LABEL_TO_FIELD = {
    "TOTAL.TOTAL_PRICE":         "amount_ttc",
    "SUBTOTAL.TOTAL_PRICE":      "amount_ttc",
    "SUBTOTAL.SUBTOTAL_PRICE":   "amount_ht",
    "SUBTOTAL.TAX_PRICE":        "tva_amount",
    "MENU.UNITPRICE":            "unit_price",
    "MENU.PRICE":                "line_total",
    "MENU.NM":                   "product_name",
    "MENU.CNT":                  "quantity",
}


def _get_model():
    global _processor, _model, _LABELS
    if _processor is None:
        print("   📦 Chargement LayoutLMv3 (première fois — ~1min)...")
        _processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False,   # On fournit notre propre OCR
        )
        _model = LayoutLMv3ForTokenClassification.from_pretrained(
            "jinhybr/OCR-LayoutLMv3-Invoice",   # Fine-tuné sur factures
            num_labels=len(CORD_LABELS),
            ignore_mismatched_sizes=True,
        )
        _model.eval()
        _LABELS = CORD_LABELS
        print("   ✅ LayoutLMv3 chargé")
    return _processor, _model


def _normalize_bbox(bbox: list, img_w: int, img_h: int) -> list:
    """
    Normalise une bbox PaddleOCR [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    vers le format LayoutLMv3 [x_min, y_min, x_max, y_max] en 0-1000.
    """
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    x_min = max(0, int(min(xs) / img_w * 1000))
    y_min = max(0, int(min(ys) / img_h * 1000))
    x_max = min(1000, int(max(xs) / img_w * 1000))
    y_max = min(1000, int(max(ys) / img_h * 1000))
    return [x_min, y_min, x_max, y_max]


def run_layoutlm(
    img_rgb: np.ndarray,
    ocr_blocks: list,
    lang_filter: str = "fr",
) -> dict:
    """
    Lance LayoutLMv3 sur l'image avec les blocs OCR fournis.

    Args:
        img_rgb      : image RGB (numpy array)
        ocr_blocks   : blocs OCR de ocr_engine.run_ocr()
        lang_filter  : 'fr' pour ne travailler que sur les blocs FR

    Returns:
        dict avec les champs extraits par LayoutLMv3
    """
    if not HAS_LAYOUTLM:
        return {
            "_layoutlm_available": False,
            "_layoutlm_error": (
                "transformers/torch non installés. "
                "pip install transformers torch"
            )
        }

    try:
        processor, model = _get_model()
    except Exception as e:
        return {"_layoutlm_available": False, "_layoutlm_error": str(e)}

    # ── Filtrer les blocs FR ──────────────────────────────────
    blocks = [b for b in ocr_blocks if b.get("lang", "fr") == lang_filter] \
             if lang_filter else ocr_blocks

    if not blocks:
        return {"_layoutlm_available": True, "_layoutlm_error": "Aucun bloc FR"}

    h, w   = img_rgb.shape[:2]
    pil_img = Image.fromarray(img_rgb)

    # ── Préparer words + boxes ────────────────────────────────
    words  = [b["text"] for b in blocks]
    boxes  = [_normalize_bbox(b["bbox"], w, h) for b in blocks]

    # ── Tokeniser ────────────────────────────────────────────
    try:
        encoding = processor(
            pil_img,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    except Exception as e:
        return {"_layoutlm_available": True, "_layoutlm_error": f"Tokenisation : {e}"}

    # ── Inférence ────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(**{k: v for k, v in encoding.items()
                           if k in ("input_ids", "attention_mask",
                                    "bbox", "pixel_values")})

    predictions  = outputs.logits.argmax(-1).squeeze().tolist()
    input_ids    = encoding["input_ids"].squeeze().tolist()
    tokens       = processor.tokenizer.convert_ids_to_tokens(input_ids)

    # ── Décoder les entités ───────────────────────────────────
    extracted: dict[str, list] = {}
    current_field = None
    current_text  = []

    for token, pred in zip(tokens, predictions):
        if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
            if current_field and current_text:
                extracted.setdefault(current_field, []).append(
                    " ".join(current_text)
                )
            current_field = None
            current_text  = []
            continue

        label = CORD_LABELS[pred] if pred < len(CORD_LABELS) else "O"

        if label == "O":
            if current_field and current_text:
                extracted.setdefault(current_field, []).append(
                    " ".join(current_text)
                )
            current_field = None
            current_text  = []
            continue

        # B- = début, I- = continuation
        bio, entity = label.split("-", 1) if "-" in label else ("O", "O")
        field = LABEL_TO_FIELD.get(entity)

        if bio == "B":
            if current_field and current_text:
                extracted.setdefault(current_field, []).append(
                    " ".join(current_text)
                )
            current_field = field
            current_text  = [token.replace("▁", "").replace("Ġ", "")]
        elif bio == "I" and current_field == field:
            current_text.append(token.replace("▁", "").replace("Ġ", ""))

    if current_field and current_text:
        extracted.setdefault(current_field, []).append(" ".join(current_text))

    # ── Convertir en résultat propre ──────────────────────────
    result = {"_layoutlm_available": True, "_layoutlm_error": None}

    from nlp_extractor import clean_amount

    # Montant TTC
    for candidate in extracted.get("amount_ttc", []):
        val = clean_amount(candidate)
        if val and 0 < val < 1_000_000:
            result["amount_ttc"] = val
            break

    # Montant HT
    for candidate in extracted.get("amount_ht", []):
        val = clean_amount(candidate)
        if val and 0 < val < 1_000_000:
            result["amount_ht"] = val
            break

    # TVA
    for candidate in extracted.get("tva_amount", []):
        val = clean_amount(candidate)
        if val and 0 < val < 100_000:
            result["tva_amount"] = val
            break

    # Lignes de produits
    names  = extracted.get("product_name", [])
    prices = extracted.get("unit_price", [])
    qtys   = extracted.get("quantity", [])
    totals = extracted.get("line_total", [])

    line_items = []
    for i in range(max(len(names), len(prices))):
        item = {}
        if i < len(names):  item["description"] = names[i]
        if i < len(prices): item["unit_price"]   = clean_amount(prices[i])
        if i < len(qtys):   item["quantity"]      = clean_amount(qtys[i])
        if i < len(totals): item["total"]          = clean_amount(totals[i])
        if item:
            line_items.append(item)

    if line_items:
        result["line_items_layoutlm"] = line_items

    result["_raw_entities"] = extracted
    return result


def merge_with_regex(regex_data: dict, layoutlm_data: dict) -> dict:
    """
    Fusionne les résultats regex et LayoutLMv3.
    Règle : LayoutLMv3 a priorité sur les montants si disponible,
    Regex a priorité sur invoice_number, date, email, ICE.
    """
    if not layoutlm_data.get("_layoutlm_available"):
        return regex_data  # LayoutLMv3 indispo → on garde tout le regex

    merged = dict(regex_data)  # copie

    # LayoutLMv3 override pour les montants
    for field in ("amount_ttc", "amount_ht", "tva_amount"):
        lm_val = layoutlm_data.get(field)
        if lm_val is not None:
            merged[field] = lm_val

    # Lignes de produits LayoutLMv3 (plus fiables)
    if layoutlm_data.get("line_items_layoutlm"):
        merged["line_items_layoutlm"] = layoutlm_data["line_items_layoutlm"]

    # Metadata
    merged["_layoutlm_raw"] = layoutlm_data.get("_raw_entities", {})
    merged["_layoutlm_used"] = True

    return merged