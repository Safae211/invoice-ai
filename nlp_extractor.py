"""
nlp_extractor.py — version finale
Pipeline : Regex → spaCy NER → Gemini LLM (si activé)
"""

import re
import numpy as np
from datetime import datetime

try:
    from dateutil import parser as date_parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

try:
    import spacy
    try:
        _nlp = spacy.load("fr_core_news_sm")
        SPACY_LANG = "fr"
    except OSError:
        try:
            _nlp = spacy.load("en_core_web_sm")
            SPACY_LANG = "en"
        except OSError:
            _nlp = None
            SPACY_LANG = None
    HAS_SPACY = _nlp is not None
except ImportError:
    HAS_SPACY = False
    _nlp      = None
    SPACY_LANG = None


# ════════════════════════════════════════════════════════════
# DÉTECTION DE LANGUE
# ════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    if not text or not text.strip():
        return "other"
    arabic = sum(1 for c in text
                 if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
    latin  = sum(1 for c in text if c.isalpha() and ord(c) < 0x0600)
    digits = sum(1 for c in text if c.isdigit())
    total  = arabic + latin + digits
    if total == 0:
        return "other"
    if (arabic / total) > 0.40 and arabic > latin and digits == 0:
        return "ar"
    if latin > 0 or digits > 0:
        return "fr"
    if (arabic / total) > 0.40 and arabic > latin:
        return "ar"
    return "other"


def split_blocks_by_language(ocr_blocks: list) -> dict:
    fr, ar = [], []
    for b in ocr_blocks:
        (ar if detect_language(b["text"]) == "ar" else fr).append(b)
    return {"fr": fr, "ar": ar}


# ════════════════════════════════════════════════════════════
# MOTS-CLÉS
# ════════════════════════════════════════════════════════════

TOTAL_KEYWORDS = [
    "montant facture", "total ttc", "montant ttc", "net à payer",
    "total à payer", "montant dû", "ttc", "total amount", "grand total",
    "amount due", "net total", "total due", "balance due", "total",
]
SUBTOTAL_KEYWORDS = [
    "montant ht", "hors taxe", "sous-total", "ht",
    "subtotal", "before tax", "net amount",
]
TVA_KEYWORDS = ["tva", "t.v.a", "taxe", "vat", "tax"]

MONTHS_FR    = r"janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre"
MONTHS_EN    = r"january|february|march|april|may|june|july|august|september|october|november|december"


# ════════════════════════════════════════════════════════════
# PATTERNS REGEX
# ════════════════════════════════════════════════════════════

PATTERNS = {
    "invoice_number": [
        r"(?:facture|invoice|fact\.?|fac\.?)\s*(?:n[°o]?\.?)?\s*:?\s*([A-Z0-9][A-Z0-9\-/_.]{3,20})",
        r"\b((?:FAC|INV|FC|FACT|CMD|PO)[-\s]?[\d]{4,})\b",
        r"n[°o]?\s*:?\s*([A-Z0-9][A-Z0-9\-/_.]{5,20})",
    ],
    "date": [
        r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4})\b",
        r"\b(\d{4}[\-/]\d{2}[\-/]\d{2})\b",
        rf"\b(\d{{1,2}}\s+(?:{MONTHS_FR})\s+\d{{4}})\b",
    ],
    "date_emission": [
        r"(?:emis[e]?\s+le|Emis\s+le|émise?\s+le)\s*[:\s]*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
    ],
    "date_limite": [
        r"(?:date\s+limite|échéance|date\s+limite\s+de\s+paiement)\s*:?\s*(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})",
    ],
    "amount_ttc": [
        r"MONTANT\s+FACTURE\s*[\n\r:]*\s*([\d]+[,.][\d]{1,2})\s*(?:Dhs?|MAD|DH)",
        r"NET\s+[ÀA]\s+PAYER\s*:?\s*([\d\s,.']+)",
        r"TOTAL\s+TTC\s*:?\s*([\d\s,.']+)",
        r"(?:montant\s+ttc|total\s+ttc)\s*:?\s*([\d\s,.'\u00a0]+)",
    ],
    "amount_ht": [
        r"(?:montant\s+)?h\.?t\.?\s*:?\s*([\d\s,.'\u00a0]+)",
        r"(?:sous[\s\-]?total|subtotal)\s*:?\s*([\d\s,.'\u00a0]+)",
    ],
    "tva_amount": [
        r"t\.?v\.?a\.?\s*(?:\d{1,2}\s*%)?\s*:?\s*([\d\s,.'\u00a0]+)",
    ],
    "tva_rate": [
        r"t\.?v\.?a\.?\s*:?\s*(\d{1,2})\s*%",
        r"(\d{1,2})\s*%\s*(?:tva|vat|tax)",
    ],
    "email":   [r"\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b"],
    "phone":   [
        # Téléphone marocain : 0X XX XX XX XX ou +212 X XX XX XX XX
        # On s'arrête aux "/" pour éviter "0535.62.50.15/16/17" → prend seulement le 1er numéro
        r"(?:tél\.?|tel\.?|téléphone|telephone|gsm|portable)\s*:?\s*(\+?[\d\s\-().]{8,18})(?=[^\d/]|$)",
        # Fallback : numéro marocain seul (commence par 05,06,07 ou +212)
        r"(\+212[\s\-]?[67][\d\s\-]{8,12})",
        r"\b(0[5-7][\d\s.\-]{7,14})\b",
    ],
    "website": [r"\b((?:www\.)[a-zA-Z0-9\-]+\.[a-zA-Z]{2,})\b"],
    "ice":     [r"\bICE\s*:?\s*(\d{15})\b"],
    "rc":      [r"\bRC\s*:?\s*([A-Z0-9\-]{3,15})\b"],
    "if_fiscal": [r"\bIF\s*:?\s*(\d{5,15})\b"],
    "capital": [r"(?:capital\s*(?:social)?)\s*:?\s*([\d\s,.]+\s*(?:MAD|DH|EUR)?)"],
    "agence":  [r"(?:agence)\s*:?\s*([A-Z][A-Z0-9\s\-]{2,30})"],
    "contrat": [r"(?:contrat\s*n[°o]?)\s*:?\s*([A-Z0-9\-]{4,20})"],
    "tarif":   [r"(?:tarif)\s*:?\s*([A-Z][A-Z0-9\s\-]{2,30})"],
    "consommation": [r"consommation\s*:?\s*([\d\s,.]+)\s*(?:kwh|kw)"],
    "periode": [r"(?:période|periode)\s*:?\s*([\d/\-\s]+(?:au|-)[\d/\-\s]+)"],
}


# ════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════

def clean_amount(raw: str):
    if not raw:
        return None
    s = str(raw).replace("\u00a0", "").replace(" ", "").strip()
    s = re.sub(r"(?i)(dhs?|mad|eur|€|\$|£)$", "", s).strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.index(".") < s.index(",") else s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", ".") if len(parts) == 2 and len(parts[1]) <= 2 else s.replace(",", "")
    s = re.sub(r"[^\d.]", "", s)
    try:
        val = float(s)
        return val if 0 < val < 1_000_000_000 else None
    except ValueError:
        return None


def apply_patterns(patterns: list, text: str):
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


def normalize_date(raw: str):
    if not raw:
        return None
    if raw.count('/') > 2 or raw.count('-') > 3:
        return None
    raw   = raw.strip()
    parts = re.split(r'[/\-.]', raw)
    if len(parts) == 3:
        try:
            p0, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2])
            if p0 > 31:
                year, month, day = p0, p1, p2
            elif p2 > 31:
                day, month, year = p0, p1, p2
                if year < 100:
                    year += 2000
            else:
                return None
            if not (1 <= day <= 31 and 1 <= month <= 12 and 2000 <= year <= 2035):
                return None
            return f"{year:04d}-{month:02d}-{day:02d}"
        except (ValueError, IndexError):
            pass
    if HAS_DATEUTIL:
        try:
            dt = date_parser.parse(raw, dayfirst=True)
            if 2000 <= dt.year <= 2035:
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def merge_line_blocks(ocr_blocks: list, y_tolerance: int = 12) -> list:
    if not ocr_blocks:
        return []
    sorted_blocks = sorted(ocr_blocks, key=lambda b: (b["y"], b["x"]))
    merged, used  = [], set()
    for i, b in enumerate(sorted_blocks):
        if i in used:
            continue
        same_line = [b]
        used.add(i)
        for j, other in enumerate(sorted_blocks):
            if j in used:
                continue
            if abs(other["y"] - b["y"]) <= y_tolerance:
                same_line.append(other)
                used.add(j)
        same_line.sort(key=lambda r: r["x"])
        merged.append({
            "text":       " ".join(r["text"] for r in same_line),
            "confidence": round(float(sum(r["confidence"] for r in same_line) / len(same_line)), 4),
            "bbox":       same_line[0]["bbox"],
            "x":          same_line[0]["x"],
            "y":          same_line[0]["y"],
            "lang":       same_line[0].get("lang", "fr"),
            "blocks_raw": same_line,
        })
    return merged


# ════════════════════════════════════════════════════════════
# SEARCH POSITIONAL
# ════════════════════════════════════════════════════════════

def find_amount_by_keyword(ocr_blocks, keywords, take_last=False, max_value=1e6):
    matches = []
    for i, b in enumerate(ocr_blocks):
        tl = b["text"].lower().strip()
        for kw in keywords:
            if kw in tl:
                matches.append((i, b))
                break
    if not matches:
        return None
    targets = [matches[-1]] if take_last else matches
    for (_, kb) in targets:
        same_line = [
            b for b in ocr_blocks
            if abs(b["y"] - kb["y"]) <= 20 and b["x"] > kb["x"]
            and re.match(r"^-?[\d][\d\s,.']*(?:dhs?|mad)?$", b["text"].strip(), re.IGNORECASE)
        ]
        if same_line:
            val = clean_amount(max(same_line, key=lambda b: b["x"])["text"])
            if val and val <= max_value:
                return val
        below = [
            b for b in ocr_blocks
            if 0 < b["y"] - kb["y"] <= 35
            and re.match(r"^-?[\d][\d\s,.']*(?:dhs?|mad)?$", b["text"].strip(), re.IGNORECASE)
        ]
        if below:
            val = clean_amount(min(below, key=lambda b: abs(b["y"] - kb["y"]))["text"])
            if val and val <= max_value:
                return val
    return None


def find_largest_amount(ocr_blocks, max_value=50_000):
    numbers = [clean_amount(b["text"]) for b in ocr_blocks]
    numbers = [v for v in numbers if v and 1 < v <= max_value]
    return max(numbers) if numbers else None


# ════════════════════════════════════════════════════════════
# SPACY NER
# ════════════════════════════════════════════════════════════

def run_spacy_ner(text_fr: str) -> dict:
    if not HAS_SPACY or not _nlp or not text_fr.strip():
        return {}
    doc    = _nlp(text_fr[:5000])
    result = {"ner_org": [], "ner_per": [], "ner_loc": [], "ner_date": [], "ner_money": []}
    for ent in doc.ents:
        t = ent.text.strip()
        if len(t) < 2:
            continue
        if ent.label_ == "ORG":
            result["ner_org"].append(t)
        elif ent.label_ in ("PER", "PERSON"):
            result["ner_per"].append(t)
        elif ent.label_ in ("LOC", "GPE"):
            result["ner_loc"].append(t)
        elif ent.label_ == "DATE":
            result["ner_date"].append(t)
        elif ent.label_ in ("MONEY", "CARDINAL"):
            val = clean_amount(t)
            if val:
                result["ner_money"].append({"text": t, "value": val})
    for k in ("ner_org", "ner_per", "ner_loc", "ner_date"):
        result[k] = list(dict.fromkeys(result[k]))
    return result


# ════════════════════════════════════════════════════════════
# PAIRES CLÉ-VALEUR
# ════════════════════════════════════════════════════════════

def extract_key_value_pairs(ocr_blocks: list) -> list:
    fr_blocks = [b for b in ocr_blocks if b.get("lang") != "ar"]
    lines     = merge_line_blocks(fr_blocks, y_tolerance=14)
    pairs, used = [], set()
    key_kws   = [
        "agence", "contrat", "tarif", "emise", "émise", "client", "tournée",
        "période", "consommation", "montant", "date", "n°", "numéro", "ref",
        "adresse", "tel", "tél", "fax", "ice", "rc", "if", "capital",
        "siège", "facture", "nom", "prénom", "raison sociale", "société",
    ]
    for i, line in enumerate(lines):
        if i in used or not any(kw in line["text"].lower() for kw in key_kws):
            continue
        raw = line.get("blocks_raw", [line])
        if len(raw) >= 2:
            pairs.append({"key": raw[0]["text"].strip(),
                          "value": " ".join(r["text"] for r in raw[1:]).strip(),
                          "x": raw[0]["x"], "y": raw[0]["y"]})
            used.add(i)
        else:
            for j, other in enumerate(lines):
                if j in used or j == i:
                    continue
                if abs(other["y"] - line["y"]) <= 14 and other["x"] > line["x"]:
                    pairs.append({"key": line["text"].strip(),
                                  "value": other["text"].strip(),
                                  "x": line["x"], "y": line["y"]})
                    used.add(i)
                    used.add(j)
                    break
    return pairs


# ════════════════════════════════════════════════════════════
# FOURNISSEUR & CLIENT
# ════════════════════════════════════════════════════════════

def extract_supplier_client(ocr_blocks, full_text_fr, ner_data=None):
    supplier, client = None, None
    fr_blocks = [b for b in ocr_blocks if b.get("lang") != "ar"]
    if not fr_blocks:
        return {"supplier": None, "client": None}
    sorted_b  = sorted(fr_blocks, key=lambda r: r["y"])
    top_y     = max(b["y"] for b in fr_blocks) * 0.22
    SKIP      = {"facture", "invoice", "date", "n°", "ref", "tel", "tél",
                 "email", "www", "page", "agence", "contrat", "tarif",
                 "client", "consommation", "capital", "siège"}
    top = [
        b for b in sorted_b
        if b["y"] <= top_y and b["confidence"] > 0.65 and len(b["text"]) > 4
        and detect_language(b["text"]) == "fr"
        and not re.match(r"^\d+[\d\s.,/\-]*$", b["text"])
        and not re.match(r".*@.*\.", b["text"])
        and not any(s in b["text"].lower() for s in SKIP)
    ]
    if top:
        supplier = max(top, key=lambda b: len(b["text"]))["text"]
    if not supplier and ner_data and ner_data.get("ner_org"):
        supplier = ner_data["ner_org"][0]

    m = re.search(
        r"(?:client\s*n°?|destinataire|facturé\s*[àa]|bill\s+to)"
        r"\s*:?\s*([A-ZÀ-Ÿa-zà-ÿ][^\n\r]{3,60})",
        full_text_fr, re.IGNORECASE,
    )
    if m:
        client = m.group(1).strip()
    if not client and ner_data and ner_data.get("ner_per"):
        client = ner_data["ner_per"][0]

    return {"supplier": supplier, "client": client}


# ════════════════════════════════════════════════════════════
# CONSOMMATION & LIGNES PRODUITS
# ════════════════════════════════════════════════════════════

def extract_consumption_table(ocr_blocks):
    table, dp, np_ = [], re.compile(r"\b(\d{1,2}/\d{2,4})\b"), re.compile(r"^\d+[,.]?\d*$")
    rows = {}
    for b in ocr_blocks:
        rows.setdefault(round(b["y"] / 15) * 15, []).append(b)
    for y_key in sorted(rows):
        row   = sorted(rows[y_key], key=lambda r: r["x"])
        texts = [r["text"] for r in row]
        dates = [t for t in texts if dp.match(t)]
        nums  = [t for t in texts if np_.match(t)]
        if dates and nums:
            for d, n in zip(dates, nums):
                val = clean_amount(n)
                if val:
                    table.append({"date": d, "consommation_kwh": val})
    return table


def extract_line_items(ocr_blocks):
    items    = []
    skip_kws = ["total", "tva", "ttc", "ht", "sous-total", "net", "subtotal",
                "montant", "facture", "agence", "capital", "tel", "ice"]
    merged   = merge_line_blocks([b for b in ocr_blocks if b.get("lang") != "ar"], y_tolerance=12)
    rows = {}
    for b in merged:
        rows.setdefault(round(b["y"] / 15) * 15, []).append(b)
    for y_key in sorted(rows):
        row   = sorted(rows[y_key], key=lambda r: r["x"])
        texts = [r["text"] for r in row]
        nums  = [t for t in texts if re.match(r"^-?[\d][\d\s,.']*$", t)]
        words = [t for t in texts if not re.match(r"^-?[\d][\d\s,.']*$", t) and len(t) > 1]
        if len(nums) >= 1 and words:
            desc = " ".join(words)
            if any(kw in desc.lower() for kw in skip_kws) or len(desc) > 120:
                continue
            amounts = [clean_amount(n) for n in nums if clean_amount(n)]
            item    = {"description": desc}
            if len(amounts) >= 3:
                item.update({"quantity": amounts[0], "unit_price": amounts[1], "total": amounts[-1]})
            elif len(amounts) == 2:
                item.update({"quantity": amounts[0], "unit_price": amounts[1], "total": None})
            elif len(amounts) == 1:
                item.update({"unit_price": amounts[0], "quantity": None, "total": None})
            items.append(item)
    return items


# ════════════════════════════════════════════════════════════
# EXTRACTION PRINCIPALE
# ════════════════════════════════════════════════════════════

def extract_entities(ocr_blocks: list, full_text: str,
                     image_size: tuple = (1000, 1000),
                     full_text_fr: str = None,
                     img_rgb=None,
                     use_llm: bool = False,
                     gemini_api_key: str = None) -> dict:
    data = {}

    if full_text_fr is None:
        full_text_fr = " ".join(b["text"] for b in ocr_blocks if b.get("lang") != "ar")

    # ── Étape 1 : Regex ──────────────────────────────────────
    for field, patterns in PATTERNS.items():
        raw = apply_patterns(patterns, full_text_fr)
        if field in ("amount_ht", "amount_ttc", "tva_amount", "consommation"):
            data[field] = clean_amount(raw)
        elif field == "tva_rate":
            try:
                data[field] = float(raw) if raw else None
            except Exception:
                data[field] = None
        elif field in ("date", "date_emission", "date_limite"):
            data[field] = normalize_date(raw)
        else:
            data[field] = raw

    # ── Nettoyage téléphone ───────────────────────────────────
    # Coupe les suffixes type "/16/17" (ex: 0535.62.50.15/16/17 → 0535.62.50.15)
    if data.get("phone"):
        phone_clean = re.split(r"/\d{2,}", data["phone"])[0].strip().rstrip(".")
        data["phone"] = phone_clean

    fr_blocks = [b for b in ocr_blocks if b.get("lang") != "ar"]

    # ── Étape 2 : Fallback positionnel ───────────────────────
    if not data.get("amount_ttc"):
        data["amount_ttc"] = find_amount_by_keyword(fr_blocks, TOTAL_KEYWORDS, take_last=True)
    if not data.get("amount_ht"):
        data["amount_ht"] = find_amount_by_keyword(fr_blocks, SUBTOTAL_KEYWORDS)
    if not data.get("tva_amount"):
        data["tva_amount"] = find_amount_by_keyword(fr_blocks, TVA_KEYWORDS)
    if not data.get("amount_ttc"):
        data["amount_ttc"] = find_largest_amount(fr_blocks)

    # ── Étape 3 : spaCy NER ──────────────────────────────────
    ner_data    = run_spacy_ner(full_text_fr)
    data["ner"] = ner_data

    # ── Étape 4 : Fournisseur & Client ───────────────────────
    merged_fr = merge_line_blocks(fr_blocks, y_tolerance=14)
    data.update(extract_supplier_client(merged_fr, full_text_fr, ner_data))

    # ── Validation N° facture ────────────────────────────────
    inv = data.get("invoice_number", "")
    if inv and (len(inv) < 4 or inv.lower() in {"de", "le", "la", "du", "en", "et"}):
        data["invoice_number"] = None
    if not data.get("date") and ner_data.get("ner_date"):
        data["date"] = normalize_date(ner_data["ner_date"][0])

    # ── Étape 5 : Gemini LLM (si activé) ─────────────────────
    if use_llm and img_rgb is not None:
        try:
            from Llm_extractor import extract_with_llm
            data = extract_with_llm(img_rgb, data, api_key=gemini_api_key)
        except ImportError:
            pass
        except Exception as e:
            print(f"   ⚠️ Gemini erreur : {e}")

    # ── Autres champs ─────────────────────────────────────────
    data["structured_fields"] = extract_key_value_pairs(ocr_blocks)
    data["consumption_table"] = extract_consumption_table(fr_blocks)
    data["line_items"]        = extract_line_items(ocr_blocks)

    lang_split        = split_blocks_by_language(ocr_blocks)
    data["fr_blocks"] = lang_split["fr"]
    data["ar_blocks"] = lang_split["ar"]

    cur = re.search(
        r"\b(MAD|EUR|USD|DH|DHS|TND|DZD)\b|([€\$£])",
        full_text_fr, re.IGNORECASE,
    )
    data["currency"] = (cur.group(1) or cur.group(2)).upper() if cur else "MAD"

    return data


# ════════════════════════════════════════════════════════════
# VALIDATION
# ════════════════════════════════════════════════════════════

def validate(data: dict, avg_confidence: float) -> dict:
    report = {
        "avg_confidence":  avg_confidence,
        "auto_validated":  avg_confidence >= 0.80,
        "math_check":      None,
        "warnings":        [],
        "spacy_available": HAS_SPACY,
    }
    ht, tva, ttc, rate = (data.get(k) for k in
                          ("amount_ht", "tva_amount", "amount_ttc", "tva_rate"))

    if ht and ttc:
        if tva:
            calc = ht + tva
            report["math_check"] = (f"✅ {ht} + {tva} ≈ {ttc}"
                                    if abs(calc - ttc) <= ttc * 0.02
                                    else f"⚠️ {ht} + {tva} = {calc:.2f} ≠ {ttc}")
            if abs(calc - ttc) > ttc * 0.02:
                report["warnings"].append("Incohérence HT + TVA ≠ TTC")
        elif rate:
            calc_tva = round(ht * (rate / 100), 2)
            calc_ttc = round(ht + calc_tva, 2)
            data["tva_amount_calculated"] = calc_tva
            report["math_check"] = (f"✅ {ht} × (1 + {rate}%) ≈ {ttc}"
                                    if abs(calc_ttc - ttc) <= ttc * 0.02
                                    else f"⚠️ Écart : {abs(calc_ttc - ttc):.2f}")
        else:
            report["math_check"] = f"ℹ️ Sous-total : {ht} → Total : {ttc}"
    else:
        if not ht:  report["warnings"].append("Montant HT non détecté")
        if not ttc: report["warnings"].append("Montant TTC non détecté")

    if not data.get("date") and not data.get("date_emission"):
        report["warnings"].append("Date non détectée")
    if not data.get("invoice_number"):
        report["warnings"].append("Numéro de facture non détecté")

    return report