"""
exporter.py — InvoiceAI-MA v2.0
CSV dynamique : seulement les champs présents dans la facture.
Bilingue : CSV Français + CSV Arabe.
"""

import io
import json
import csv
import os
from datetime import datetime


# ══════════════════════════════════════════════════════════════
# MAPPING DES CHAMPS — Français & Arabe
# ══════════════════════════════════════════════════════════════

FIELDS_FR = {
    "source_file":    "Fichier source",
    "processed_at":   "Date de traitement",
    "invoice_number": "N° Facture",
    "date":           "Date de facturation",
    "date_limite":    "Date limite de paiement",
    "supplier":       "Fournisseur",
    "client":         "Client",
    "agence":         "Agence",
    "contrat":        "N° Contrat",
    "tarif":          "Tarif",
    "periode":        "Période",
    "amount_ht":      "Montant HT",
    "tva_rate":       "TVA (%)",
    "tva_amount":     "Montant TVA",
    "amount_ttc":     "Montant TTC",
    "currency":       "Devise",
    "consommation":   "Consommation (kWh)",
    "email":          "Email",
    "phone":          "Téléphone",
    "ice":            "ICE",
    "rc":             "RC",
    "capital":        "Capital Social",
}

FIELDS_AR = {
    "source_file":    "الملف المصدر",
    "processed_at":   "تاريخ المعالجة",
    "invoice_number": "رقم الفاتورة",
    "date":           "تاريخ الفاتورة",
    "date_limite":    "تاريخ الاستحقاق",
    "supplier":       "المورد",
    "client":         "الزبون",
    "agence":         "الوكالة",
    "contrat":        "رقم العقد",
    "tarif":          "التعريفة",
    "periode":        "الفترة",
    "amount_ht":      "المبلغ بدون ضريبة",
    "tva_rate":       "نسبة الضريبة (%)",
    "tva_amount":     "مبلغ الضريبة",
    "amount_ttc":     "المبلغ الإجمالي",
    "currency":       "العملة",
    "consommation":   "الاستهلاك (كيلوواط/ساعة)",
    "email":          "البريد الإلكتروني",
    "phone":          "الهاتف",
    "ice":            "ICE",
    "rc":             "السجل التجاري",
    "capital":        "رأس المال الاجتماعي",
}

# Ordre des champs dans le CSV
FIELD_KEYS = list(FIELDS_FR.keys())


# ══════════════════════════════════════════════════════════════
# CONSTRUCTION DE LA LIGNE — DYNAMIQUE
# Seulement les champs qui ont une vraie valeur
# ══════════════════════════════════════════════════════════════

def _fmt(val) -> str:
    """Formate une valeur pour le CSV."""
    if val is None or val == "" or val == [] or val == {}:
        return "—"
    if isinstance(val, bool):
        return "Oui" if val else "Non"
    if isinstance(val, float):
        return f"{val:.2f}".rstrip("0").rstrip(".")
    return str(val).strip()


def _build_row(result: dict) -> dict:
    """Construit un dict de valeurs à partir du résultat complet."""
    data  = result.get("extracted_data", {})
    meta  = result.get("metadata", {})

    return {
        "source_file":    _fmt(meta.get("source_file")),
        "processed_at":   _fmt(meta.get("processed_at", "").replace("T", " ").split(".")[0]),
        "invoice_number": _fmt(data.get("invoice_number")),
        "date":           _fmt(data.get("date") or data.get("date_emission")),
        "date_limite":    _fmt(data.get("date_limite")),
        "supplier":       _fmt(data.get("supplier")),
        "client":         _fmt(data.get("client")),
        "agence":         _fmt(data.get("agence")),
        "contrat":        _fmt(data.get("contrat")),
        "tarif":          _fmt(data.get("tarif")),
        "periode":        _fmt(data.get("periode")),
        "amount_ht":      _fmt(data.get("amount_ht")),
        "tva_rate":       _fmt(data.get("tva_rate")),
        "tva_amount":     _fmt(data.get("tva_amount")),
        "amount_ttc":     _fmt(data.get("amount_ttc")),
        "currency":       _fmt(data.get("currency", "MAD")),
        "consommation":   _fmt(data.get("consommation") or data.get("consommation_kwh")),
        "email":          _fmt(data.get("email")),
        "phone":          _fmt(data.get("phone")),
        "ice":            _fmt(data.get("ice") or data.get("siret")),
        "rc":             _fmt(data.get("rc")),
        "capital":        _fmt(data.get("capital")),
    }


def _active_keys(row: dict) -> list:
    """Retourne seulement les clés qui ont une vraie valeur (pas '—')."""
    return [k for k in FIELD_KEYS if row.get(k, "—") != "—"]


# ══════════════════════════════════════════════════════════════
# CSV DYNAMIQUE — seulement les champs présents
# ══════════════════════════════════════════════════════════════

def to_csv_fr_bytes(result: dict) -> bytes:
    """CSV français — colonnes dynamiques (seulement champs détectés)."""
    row  = _build_row(result)
    keys = _active_keys(row)

    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow([FIELDS_FR[k] for k in keys])
    writer.writerow([row[k] for k in keys])
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def to_csv_ar_bytes(result: dict) -> bytes:
    """CSV arabe — colonnes dynamiques (seulement champs détectés)."""
    row  = _build_row(result)
    keys = _active_keys(row)

    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow([FIELDS_AR[k] for k in keys])
    writer.writerow([row[k] for k in keys])
    return ("\ufeff" + output.getvalue()).encode("utf-8")


# ══════════════════════════════════════════════════════════════
# CSV FICHIER (compatibilité pipeline.py)
# ══════════════════════════════════════════════════════════════

def to_csv(result: dict, output_path: str = "ocr_output.csv") -> str:
    with open(output_path, "wb") as f:
        f.write(to_csv_fr_bytes(result))
    return output_path


def to_csv_ar(result: dict, output_path: str = "ocr_output_ar.csv") -> str:
    with open(output_path, "wb") as f:
        f.write(to_csv_ar_bytes(result))
    return output_path


# ══════════════════════════════════════════════════════════════
# JSON
# ══════════════════════════════════════════════════════════════

def to_json(result: dict, output_path: str = "ocr_output.json") -> str:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return output_path


# ══════════════════════════════════════════════════════════════
# BUILD RESULT
# ══════════════════════════════════════════════════════════════

def build_result(
    source_file: str,
    ocr_blocks: list,
    full_text: str,
    extracted_data: dict,
    validation: dict,
    nb_pages: int,
) -> dict:
    return {
        "metadata": {
            "source_file":   os.path.basename(source_file),
            "processed_at":  datetime.now().isoformat(),
            "nb_pages":      nb_pages,
            "nb_ocr_blocks": len(ocr_blocks),
        },
        "full_text":      full_text,
        "extracted_data": extracted_data,
        "validation":     validation,
        "ocr_blocks":     ocr_blocks,
    }