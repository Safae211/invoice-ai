"""
Llm_extractor.py — version améliorée
Modèle : gemini-2.5-flash (gratuit)
Corrections :
  - Prompt ultra-précis pour factures marocaines FR+AR
  - Extraction stricte du N° facture (ne prend pas un mot partiel)
  - Merge intelligent : un champ LLM écrase regex SEULEMENT si valide
  - Gestion d'erreur robuste avec extraction JSON automatique
"""

import re
import json
import time
import numpy as np
from PIL import Image
import io

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ─── Clé API & Modèle FREE ───────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"

# ─── Rate limit : 15 req/min → 1 req toutes les 4 secondes minimum ───────────
_RATE_LIMIT_DELAY = 4.5       # secondes entre chaque appel (sécurité ~13 req/min)
_MAX_RETRIES      = 3         # nb de tentatives en cas d'erreur 429
_last_call_time   = 0.0       # timestamp du dernier appel réussi

# ─── Prompt amélioré ─────────────────────────────────────────────────────────
INVOICE_PROMPT = """
Tu es un expert comptable spécialisé dans les factures marocaines (français + arabe).
Analyse cette image de facture et extrait EXACTEMENT les informations demandées.

RÈGLES STRICTES :
1. Retourne UNIQUEMENT un objet JSON valide, sans texte avant ni après, sans backticks.
2. Si un champ est absent ou illisible, mets null.
3. invoice_number : le numéro COMPLET de facture (ex: "2026204113211", "FAC-2024-001").
   - Cherche près des mots : N° Facture, Facture N°, Numéro, رقم الفاتورة
   - OBLIGATOIRE : doit contenir des chiffres et faire au moins 4 caractères
   - Ne prends PAS un mot partiel comme "ture", "Fac", "ect" — seulement le code complet
4. supplier : le nom COMPLET du fournisseur/émetteur (en haut de page, ex: "OFFICE NATIONAL DE L'EAU ET DE L'ELECTRICITE").
5. client : le nom COMPLET du client/destinataire. Cherche: Client, Destinataire, العميل, M./Mme.
   - Ne mets JAMAIS un fragment comme "Tal" seul — prends le nom complet (prénom + nom).
6. date : format YYYY-MM-DD uniquement.
7. amount_ttc : montant total TTC, nombre décimal pur sans symbole ni unité.
8. currency : "MAD" ou "DHS" ou "EUR". Défaut si non trouvé : "MAD".
9. Montants = nombres décimaux purs (ex: 40.2 et non "40.2 DHS").

JSON à retourner (null obligatoire si absent) :
{
  "invoice_number": null,
  "date": null,
  "date_limite": null,
  "supplier": null,
  "client": null,
  "amount_ht": null,
  "tva_rate": null,
  "tva_amount": null,
  "amount_ttc": null,
  "currency": "MAD",
  "email": null,
  "phone": null,
  "ice": null,
  "rc": null,
  "capital": null,
  "agence": null,
  "contrat": null,
  "tarif": null,
  "periode": null,
  "consommation_kwh": null,
  "confidence": 0.95
}
"""

# ─── Validateurs ─────────────────────────────────────────────────────────────

def _is_valid_invoice_number(val) -> bool:
    if not val or not isinstance(val, str):
        return False
    val = val.strip()
    if len(val) < 4:
        return False
    if not any(c.isdigit() for c in val):
        return False
    STOP_WORDS = {"ture", "fact", "fac", "inv", "true", "none", "null", "ect"}
    if val.lower() in STOP_WORDS:
        return False
    return True

def _is_valid_name(val) -> bool:
    if not val or not isinstance(val, str):
        return False
    val = val.strip()
    if len(val) < 3 or not any(c.isalpha() for c in val):
        return False
    # Rejette fragments courts en minuscule
    if len(val) <= 4 and val[0].islower():
        return False
    return True

def _is_valid_amount(val) -> bool:
    if val is None:
        return False
    try:
        v = float(val)
        return 0 < v < 1_000_000_000
    except (ValueError, TypeError):
        return False

# ─── Appel Gemini avec rate limit ────────────────────────────────────────────

def _wait_for_rate_limit():
    """Attend le délai minimum entre deux appels API."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _RATE_LIMIT_DELAY:
        wait = _RATE_LIMIT_DELAY - elapsed
        print(f"   ⏳ Rate limit : attente {wait:.1f}s avant appel Gemini...")
        time.sleep(wait)


def run_gemini(img_rgb: np.ndarray, api_key: str = None) -> dict:
    global _last_call_time

    if not HAS_GEMINI:
        return {"_llm_available": False, "_llm_error": "pip install google-genai"}

    key = api_key or GEMINI_API_KEY

    # Préparer l'image une seule fois (évite de re-encoder à chaque retry)
    try:
        pil_img   = Image.fromarray(img_rgb.astype(np.uint8))
        buf       = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()
    except Exception as e:
        return {"_llm_available": False, "_llm_error": f"Erreur encodage image : {e}"}

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        # ── Respecter le délai minimum ────────────────────────
        _wait_for_rate_limit()

        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    INVOICE_PROMPT,
                ],
            )
            _last_call_time = time.time()  # ← mémoriser l'heure de l'appel réussi

            raw = response.text.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*",     "", raw)

            json_match = re.search(r"\{[\s\S]*\}", raw)
            if json_match:
                raw = json_match.group(0)

            data = json.loads(raw.strip())
            data["_llm_available"] = True
            data["_llm_error"]     = None
            data["_llm_model"]     = GEMINI_MODEL
            return data

        except json.JSONDecodeError as e:
            _last_call_time = time.time()
            return {"_llm_available": False, "_llm_error": f"JSON invalide: {e}"}

        except Exception as e:
            err_str = str(e)
            last_error = err_str

            # Détecter une erreur 429 (rate limit dépassé)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                # Backoff exponentiel : 10s, 20s, 40s
                wait_time = 10 * (2 ** (attempt - 1))
                print(f"   ⚠️ Rate limit Gemini (tentative {attempt}/{_MAX_RETRIES}) — "
                      f"attente {wait_time}s...")
                time.sleep(wait_time)
                continue  # → retry
            else:
                # Erreur non liée au rate limit → pas la peine de réessayer
                return {"_llm_available": False, "_llm_error": err_str}

    # Toutes les tentatives ont échoué
    return {
        "_llm_available": False,
        "_llm_error": f"Rate limit persistant après {_MAX_RETRIES} tentatives : {last_error}"
    }

# ─── Merge intelligent ───────────────────────────────────────────────────────

def merge_llm_with_regex(regex_data: dict, llm_data: dict) -> dict:
    if not llm_data.get("_llm_available") or llm_data.get("_llm_error"):
        return regex_data

    merged = dict(regex_data)

    # Champs textuels avec validation stricte
    if _is_valid_invoice_number(llm_data.get("invoice_number")):
        merged["invoice_number"] = llm_data["invoice_number"].strip()

    if _is_valid_name(llm_data.get("supplier")):
        merged["supplier"] = llm_data["supplier"].strip()

    if _is_valid_name(llm_data.get("client")):
        merged["client"] = llm_data["client"].strip()

    # Champs textuels simples
    for field in ("date", "date_limite", "email", "phone",
                  "ice", "rc", "capital", "agence", "contrat", "tarif", "periode"):
        val = llm_data.get(field)
        if val and isinstance(val, str) and len(val.strip()) >= 3:
            merged[field] = val.strip()

    # Montants
    for field in ("amount_ht", "tva_amount", "amount_ttc"):
        val = llm_data.get(field)
        if _is_valid_amount(val):
            merged[field] = float(val)

    # Taux TVA
    tva_rate = llm_data.get("tva_rate")
    if tva_rate is not None:
        try:
            rate = float(tva_rate)
            if 0 < rate <= 100:
                merged["tva_rate"] = rate
        except (ValueError, TypeError):
            pass

    # Devise
    currency = llm_data.get("currency")
    if currency and isinstance(currency, str) and len(currency) >= 2:
        merged["currency"] = currency.upper()

    # Consommation kWh
    kwh = llm_data.get("consommation_kwh")
    if _is_valid_amount(kwh):
        merged["consommation"] = float(kwh)

    merged["_llm_used"]       = True
    merged["_llm_model"]      = GEMINI_MODEL
    merged["_llm_confidence"] = llm_data.get("confidence")
    return merged

# ─── Point d'entrée utilisé par nlp_extractor.py ────────────────────────────

def extract_with_llm(img_rgb, regex_data, api_key=None):
    llm_data = run_gemini(img_rgb, api_key=api_key)
    return merge_llm_with_regex(regex_data, llm_data)