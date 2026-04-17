"""
ocr_engine.py — version améliorée
Améliorations :
  1. detect_language() : seuil 60% + unicode étendu arabe/persan/ourdou
  2. Preprocessing arabe renforcé pour meilleure lecture
  3. OCR arabe sur image COMPLÈTE (pas seulement le masque)
     → capte les blocs arabes même dans les zones mixtes
  4. Re-classification post-OCR sur le texte réel
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

_ocr_fr = None
_ocr_ar = None


def get_ocr_fr():
    global _ocr_fr
    if _ocr_fr is None:
        from paddleocr import PaddleOCR
        _ocr_fr = PaddleOCR(use_angle_cls=True, lang="fr",
                             use_gpu=False, show_log=False, enable_mkldnn=False)
    return _ocr_fr


def get_ocr_ar():
    global _ocr_ar
    if _ocr_ar is None:
        from paddleocr import PaddleOCR
        _ocr_ar = PaddleOCR(use_angle_cls=True, lang="ar",
                             use_gpu=False, show_log=False, enable_mkldnn=False)
    return _ocr_ar


# ══════════════════════════════════════════════════════════════
# DÉTECTION DE LANGUE — SEUIL 60% + UNICODE ÉTENDU
# ══════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Retourne 'ar', 'fr', ou 'other'.
    Seuil relevé à 60% pour réduire les faux positifs.
    Couverture unicode étendue : arabe, persan, ourdou, présentation arabe.
    """
    if not text or not text.strip():
        return "other"

    def is_arabic_char(c):
        cp = ord(c)
        return (
            0x0600 <= cp <= 0x06FF or   # Arabe de base
            0x0750 <= cp <= 0x077F or   # Supplément arabe
            0xFB50 <= cp <= 0xFDFF or   # Présentation A
            0xFE70 <= cp <= 0xFEFF      # Présentation B
        )

    arabic = sum(1 for c in text if is_arabic_char(c))
    latin  = sum(1 for c in text if c.isalpha() and not is_arabic_char(c))
    digits = sum(1 for c in text if c.isdigit())
    total  = arabic + latin + digits

    if total == 0:
        return "other"

    # Présence de chiffres ou latin → FR (chiffres arabes-indiens sont neutres)
    if digits > 0 or latin > 0:
        return "fr"

    arabic_ratio = arabic / total
    if arabic_ratio >= 0.60:
        return "ar"

    return "other"


# ══════════════════════════════════════════════════════════════
# MASQUES IMAGE FR / AR
# ══════════════════════════════════════════════════════════════

def _has_arabic_pixels(region: np.ndarray, threshold: float = 0.08) -> bool:
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    right_half = binary[:, w // 2:]
    density = np.sum(right_half > 0) / (right_half.size + 1e-6)
    return density > threshold


def split_image_by_language(img_rgb: np.ndarray) -> dict:
    h, w   = img_rgb.shape[:2]
    band_h = max(40, h // 30)
    mask_ar = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, band_h):
        band = img_rgb[y: y + band_h, :]
        if _has_arabic_pixels(band):
            mask_ar[y: y + band_h, :] = 255

    img_fr = img_rgb.copy()
    img_fr[mask_ar == 255] = 255

    return {"fr": img_fr}


# ══════════════════════════════════════════════════════════════
# ORIENTATION
# ══════════════════════════════════════════════════════════════

def fix_orientation(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb


# ══════════════════════════════════════════════════════════════
# PRÉTRAITEMENT
# ══════════════════════════════════════════════════════════════

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if w < 1200:
        scale   = 1200 / w
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=7)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def preprocess_arabic(img_rgb: np.ndarray) -> np.ndarray:
    """
    Prétraitement renforcé pour l'arabe :
    upscaling plus agressif + contraste plus fort pour les petits caractères.
    """
    h, w = img_rgb.shape[:2]
    # Upscaling plus fort pour l'arabe (1600px min)
    if w < 1600:
        scale   = 1600 / w
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
    gray     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # CLAHE plus agressif pour l'arabe
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    enhanced = clahe.apply(denoised)
    # Sharpening léger
    kernel   = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)


# ══════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════

def load_file(path: str) -> list:
    p   = Path(path)
    ext = p.suffix.lower()
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    images = []
    if ext == ".pdf":
        from pdf2image import convert_from_path
        for page in convert_from_path(str(p), dpi=300):
            images.append(np.array(page.convert("RGB")))
    elif ext in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}:
        images.append(np.array(Image.open(str(p)).convert("RGB")))
    else:
        raise ValueError(f"Format non supporté : {ext}")
    return images


# ══════════════════════════════════════════════════════════════
# PARSER PADDLEOCR
# ══════════════════════════════════════════════════════════════

def _parse_paddle_result(result, lang: str) -> list:
    blocks = []
    if not result or not result[0]:
        return blocks
    for line in result[0]:
        bbox = line[0]
        text = line[1][0].strip()
        conf = round(float(line[1][1]), 4)
        if not text:
            continue
        if len(text) == 1 and conf < 0.5:
            continue
        blocks.append({
            "text":       text,
            "confidence": conf,
            "bbox":       [[int(p[0]), int(p[1])] for p in bbox],
            "x":          int(bbox[0][0]),
            "y":          int(bbox[0][1]),
            "lang":       lang,
        })
    return blocks


# ══════════════════════════════════════════════════════════════
# DÉDUPLICATION
# ══════════════════════════════════════════════════════════════

def _deduplicate(blocks_fr: list, blocks_ar: list, grid: int = 15) -> list:
    seen: dict = {}
    for block in blocks_fr + blocks_ar:
        key = (block["x"] // grid, block["y"] // grid)
        if key not in seen or block["confidence"] > seen[key]["confidence"]:
            seen[key] = block
    return list(seen.values())


# ══════════════════════════════════════════════════════════════
# RE-CLASSIFICATION POST-OCR
# ══════════════════════════════════════════════════════════════

def _reclassify_blocks(blocks: list) -> list:
    """
    Re-classifie chaque bloc avec detect_language() sur le texte réel.
    Les blocs arabes détectés par l'OCR arabe mais contenant des chiffres
    seront correctement reclassés FR.
    """
    for b in blocks:
        detected = detect_language(b["text"])
        # Si l'OCR arabe a trouvé ce bloc mais il ne contient que du latin/chiffres
        # → garder la classification de l'OCR (il a vu l'image, pas le texte)
        # Sinon on corrige
        if detected == "ar":
            b["lang"] = "ar"
        elif detected == "fr":
            b["lang"] = "fr"
        # Si "other", on garde la lang originale de l'OCR
    return blocks


# ══════════════════════════════════════════════════════════════
# OCR PRINCIPAL — ARABE SUR IMAGE COMPLÈTE
# ══════════════════════════════════════════════════════════════

def run_ocr(img_rgb: np.ndarray) -> list:
    # 1. Orientation
    img_oriented = fix_orientation(img_rgb)

    # 2. Prétraitement FR (image sans zones arabes masquées)
    split      = split_image_by_language(img_oriented)
    img_fr_prep = preprocess(split["fr"])

    # 3. Prétraitement AR — sur l'image COMPLÈTE pour ne rien rater
    #    Le modèle arabe est capable de distinguer AR/FR lui-même
    img_ar_prep = preprocess_arabic(img_oriented)

    # 4. OCR français
    result_fr = get_ocr_fr().ocr(img_fr_prep, cls=True)
    blocks_fr = _parse_paddle_result(result_fr, lang="fr")
    print(f"   🇫🇷 OCR FR : {len(blocks_fr)} blocs")

    # 5. OCR arabe (image complète)
    result_ar = get_ocr_ar().ocr(img_ar_prep, cls=True)
    blocks_ar = _parse_paddle_result(result_ar, lang="ar")
    print(f"   🇲🇦 OCR AR : {len(blocks_ar)} blocs")

    # 6. Fusion sans doublons
    all_blocks = _deduplicate(blocks_fr, blocks_ar, grid=15)

    # 7. Re-classification sur le TEXTE réel
    all_blocks = _reclassify_blocks(all_blocks)

    # 8. Tri spatial
    all_blocks.sort(key=lambda b: (round(b["y"] / 20) * 20, b["x"]))

    fr_count = sum(1 for b in all_blocks if b["lang"] == "fr")
    ar_count = sum(1 for b in all_blocks if b["lang"] == "ar")
    print(f"   ✅ Total : {len(all_blocks)} blocs (FR:{fr_count} AR:{ar_count})")
    return all_blocks


# ══════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════

def get_full_text(blocks: list, lang_filter: str = None) -> str:
    if lang_filter:
        blocks = [b for b in blocks if b.get("lang") == lang_filter]
    return " ".join(b["text"] for b in blocks)


def get_avg_confidence(blocks: list) -> float:
    if not blocks:
        return 0.0
    return round(float(np.mean([b["confidence"] for b in blocks])), 4)


def get_blocks_by_lang(blocks: list) -> dict:
    return {
        "fr": [b for b in blocks if b.get("lang") == "fr"],
        "ar": [b for b in blocks if b.get("lang") == "ar"],
    }
