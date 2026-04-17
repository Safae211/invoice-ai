"""
app.py — InvoiceAI-MA v2.0
Extraction intelligente de documents financiers marocains
Innov'Dom Challenge 2026 · ENSA Fès
"""

import streamlit as st
import numpy as np
import json
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

from ocr_engine import (
    load_file, run_ocr, get_full_text,
    get_avg_confidence, preprocess, get_blocks_by_lang
)
from nlp_extractor import extract_entities, validate
from exporter import build_result, to_json, to_csv, to_csv_fr_bytes, to_csv_ar_bytes
from exporter import FIELDS_FR, FIELD_KEYS, _build_row

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="InvoiceAI-MA",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');
textarea, input[type="text"] { unicode-bidi: plaintext !important; }
.stDataFrame td { unicode-bidi: plaintext !important; }
.arabic-block {
    direction: rtl; text-align: right;
    font-family: 'Amiri', Arial, sans-serif;
    font-size: 16px; line-height: 2.4; color: #e0e0e0;
    background: #1a1a2e; border: 1px solid #2d2d4e;
    border-radius: 10px; padding: 18px 22px; min-height: 180px;
    unicode-bidi: embed;
}
.french-block {
    direction: ltr; text-align: left; font-size: 14px; line-height: 1.9;
    color: #e0e0e0; background: #1a1a2e; border: 1px solid #2d2d4e;
    border-radius: 10px; padding: 18px 22px; min-height: 180px; word-break: break-word;
}
.field-card {
    background: #1e1e2e; border-radius: 8px; padding: 12px 16px;
    margin-bottom: 8px; border-left: 3px solid #4CAF50;
}
.field-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
.field-value { font-size: 16px; color: #fff; font-weight: 500; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Compteur de factures traitées (session uniquement — réinitialisé à chaque visite) ──
if "invoice_count" not in st.session_state:
    st.session_state.invoice_count = 0   # Toujours 0 à l'ouverture du site
if "last_processed" not in st.session_state:
    st.session_state.last_processed = None

# ── Header ─────────────────────────────────────────────────
col_title, col_counter = st.columns([3, 1])
with col_title:
    st.title("🧾 InvoiceAI-MA")
    st.caption("Extraction intelligente de documents financiers marocains · PaddleOCR + Gemini Vision · Innov'Dom Challenge 2026 · ENSA Fès")
with col_counter:
    st.metric("📊 Factures traitées", st.session_state.invoice_count, delta="+" + str(st.session_state.invoice_count) if st.session_state.invoice_count > 0 else None)
st.divider()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Paramètres")
    confidence_threshold = st.slider("Seuil de confiance (%)", 50, 100, 80, 5) / 100
    show_preprocessing   = st.toggle("Afficher le prétraitement", value=False)
    show_ocr_boxes       = st.toggle("Afficher les bounding boxes", value=True)

    st.divider()
    st.markdown("#### 🧠 Modèle LLM")
    use_llm = st.toggle(
        "✨ Gemini Vision (recommandé)",
        value=True,
        help="Gemini voit l'image directement — extrait FR + AR avec précision maximale"
    )
    # Clé API intégrée — ne pas afficher dans l'interface
    GEMINI_API_KEY = "AIzaSyAEnCfNkqWDMJJEjBVksC2-hmygMRAX6uc"
    gemini_api_key = GEMINI_API_KEY

    if use_llm:
        st.success("✅ Gemini 2.5 Flash actif")
    else:
        st.info("ℹ️ Mode Regex+NLP uniquement")

    st.divider()
    st.markdown("**Stack technique :**")
    st.markdown("- 🔍 PaddleOCR (FR + AR)")
    st.markdown("- ✨ Gemini 2.5 Flash Vision")
    st.markdown("- 🔤 Regex + NER spaCy")
    st.markdown("- 🖼️ OpenCV (prétraitement)")
    st.markdown("- 🚀 Streamlit (interface)")

    st.divider()
    st.markdown("**Types de documents supportés :**")
    st.markdown("- 🔌 Factures électricité (ONEE, RADEEF)")
    st.markdown("- 💧 Factures eau")
    st.markdown("- 📱 Factures télécom (Orange, IAM)")
    st.markdown("- 🛒 Reçus & factures commerciales")
    st.markdown("- 📄 Bons de commande")


# ════════════════════════════════════════════════════════════
# UPLOAD
# ════════════════════════════════════════════════════════════

uploaded = st.file_uploader(
    "📥 Uploade une facture ou un bon de commande (PDF, JPG, PNG)",
    type=["pdf", "jpg", "jpeg", "png", "tiff", "bmp"],
    help="Supporte les documents en français, arabe ou bilingues"
)
if not uploaded:
    st.info("👆 Uploade une facture pour commencer l'analyse automatique.")

    # Section d'explication quand pas de fichier
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🔍 Étape 1 — OCR")
        st.markdown("PaddleOCR lit le texte français ET arabe avec une précision >95%")
    with c2:
        st.markdown("### 🧠 Étape 2 — IA")
        st.markdown("Gemini Vision comprend la structure et extrait les champs intelligemment")
    with c3:
        st.markdown("### 📊 Étape 3 — Export")
        st.markdown("Résultats exportés en CSV (FR + AR) et JSON pour intégration système")
    st.stop()


# ════════════════════════════════════════════════════════════
# TRAITEMENT
# ════════════════════════════════════════════════════════════

suffix = "." + uploaded.name.split(".")[-1].lower()
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

with st.spinner("📥 Chargement du document..."):
    try:
        images = load_file(tmp_path)
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        st.stop()

st.success(f"✅ Document chargé — {len(images)} page(s)")

# ── OCR ──────────────────────────────────────────────────────
all_blocks = []
progress   = st.progress(0, text="🔍 Reconnaissance optique en cours...")
for i, img in enumerate(images):
    progress.progress((i + 0.5) / len(images), text=f"🔍 Analyse page {i+1}/{len(images)}...")
    all_blocks.extend(run_ocr(img))
progress.progress(1.0, text="✅ OCR terminé !")

full_text    = get_full_text(all_blocks)
full_text_fr = get_full_text(all_blocks, lang_filter="fr")
avg_conf     = get_avg_confidence(all_blocks)

# ── Détection langue du document ─────────────────────────────
def detect_doc_language(text, blocks):
    import re
    t = text.lower()
    en_kw = ["invoice","seller","client","qty","net price","gross worth",
             "tax id","iban","due date","bill to","vat","issued to","items",
             "description","subtotal","amount due","payment terms"]
    fr_kw = ["facture","montant","fournisseur","hors taxe","toutes taxes",
             "echeance","tva","sous-total","numero","emise","destinataire",
             "agence","contrat","tarif"]
    en_score = sum(1 for kw in en_kw if kw in t)
    fr_score = sum(1 for kw in fr_kw if kw in t)
    ar_ratio = sum(1 for b in blocks if b.get("lang") == "ar") / max(len(blocks), 1)
    if ar_ratio > 0.10:
        return "mixed" if (en_score >= 2 or fr_score >= 2) else "ar"
    if en_score > fr_score and en_score >= 2:
        return "en"
    if fr_score > en_score and fr_score >= 2:
        return "fr"
    return "en" if en_score >= fr_score else "fr"

doc_lang = detect_doc_language(full_text, all_blocks)

# ── NLP + LLM ────────────────────────────────────────────────
llm_active  = use_llm and bool(gemini_api_key)
spinner_msg = "✨ Gemini Vision — analyse intelligente..." if llm_active else "🧠 Extraction NLP..."

with st.spinner(spinner_msg):
    extracted = extract_entities(
        ocr_blocks     = all_blocks,
        full_text      = full_text,
        full_text_fr   = full_text_fr,
        img_rgb        = images[0] if llm_active else None,
        use_llm        = llm_active,
        gemini_api_key = gemini_api_key if llm_active else None,
    )

report = validate(extracted, avg_conf)
result = build_result(
    source_file    = uploaded.name,
    ocr_blocks     = all_blocks,
    full_text      = full_text,
    extracted_data = extracted,
    validation     = report,
    nb_pages       = len(images),
)

# Incrémenter le compteur (session uniquement)
if st.session_state.last_processed != uploaded.name:
    st.session_state.invoice_count += 1
    st.session_state.last_processed = uploaded.name


# ════════════════════════════════════════════════════════════
# MÉTRIQUES
# ════════════════════════════════════════════════════════════

st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📄 Pages",           len(images))
c2.metric("🔍 Zones OCR",       len(all_blocks))
fr_count = sum(1 for b in all_blocks if b.get("lang") == "fr")
ar_count = sum(1 for b in all_blocks if b.get("lang") == "ar")
c3.metric("🔤 Blocs OCR",       fr_count)
c5.metric("🎯 Confiance",        f"{avg_conf:.1%}")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "📊 Données extraites",
    "🖼️ Visualisation OCR",
    "💾 Export",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — DONNÉES EXTRAITES
# ════════════════════════════════════════════════════════════

with tab1:
    st.subheader("📊 Champs extraits de la facture")
    d = extracted

    # Badge LLM
    if d.get("_llm_used"):
        conf     = d.get("_llm_confidence")
        conf_str = f" — confiance {conf:.0%}" if conf else ""
        st.success(f"✨ Gemini Vision actif{conf_str} — extraction directe depuis l'image (FR + AR)")
    else:
        st.info("ℹ️ Mode Regex+NLP — active Gemini dans la sidebar pour +précision")

    # ── Identification & Montants ────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🧾 Identification")
        st.text_input("N° Facture",  value=d.get("invoice_number") or "—", disabled=True)
        st.text_input("Date",        value=d.get("date") or d.get("date_emission") or "—", disabled=True)
        if d.get("date_limite"):
            st.text_input("Date limite paiement", value=d.get("date_limite"), disabled=True)
        st.text_input("Fournisseur", value=d.get("supplier") or "—", disabled=True)
        st.text_input("Client",      value=d.get("client") or "—", disabled=True)

    with c2:
        st.markdown("#### 💰 Montants")
        if d.get("amount_ht"):
            st.text_input("Montant HT",  value=str(d.get("amount_ht")), disabled=True)
        if d.get("tva_rate"):
            st.text_input("TVA (%)",     value=str(d.get("tva_rate")), disabled=True)
        if d.get("tva_amount"):
            st.text_input("Montant TVA", value=str(d.get("tva_amount")), disabled=True)
        st.text_input("Montant TTC", value=str(d.get("amount_ttc") or "—"), disabled=True)
        st.text_input("Devise",      value=d.get("currency") or "MAD", disabled=True)

    # ── Contacts ─────────────────────────────────────────────
    contacts = {k: d.get(k) for k in ("email", "phone") if d.get(k)}
    if contacts:
        st.markdown("#### 📬 Contacts")
        cols = st.columns(len(contacts))
        for i, (k, v) in enumerate(contacts.items()):
            label = "Email" if k == "email" else "Téléphone"
            cols[i].text_input(label, value=v, disabled=True)

    # ── Identifiants marocains ────────────────────────────────
    id_ma = {k: d.get(k) for k in
             ("ice", "rc", "if_fiscal", "capital", "agence", "contrat", "tarif", "periode")
             if d.get(k)}
    if id_ma:
        st.markdown("#### 🇲🇦 Identifiants & Détails")
        cols = st.columns(min(len(id_ma), 4))
        labels = {"ice": "ICE", "rc": "RC", "if_fiscal": "IF",
                  "capital": "Capital", "agence": "Agence",
                  "contrat": "N° Contrat", "tarif": "Tarif", "periode": "Période"}
        for idx, (key, val) in enumerate(id_ma.items()):
            cols[idx % 4].text_input(labels.get(key, key.upper()), value=str(val), disabled=True)

    # ── Consommation (factures énergie) ──────────────────────
    if d.get("consommation") or d.get("consommation_kwh"):
        kwh = d.get("consommation") or d.get("consommation_kwh")
        st.markdown("#### ⚡ Consommation")
        st.text_input("Consommation (kWh)", value=str(kwh), disabled=True)

    # ── Tableau consommation ──────────────────────────────────
    ct = d.get("consumption_table", [])
    if ct:
        st.markdown("#### 📈 Historique de consommation")
        st.dataframe(pd.DataFrame(ct), use_container_width=True)

    # ── Lignes produits ───────────────────────────────────────
    items = d.get("line_items_llm") or d.get("line_items", [])
    valid_items = [it for it in items if it.get("description") and len(str(it.get("description", ""))) > 2]
    if valid_items:
        st.markdown("#### 🛒 Lignes de produits / services")
        st.dataframe(pd.DataFrame(valid_items), use_container_width=True)

    # ── NER spaCy ────────────────────────────────────────────
    ner = d.get("ner", {})
    if any(ner.get(k) for k in ("ner_org", "ner_per", "ner_loc")) and not d.get("_llm_used"):
        st.markdown("#### 🤖 Entités détectées (spaCy NER)")
        n1, n2, n3 = st.columns(3)
        if ner.get("ner_org"):
            n1.text_area("Organisations", value="\n".join(ner["ner_org"]), height=80, disabled=True)
        if ner.get("ner_per"):
            n2.text_area("Personnes",     value="\n".join(ner["ner_per"]), height=80, disabled=True)
        if ner.get("ner_loc"):
            n3.text_area("Lieux",         value="\n".join(ner["ner_loc"]), height=80, disabled=True)

    # ── Validation métier ─────────────────────────────────────
    st.markdown("#### ✅ Validation métier")
    if report.get("math_check"):
        st.info(report["math_check"])
    critical_warnings = [w for w in report.get("warnings", [])
                         if "non détecté" in w and "HT" not in w and "numéro" not in w]
    for w in critical_warnings:
        st.warning(f"⚠️ {w}")
    if not critical_warnings:
        st.success("✅ Extraction réussie !")


# ════════════════════════════════════════════════════════════
# TAB 2 — VISUALISATION OCR
# ════════════════════════════════════════════════════════════

with tab2:
    st.subheader("🖼️ Visualisation des zones détectées")
    page_idx = 0
    if len(images) > 1:
        page_idx = st.selectbox("Page", range(len(images)),
                                format_func=lambda x: f"Page {x+1}")

    img = images[page_idx]
    if show_preprocessing:
        ca, cb = st.columns(2)
        ca.image(img, caption="Original", use_column_width=True)
        cb.image(preprocess(img), caption="Prétraitée (OpenCV)", use_column_width=True)

    if show_ocr_boxes:
        fig, ax = plt.subplots(figsize=(12, 16))
        ax.imshow(img)
        page_blocks = [b for b in all_blocks]
        for b in page_blocks:
            bbox  = b["bbox"]
            conf  = b["confidence"]
            lang  = b.get("lang", "fr")
            color = "#FF6B6B" if lang == "ar" else ("limegreen" if conf >= confidence_threshold else "orange")
            xs    = [p[0] for p in bbox] + [bbox[0][0]]
            ys    = [p[1] for p in bbox] + [bbox[0][1]]
            ax.plot(xs, ys, color=color, linewidth=1.5)
            ax.text(bbox[0][0], max(0, bbox[0][1] - 4),
                    f"{b['text'][:18]} {conf:.0%}", color=color, fontsize=5.5,
                    bbox=dict(facecolor="white", alpha=0.65, pad=1, edgecolor="none"))
        ax.set_title(
            f"Page {page_idx+1} — {len(page_blocks)} zones | Confiance moy. {avg_conf:.1%}\n"
            f"[🟢 FR fiable  🟠 FR faible  🔴 AR]",
            fontsize=12, fontweight="bold"
        )
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("##### Détail des zones OCR")
    df_ocr = pd.DataFrame([{
        "Texte":     b["text"],
        "Confiance": f"{b['confidence']:.1%}",
        "Langue":    b.get("lang", "fr").upper(),
        "X":         b["x"],
        "Y":         b["y"],
        "Fiable":    "✅" if b["confidence"] >= confidence_threshold else "⚠️",
    } for b in all_blocks])
    st.dataframe(df_ocr, use_container_width=True, height=300)



# ════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ════════════════════════════════════════════════════════════

with tab3:
    st.subheader("💾 Export des données")
    json_str  = json.dumps(result, ensure_ascii=False, indent=2)
    base_name = uploaded.name.rsplit(".", 1)[0]

    # ── Aperçu des données extraites ─────────────────────────
    st.markdown("#### 📋 Données extraites — Aperçu")
    row_data = _build_row(result)
    preview_rows = [
        {"Champ": FIELDS_FR[k], "Valeur": row_data[k]}
        for k in FIELD_KEYS
        if row_data[k] != "—"
    ]
    if preview_rows:
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("Aucune donnée extraite.")

    st.divider()

    # ── Boutons téléchargement — dynamiques selon langue ────────
    st.markdown("#### ⬇️ Télécharger")

    has_arabic = doc_lang in ("ar", "mixed")
    has_french = doc_lang in ("fr", "mixed")
    is_english = doc_lang == "en"

    nb_cols = 2 + (1 if has_arabic else 0)
    cols = st.columns(nb_cols)

    with cols[0]:
        st.markdown("**📦 JSON complet**")
        st.download_button(
            label="⬇️ Télécharger JSON",
            data=json_str,
            file_name=base_name + "_invoiceai.json",
            mime="application/json",
        )

    with cols[1]:
        csv_label = "🇬🇧 CSV English" if is_english else "🇫🇷 CSV Français"
        btn_label = "⬇️ Download CSV EN" if is_english else "⬇️ Télécharger CSV FR"
        csv_suffix = "_en.csv" if is_english else "_fr.csv"
        st.markdown(f"**{csv_label}**")
        csv_main = to_csv_fr_bytes(result)
        st.download_button(
            label=btn_label,
            data=csv_main,
            file_name=base_name + csv_suffix,
            mime="text/csv",
        )

    if has_arabic:
        with cols[2]:
            st.markdown("**🇲🇦 CSV عربي**")
            csv_ar = to_csv_ar_bytes(result)
            st.download_button(
                label="⬇️ تحميل CSV عربي",
                data=csv_ar,
                file_name=base_name + "_ar.csv",
                mime="text/csv",
            )

    st.divider()

    # ── Distribution confiance OCR — langues présentes uniquement ──
    st.markdown("#### 📈 Distribution de confiance OCR")
    if all_blocks:
        cf_fr = [b["confidence"] for b in all_blocks if b.get("lang") != "ar"]
        cf_ar = [b["confidence"] for b in all_blocks if b.get("lang") == "ar"]
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        has_data = False
        if cf_fr and (has_french or is_english):
            lang_label = f"EN ({len(cf_fr)} blocks)" if is_english else f"FR ({len(cf_fr)} blocs)"
            ax2.hist(cf_fr, bins=20, color="#4CAF50", alpha=0.75, edgecolor="white", label=lang_label)
            has_data = True
        if cf_ar and has_arabic:
            ax2.hist(cf_ar, bins=20, color="#FF6B6B", alpha=0.75, edgecolor="white", label=f"AR ({len(cf_ar)} blocs)")
            has_data = True
        if has_data:
            seuil_label = f"Threshold {confidence_threshold:.0%}" if is_english else f"Seuil {confidence_threshold:.0%}"
            ax2.axvline(confidence_threshold, color="red", linestyle="--", label=seuil_label)
            ax2.set_xlabel("OCR Confidence Score" if is_english else "Score de confiance OCR")
            ax2.set_ylabel("Zones")
            if has_french and has_arabic:
                ax2.set_title("Distribution des scores — Français vs Arabe")
            elif is_english:
                ax2.set_title("OCR Confidence Score Distribution")
            else:
                ax2.set_title("Distribution des scores OCR")
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
        plt.close()

try:
    os.unlink(tmp_path)
except Exception:
    pass