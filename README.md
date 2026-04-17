# 🧾 Invoice OCR Analyzer
### Innov'Dom Challenge 2026 — ENSA Fès

Pipeline complet : **EasyOCR → NLP → Streamlit**

---

## ⚡ Installation (Mac — 5 minutes)

```bash
# 1. Cloner / copier le dossier invoice_app

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Installer poppler (pour les PDF)
brew install poppler

# 5. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur http://localhost:8501

---

## 📁 Structure du projet

```
invoice_app/
├── app.py              ← Application Streamlit (interface)
├── ocr_engine.py       ← Module OCR (EasyOCR + prétraitement OpenCV)
├── nlp_extractor.py    ← Module NLP (Regex + heuristiques)
├── exporter.py         ← Export JSON / CSV
├── pipeline.py         ← Pipeline complet (sans UI)
└── requirements.txt    ← Dépendances Python
```

---

## 🔧 Utilisation en script (sans UI)

```python
from pipeline import process_invoice

result = process_invoice("ma_facture.pdf")
print(result["extracted_data"])
```

---

## 📊 Champs extraits

| Champ | Description |
|-------|-------------|
| `invoice_number` | Numéro de facture |
| `date` | Date de la facture (ISO) |
| `supplier` | Nom du fournisseur |
| `client` | Nom du client |
| `amount_ht` | Montant hors taxes |
| `tva_rate` | Taux TVA (%) |
| `tva_amount` | Montant TVA |
| `amount_ttc` | Montant TTC |
| `currency` | Devise (MAD, EUR...) |
| `email` | Email détecté |
| `phone` | Téléphone détecté |
| `siret` | SIRET / ICE / RC |
| `rib` | RIB / IBAN |
| `line_items` | Lignes de produits |

---

## 🏆 Stack technique

- **EasyOCR** — Reconnaissance optique (Python 3.12 ✅, GPU ✅)
- **OpenCV** — Prétraitement : débruitage, CLAHE, deskew
- **Regex + NLP** — Extraction des entités
- **Streamlit** — Interface web
- **pandas** — Export CSV
