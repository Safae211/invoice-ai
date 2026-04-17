from Llm_extractor import run_gemini
from PIL import Image
import numpy as np

img = np.array(Image.open('facture.png').convert('RGB'))
result = run_gemini(img)
print('ERROR:', result.get('_llm_error'))
print('AVAILABLE:', result.get('_llm_available'))
print('MONTANT:', result.get('amount_ttc'))
print('DATE:', result.get('date'))
print('FOURNISSEUR:', result.get('supplier'))
