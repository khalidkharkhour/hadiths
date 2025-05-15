import os
import pandas as pd
from pyarabic.araby import tokenize, normalize_hamza

def preprocess_text(text):
    """Normaliser et tokeniser le texte arabe."""
    if not isinstance(text, str):
        return ""
    normalized_text = normalize_hamza(text)
    tokens = tokenize(normalized_text)
    return ' '.join(tokens)

def load_hadith_data(filepath='All/all_hadiths_clean.csv'):
    """
    Charger les données des hadiths à partir du fichier CSV.
    Retourne un DataFrame contenant les colonnes pertinentes:
    - text (text_ar prétraité)
    - source
    - hadith_no
    - hadith_id
    - chapter
    """
    try:
        # Charger le fichier CSV
        df = pd.read_csv(filepath)
        print(f"✅ Fichier chargé avec {len(df)} hadiths")
        
        # Vérifier les colonnes requises
        required_columns = ['text_ar', 'source', 'hadith_no', 'hadith_id', 'chapter']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️ Colonnes manquantes: {', '.join(missing_columns)}")
            # On continue avec les colonnes disponibles
        
        # Sélectionner et renommer les colonnes nécessaires
        result_df = pd.DataFrame()
        result_df['text'] = df['text_ar'].apply(preprocess_text)
        
        # Ajouter les colonnes disponibles pour les métadonnées
        for col in ['source', 'hadith_no', 'hadith_id', 'chapter']:
            if col in df.columns:
                result_df[col] = df[col]
        
        # Supprimer les lignes où le texte est vide
        result_df = result_df.dropna(subset=['text'])
        
        print(f"✅ Données préparées: {len(result_df)} entrées valides")
        return result_df
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {str(e)}")
        # Retourner un DataFrame vide avec les bonnes colonnes
        return pd.DataFrame(columns=['text', 'source', 'hadith_no', 'hadith_id', 'chapter'])

def prepare_data(df, source_name="all_hadiths_clean"):
    """Préparer les textes et étiquettes pour l'entraînement."""
    texts = df['text'].dropna().tolist()  # Le texte est déjà prétraité
    labels = [source_name] * len(texts)
    return texts, labels
