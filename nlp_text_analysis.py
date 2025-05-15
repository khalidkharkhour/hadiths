import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Dict, List, Optional, Tuple
from utils import preprocess_text, load_hadith_data

class TextAnalyzer:
    def __init__(self):
        """Initialize NLP models and pipelines."""
        # Load spaCy Arabic model for NER
        try:
            self.nlp = spacy.load("xx_ent_wiki_sm")
        except OSError:
            print("⚠️ spaCy model not found. Please install with: python -m spacy download xx_ent_wiki_sm")
            self.nlp = None
        
        # Load transformers pipelines
        try:
            print("Chargement du modèle de résumé...")
            self.summarizer = pipeline(
                "summarization",
                model="csebuetnlp/mT5_multilingual_XLSum",
                tokenizer="csebuetnlp/mT5_multilingual_XLSum"
            )
            self.use_summarizer = True
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du modèle de résumé: {e}")
            print("⚠️ Utilisation d'une méthode de résumé alternative")
            self.summarizer = None
            self.use_summarizer = False
            
        try:
            print("Chargement du modèle d'analyse de sentiment...")
            self.sentiment_analyzer = pipeline(
                "text-classification",
                model="CAMeL-Lab/bert-base-arabic-camelbert-ca",
                tokenizer="CAMeL-Lab/bert-base-arabic-camelbert-ca"
            )
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du modèle de sentiment: {e}")
            self.sentiment_analyzer = None
        
        # Load hadith data for comparison
        try:
            self.hadith_df = load_hadith_data()
            # Conserver les métadonnées avec les textes pour les retrouver plus tard
            self.hadith_texts = self.hadith_df['text'].tolist()
            self.hadith_metadata = self.hadith_df.to_dict('records')
        except Exception as e:
            print(f"❌ Failed to load hadith data: {e}")
            self.hadith_df = pd.DataFrame()
            self.hadith_texts = []
            self.hadith_metadata = []
        
        # Load embedding model for similarity
        try:
            print("Chargement du modèle d'embedding...")
            self.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.embedding_model)
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du modèle d'embedding: {e}")
            self.tokenizer = None
            self.model = None

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.logits.detach().numpy().mean(axis=0)

    def extract_key_sentences(self, text, num_sentences=3):
        """Extraire les phrases les plus importantes du texte comme méthode alternative de résumé."""
        # Diviser le texte en phrases
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return text
            
        if len(sentences) <= num_sentences:
            return '. '.join(sentences) + '.'
            
        # Compter les mots dans chaque phrase
        word_counts = [len(s.split()) for s in sentences]
        
        # Éliminer les phrases trop courtes (probablement pas informatives)
        valid_sentences = [(i, s) for i, s in enumerate(sentences) if len(s.split()) >= 5]
        
        if not valid_sentences:
            # Si toutes les phrases sont trop courtes, prendre les premières
            return '. '.join(sentences[:num_sentences]) + '.'
            
        # Extraire les mots-clés du texte complet
        keywords = self._extract_keywords(text)
        
        # Calculer les scores des phrases en fonction des mots-clés présents
        sentence_scores = []
        for idx, sentence in valid_sentences:
            score = sum(1 for keyword in keywords if keyword in sentence.split())
            # Bonus pour les premières phrases (souvent plus importantes)
            if idx < 2:
                score += 2
            sentence_scores.append((idx, sentence, score))
        
        # Trier par score décroissant
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Prendre les meilleures phrases et les réorganiser selon leur position originale
        best_sentences = [(idx, sent) for idx, sent, _ in sentence_scores[:num_sentences]]
        best_sentences.sort(key=lambda x: x[0])  # Trier par position originale
        
        # Reconstruire le résumé
        summary = '. '.join([sent for _, sent in best_sentences]) + '.'
        return summary

    def summarize_text(self, text: str, max_length: int = 100, min_length: int = 10) -> str:
        """Summarize the input text."""
        try:
            # Vérifier si le texte est assez long pour être résumé
            processed_text = preprocess_text(text)
            
            # Si le texte est trop court, retourner le texte original au lieu de le résumer
            if len(processed_text.split()) < 30:  # Si moins de 30 mots
                return processed_text
            
            # Si le modèle de résumé est disponible et activé, l'utiliser
            if self.use_summarizer and self.summarizer:
                # Ajuster les paramètres pour les textes arabes
                summary = self.summarizer(
                    processed_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=4,  # Améliore la qualité du résumé
                    early_stopping=True
                )[0]['summary_text']
                
                # Vérifier que le résumé n'est pas le message d'erreur connu
                if "هذه ليست صورة لرجل" in summary or len(summary.split()) < 5:
                    return self.extract_key_sentences(processed_text)
                    
                return summary
            else:
                # Utiliser la méthode alternative basée sur l'extraction de phrases clés
                return self.extract_key_sentences(processed_text)
                
        except Exception as e:
            # En cas d'erreur, extraire les phrases importantes comme résumé
            try:
                return self.extract_key_sentences(processed_text)
            except:
                # Si tout échoue, retourner les premiers mots du texte
                short_text = processed_text[:200] + "..." if len(processed_text) > 200 else processed_text
                return short_text

    def identify_context(self, text: str) -> Dict[str, str]:
        """Identify historical, cultural, or religious context."""
        sentiment = self.sentiment_analyzer(text)[0]
        keywords = self._extract_keywords(text)
        
        context = {
            "المشاعر": sentiment['label'],
            "الثقة": f"{sentiment['score']:.2f}",
            "الكلمات_المفتاحية": ", ".join(keywords),
            "السياق_المستنتج": self._infer_context_from_keywords(keywords)
        }
        return context

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text (simplified)."""
        tokens = preprocess_text(text).split()
        from collections import Counter
        word_freq = Counter(tokens)
        return [word for word, count in word_freq.most_common(5)]

    def _infer_context_from_keywords(self, keywords: List[str]) -> str:
        """Infer context based on keywords."""
        religious_terms = ['الله', 'رسول', 'حديث', 'قرآن', 'صلاة']
        historical_terms = ['سنة', 'هجرية', 'خليفة', 'معركة']
        
        if any(term in keywords for term in religious_terms):
            return "ديني (إسلامي)"
        elif any(term in keywords for term in historical_terms):
            return "تاريخي (العصر الإسلامي)"
        else:
            return "عام"

    def detect_entities(self, text: str) -> Dict[str, List[str]]:
        """Detect named entities using spaCy."""
        if not self.nlp:
            return {"خطأ": "لم يتم تحميل نموذج spaCy"}
        
        doc = self.nlp(text)
        entities = {"الأشخاص": [], "الأماكن": [], "التواريخ": [], "المنظمات": []}
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["الأشخاص"].append(ent.text)
            elif ent.label_ == "GPE":
                entities["الأماكن"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["التواريخ"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["المنظمات"].append(ent.text)
        
        return entities

    def evaluate_reliability(self, text: str) -> Dict[str, str]:
        """Evaluate text reliability by checking contradictions and sentiment."""
        sentences = text.split('.')
        contradictions = []
        sentiment_scores = []
        
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            sentiment = self.sentiment_analyzer(sent)[0]
            sentiment_scores.append((sent, sentiment['label'], sentiment['score']))
        
        # Check for conflicting sentiments
        for i in range(len(sentiment_scores) - 1):
            sent1, label1, score1 = sentiment_scores[i]
            sent2, label2, score2 = sentiment_scores[i + 1]
            if label1 != label2 and score1 > 0.7 and score2 > 0.7:
                contradictions.append(f"تناقض بين: '{sent1}' ({label1}) و '{sent2}' ({label2})")
        
        reliability = {
            "التناقضات": contradictions if contradictions else ["لم يتم اكتشاف تناقضات"],
            "اتساق_التقييم": "متسق" if not contradictions else "غير متسق",
            "ملاحظة_الموثوقية": "التقييم العالي لاتساق النص يشير إلى موثوقيته، لكن وجود تناقضات يقلل من مصداقيته."
        }
        return reliability

    def compare_with_hadith(self, text: str, threshold: float = 0.8) -> Dict[str, any]:
        """
        Compare text with hadith database for similarity.
        Returns similarity metrics and metadata of the most similar hadiths.
        """
        if not self.hadith_texts:
            return {"خطأ": "لا توجد بيانات أحاديث متاحة للمقارنة"}
        
        text_embedding = self.get_text_embedding(text)
        similarities = []
        
        # Limiter à 100 hadiths pour éviter les calculs trop longs
        max_hadiths = min(100, len(self.hadith_texts))
        
        for i in range(max_hadiths):
            hadith_embedding = self.get_text_embedding(self.hadith_texts[i])
            sim = cosine_similarity([text_embedding], [hadith_embedding])[0][0]
            similarities.append((sim, i))
        
        # Trier par similarité décroissante
        similarities.sort(reverse=True)
        
        # Récupérer les 3 hadiths les plus similaires
        top_matches = similarities[:3]
        similar_hadiths = []
        
        for sim, idx in top_matches:
            if idx < len(self.hadith_metadata):
                metadata = self.hadith_metadata[idx]
                hadith_info = {
                    "similarité": f"{sim:.2f}",
                    "texte": self.hadith_texts[idx][:100] + "..." if len(self.hadith_texts[idx]) > 100 else self.hadith_texts[idx]
                }
                
                # Ajouter les métadonnées disponibles
                for key in ['source', 'hadith_no', 'hadith_id', 'chapter']:
                    if key in metadata:
                        hadith_info[key] = metadata[key]
                
                similar_hadiths.append(hadith_info)
        
        max_similarity = top_matches[0][0] if top_matches else 0
        
        result = {
            "أقصى_التشابه": f"{max_similarity:.2f}",
            "مشابه": "مشابه للأحاديث المعروفة" if max_similarity > threshold else "غير مشابه للأحاديث المعروفة",
            "ملاحظة_الأصالة": "التشابه العالي يشير إلى إمكانية الأصالة، لكن يتطلب مزيدًا من التحقق.",
            "الأحاديث_المشابهة": similar_hadiths
        }
        
        return result
    def scientific_analysis(self, text: str) -> Dict[str, str]:
        """Fournir une analyse philologique, codicologique et historique de base."""
        keywords = self._extract_keywords(text)
        
        analysis = {
            "paléographie": "❌ Non applicable sans image manuscrite",
            "diplomatique": "✅ Formules classiques détectées" if any(k in keywords for k in ['قال', 'سمعت', 'حدثنا']) else "❌ Aucune formule diplomatique claire",
            "codicologie": "❌ Non applicable sans métadonnées physiques",
            "philologie": f"📘 Mots fréquents : {', '.join(keywords)}",
            "critique_textuelle": "🔍 Analyse limitée (pas de variantes disponibles)",
            "chronologie": self._infer_chronology(keywords),
            "historiographie": self._infer_historiography(keywords)
        }
        return analysis

    def _infer_chronology(self, keywords: List[str]) -> str:
        if any(k in keywords for k in ['هجرية', 'سنة', 'قرن']):
            return "📅 Éléments de datation repérés"
        return "❌ Aucune référence temporelle"

    def _infer_historiography(self, keywords: List[str]) -> str:
        if any(k in keywords for k in ['رواة', 'إسناد', 'البخاري', 'مسلم', 'الطبري']):
            return "🧾 Mention d’autorités ou chaînes de transmission"
        return "❌ Pas de trace historiographique explicite"

def analyze_text(text: str) -> Dict[str, any]:
    """Main function to analyze a given text."""
    analyzer = TextAnalyzer()
    
    results = {
        "الملخص": analyzer.summarize_text(text),
        "السياق": analyzer.identify_context(text),
        "الكيانات": analyzer.detect_entities(text),
        "الموثوقية": analyzer.evaluate_reliability(text),
         "مقارنة_الأحاديث": analyzer.compare_with_hadith(text),
                                                                                       "analyse_scientifique": analyzer.scientific_analysis(text)
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_text = "قال رسول الله صلى الله عليه وسلم: من كذب علي متعمدا فليتبوأ مقعده من النار. رواه البخاري."
    results = analyze_text(sample_text)
    
    print("=== نتائج تحليل النص ===")
    print(f"الملخص: {results['الملخص']}")
    print(f"السياق: {results['السياق']}")
    print(f"الكيانات: {results['الكيانات']}")
    print(f"الموثوقية: {results['الموثوقية']}")
    print(f"مقارنة الأحاديث: {results['مقارنة_الأحاديث']}")
