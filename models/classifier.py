from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class HadithClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = RandomForestClassifier()

    def train(self, texts, labels):
        """Entraîner le modèle de classification des hadiths."""
        X = self.vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        print(f"Modèle entraîné avec {len(texts)} échantillons.")

    def predict(self, text):
        """Prédire la catégorie d'un hadith donné."""
        text_vectorized = self.vectorizer.transform([text])
        return self.model.predict(text_vectorized)[0]

