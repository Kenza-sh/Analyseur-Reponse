import logging
import azure.functions as func
import re
import json
from typing import Optional, Dict
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AnalyseurConversation:
    def __init__(self):
        # Exemples de réponses classées
        self.exemples = {
            "oui": ["Oui, bien sûr.", "yes", "je suis ok", "D'accord, je suis partant.", "Oui, je le fais volontiers.",
                    "C'est une excellente idée.", "Oui, sans hésiter.", "Oui, je confirme.", "Bien sûr, je suis d'accord.",
                    "Oui, c'est tout à fait correct.", "Je suis pour.", "Oui, je suis avec vous.", "Sans problème, c'est oui.",
                    "Oui, pourquoi pas.", "D'accord, je le ferai.", "Oui, c'est une bonne solution.", "Absolument",
                    "Avec plaisir", "Ça me va", "Bien entendu", "Pas de souci", "Ok, c'est bon", "C'est bon pour moi"],
            
            "non": ["Non, merci.", "no", "Je ne suis pas d'accord.", "Non, ce n'est pas pour moi.", "Je préfère ne pas.",
                    "Non, ce n'est pas possible.", "Non, je ne veux pas.", "Je ne suis pas intéressé.", "Non, je ne le ferai pas.",
                    "Ce n'est pas ce que je veux.", "Non, je ne crois pas.", "Non, je ne pense pas.", "Non, c'est non.",
                    "Je m'abstiens.", "Non, pas question.", "Ce n'est pas acceptable pour moi.", "Non, je refuse catégoriquement.",
                    "Je ne consens pas", "je ne veux pas", "Pas du tout", "C'est non", "Je ne peux pas", "Je refuse"],
            
            "indéterminé": ["Je ne suis pas sûr.", "Je ne sais pas.", "Peut-être, je ne sais pas vraiment.",
                            "Je ne suis pas convaincu.", "C'est compliqué.", "Je doute.", "Ça m'embête.", "Je ne suis pas certain.",
                            "C'est flou.", "Je crois que non.", "Je ne suis pas certain de ma réponse.", "Je ne sais pas quoi répondre.",
                            "Je suis hésitant.", "C'est ambigu.", "Je ne suis pas clair sur ma réponse.", "Je n'ai pas d'avis.",
                            "Je ne sais pas trop.", "Je suis indécis.", "C'est un peu flou pour moi.", "Je suis partagé.",
                            "J'ai des doutes.", "Je n'ai pas de réponse précise.", "Je ne peux pas me prononcer.", "C'est incertain."]
        }

        # Regex pour détecter les intentions de quitter la conversation
        self.pattern_quitter = r"""\b(quitt(?:e|er|é|ant)?|part(?:i|ir|ait|ie|is|ons)?|arrêt(?:er|e|ons)?|fini(?:r|e|s)?|stop|au revoir|termin(?:er|é|e)?|m'en vais|ferm(?:er|é|ée|ons)?|bientôt|clôtur(?:er|e|ons)?|fin(?:ir|ie)?|c'est tout|ça y est|je file|je m'en vais|je dois y aller|je me tire|je me casse|je bounce|j'y vais|bon, j'y vais|go)\b"""

        # Initialisation du modèle TF-IDF
        self.init_tfidf_model()

    def init_tfidf_model(self):
        """Initialise le vecteur TF-IDF avec les exemples existants."""
        self.vectorizer = TfidfVectorizer()

        self.phrases = []
        self.categories = []

        for category, examples in self.exemples.items():
            self.phrases.extend(examples)
            self.categories.extend([category] * len(examples))

        # Création du vocabulaire et calcul des vecteurs TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.phrases)

    def quitter_conversation(self, question):
        """Vérifie si la question indique une intention de quitter la conversation."""
        return bool(re.search(self.pattern_quitter, question.lower()))

    def positive_negative_reponse(self, response):
        """Classifie une réponse en 'oui', 'non' ou 'indéterminé' en utilisant la similarité cosinus."""
        if not response.strip():
            return "indéterminé"

        response_vector = self.vectorizer.transform([response])
        similarities = cosine_similarity(response_vector, self.tfidf_matrix)

        # Trouver l'index de la phrase la plus similaire
        best_match_index = np.argmax(similarities)

        return self.categories[best_match_index]

    def recueil_consentement(self, reponse):
        """Retourne 1 pour 'oui', 0 pour 'non' et 2 pour 'indéterminé'."""
        classification = self.positive_negative_reponse(reponse)
        return {"oui": 1, "non": 0, "indéterminé": 2}[classification]



    
analyzer=AnalyseurConversation()
handlers: Dict[str, callable] = {
    "recueil_consentement": analyzer.recueil_consentement,
    "positive_negative_reponse": analyzer.positive_negative_reponse,
    "quitter_conversation": analyzer.quitter_conversation
}

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Gère la requête en fonction de l'action demandée"""
    logger.info("Début du traitement de la requête HTTP")

    try:
        req_body = req.get_json()
        logger.info("Corps de la requête JSON récupéré avec succès")
    except ValueError as e:
        logger.error(f"Erreur lors du traitement de la requête : {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON format"}),
            mimetype="application/json",
            status_code=400
        )

    # Extraction et validation des paramètres
    action = req_body.get("action", "").strip()
    texte = req_body.get("texte", "").strip()

    if not action or not texte:
        return func.HttpResponse(
            json.dumps({"error": "Paramètres 'action' et 'texte' requis"}),
            mimetype="application/json",
            status_code=400
        )

    logger.info(f"Action reçue : {action}")
    logger.info(f"Texte reçu : {texte[:50]}...")  # Limite pour éviter d'exposer des données sensibles

    # Vérification si l'action est valide
    handler = handlers.get(action)
    if not handler:
        logger.error(f"Action inconnue : {action}")
        return func.HttpResponse(
            json.dumps({"error": "Action inconnue"}),
            mimetype="application/json",
            status_code=400
        )

    logger.info(f"Exécution de l'action : {action}")

    # Exécuter la fonction correspondante et retourner le résultat
    try:
        result = handler(texte)
        return func.HttpResponse(
            json.dumps({"response": result}),
            mimetype="application/json",
            status_code=200
        )
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution de l'action '{action}': {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Erreur lors de l'exécution: {str(e)}"}),
            mimetype="application/json",
            status_code=500
        )

           
