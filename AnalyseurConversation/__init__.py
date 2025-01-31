import faiss
import logging
import azure.functions as func
import re
import json
from typing import Optional, Dict
import os


app =func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
# Configuration du logger optimisée pour Azure Functions
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AnalyseurConversation:
    def __init__(self, model_name="text-embedding-ada-002", openai_api_key=os.environ["OPENAI_API_KEY"]):
        openai.api_key = openai_api_key  # Set OpenAI API key

        exemples = {
            "oui": ["Oui, bien sûr.", "yes", "je suis ok", "D'accord, je suis partant.", "Oui, je le fais volontiers.", "C'est une excellente idée.", "Oui, sans hésiter.", "Oui, je confirme.", "Bien sûr, je suis d'accord.", "Oui, c'est tout à fait correct.", "Je suis pour.", "Oui, je suis avec vous.", "Sans problème, c'est oui.", "Oui, je suis partiellement d'accord.", "Oui, pourquoi pas.", "D'accord, je le ferai.", "Oui, c'est une bonne solution.", "Bien sûr, je le fais sans hésiter.", "Je consens", "je veux", "Absolument", "Avec plaisir", "Oui, je suis d'accord", "Ça me va", "Bien entendu", "Pas de souci", "Ok, c'est bon", "C'est bon pour moi", "Je suis favorable", "C'est d'accord", "Tout à fait", "J'accepte"],
            "non": ["Non, merci.", "no", "Je ne suis pas d'accord.", "Non, ce n'est pas pour moi.", "Je préfère ne pas.", "Non, ce n'est pas possible.", "Non, je ne veux pas.", "Je ne suis pas intéressé.", "Non, je ne le ferai pas.", "Ce n'est pas ce que je veux.", "Non, je ne crois pas.", "Non, je ne pense pas.", "Non, c'est non.", "Je m'abstiens.", "Non, pas question.", "Ce n'est pas acceptable pour moi.", "Non, je refuse catégoriquement.", "Je ne consens pas", "je ne veux pas", "Pas du tout", "C'est non", "Je ne peux pas", "Ce n'est pas possible", "Je ne suis pas d'accord avec ça", "C'est hors de question", "Je refuse", "Non, ça ne m'intéresse pas", "Je décline"],
            "indéterminé": ["Je ne suis pas sûr.", "Je ne sais pas.", "Peut-être, je ne sais pas vraiment.", "Je ne suis pas convaincu.", "C'est compliqué.", "Je doute.", "Ça m'embête.", "Je ne suis pas certain.", "C'est flou.", "Je crois que non.", "Je ne suis pas certain de ma réponse.", "Je ne sais pas quoi répondre.", "Je suis hésitant.", "C'est ambigu.", "Je ne suis pas clair sur ma réponse.", "Je n'ai pas d'avis.", "Je ne sais pas trop.", "Je suis indécis.", "C'est un peu flou pour moi.", "Je suis partagé.", "J'ai des doutes.", "Je n'ai pas de réponse précise.", "Je ne peux pas me prononcer.", "C'est incertain.", "Je suis perplexe.", "Je n'ai pas de certitude.", "C'est difficile à dire.", "Je n'ai pas d'opinion claire."]
        }

        self.model_name = model_name
        self.init_faiss_index(exemples)
        self.pattern_quitter = r"""\b(quitt(?:e|er|é|ant)?|part(?:i|ir|ait|ie|is|ons)?|arrêt(?:er|e|ons)?|fini(?:r|e|s)?|stop|au revoir|termin(?:er|é|e)?|m'en vais|ferm(?:er|é|ée|ons)?|bientôt|clôtur(?:er|e|ons)?|fin(?:ir|ie)?|c'est tout|ça y est|je file|je m'en vais|je dois y aller|je me tire|je me casse|je bounce|j'y vais|bon, j'y vais|go)\b"""

    def get_openai_embedding(self, text):
        response = openai.Embedding.create(
            input=text,
            model=self.model_name
        )
        return np.array(response["data"][0]["embedding"])

    def init_faiss_index(self, exemples):
        phrases = []
        categories = []

        for category, examples in exemples.items():
            for phrase in examples:
                phrases.append(phrase.lower())
                categories.append(category)

        embeddings = np.array([self.get_openai_embedding(phrase) for phrase in phrases])
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.categories = categories

    def quitter_conversation(self, question):
        return bool(re.search(self.pattern_quitter, question.lower()))

    def positive_negative_reponse(self, reponse):
        question_embedding = self.get_openai_embedding(reponse.lower()).reshape(1, -1)
        distances, indices = self.index.search(question_embedding, 1)
        category = self.categories[indices[0][0]]

        if category == "oui":
            return 1
        elif category == "non":
            return 0
        else:
            return 2

    def recueil_consentement(self, reponse):
        classification = self.positive_negative_reponse(reponse)
        return classification
    
analyzer=AnalyseurConversation()
handlers: Dict[str, callable] = {
    "recueil_consentement": analyzer.recueil_consentement,
    "positive_negative_reponse": analyzer.positive_negative_reponse,
    "quitter_conversation": analyzer.quitter_conversation
}

@app.route(route="analyseur_conversation")
def analyseur_conversation(req: func.HttpRequest) -> func.HttpResponse:
    """Gère la requête en fonction de l'action demandée"""
    logger.info("Début du traitement de la requête HTTP")
    try:
        req_body = req.get_json()
        logger.info("Corps de la requête JSON récupéré avec succès")
    except ValueError:
        logger.error("Erreur lors de la récupération du corps de la requête. Le JSON est invalide.")
        return func.HttpResponse("Invalid JSON", status_code=400)

    action = req_body.get("action", "").strip()
    texte = req_body.get("texte", "").strip()

    # Vérification des paramètres
    if not action or not texte:
        logger.warning("Paramètres manquants ou invalides : 'action' ou 'texte' manquants.")
        return func.HttpResponse("Paramètres 'action' et 'texte' requis", status_code=400)
    logger.info(f"Action reçue : {action}")
    logger.info(f"Texte reçu : {texte[:50]}...")  # Affichage limité pour ne pas exposer de données sensibles dans les logs
    # Vérification si l'action est valide
    handler = handlers.get(action)
    if not handler:
        logger.error(f"Action inconnue : {action}")
        return func.HttpResponse("Action inconnue", status_code=400)
    logger.info(f"Exécution de l'action : {action}")
    # Exécuter la fonction correspondante et retourner le résultat
    try:
        result = handler(texte)
        logger.info(f"Résultat de l'action {action} : {result}")
        return func.HttpResponse(json.dumps({action: result}), mimetype="application/json", status_code=200)
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction pour l'action {action}: {str(e)}")
        return func.HttpResponse(f"Erreur lors de l'extraction : {str(e)}", status_code=500)
