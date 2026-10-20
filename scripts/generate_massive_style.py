"""Generate the Semmai MASSIVE-format multilingual intent + slot dataset.

6 locales (en-US, ta-IN, hi-IN, fr-FR, es-ES, de-DE) x 12 intents, written
as natural command templates with slot fillers in each language.

Train and test use DIFFERENT templates (templates 0-2 train, 3-4 test), and
~30% of test utterances use slot values never seen in training, so the
evaluation measures generalisation rather than memorisation.

Records follow the Amazon MASSIVE jsonl schema:
    {"id", "locale", "partition", "scenario", "intent", "utt", "annot_utt"}

Usage: python scripts/generate_massive_style.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "massive_style"
LOCALES = ["en-US", "ta-IN", "hi-IN", "fr-FR", "es-ES", "de-DE"]

# ------------------------------------------------------------------ fillers
# (seen values, unseen values used only in test)
F = {
 "time": {
  "en-US": (["7 am", "6:30 am", "5 pm", "9 pm", "noon"], ["8 am", "10 pm"]),
  "ta-IN": (["காலை 7 மணி", "காலை 6:30 மணி", "மாலை 5 மணி", "இரவு 9 மணி", "மதியம் 12 மணி"], ["காலை 8 மணி", "இரவு 10 மணி"]),
  "hi-IN": (["सुबह 7 बजे", "सुबह 6:30 बजे", "शाम 5 बजे", "रात 9 बजे", "दोपहर 12 बजे"], ["सुबह 8 बजे", "रात 10 बजे"]),
  "fr-FR": (["7 heures", "6h30", "17 heures", "21 heures", "midi"], ["8 heures", "22 heures"]),
  "es-ES": (["las 7 de la mañana", "las 6:30", "las 5 de la tarde", "las 9 de la noche"], ["las 8 de la mañana", "las 10 de la noche"]),
  "de-DE": (["7 Uhr", "6:30 Uhr", "17 Uhr", "21 Uhr"], ["8 Uhr", "22 Uhr"])},
 "date": {
  "en-US": (["tomorrow", "today", "monday", "friday", "this weekend"], ["sunday"]),
  "ta-IN": (["நாளை", "இன்று", "திங்கள்கிழமை", "வெள்ளிக்கிழமை"], ["ஞாயிற்றுக்கிழமை"]),
  "hi-IN": (["कल", "आज", "सोमवार", "शुक्रवार"], ["रविवार"]),
  "fr-FR": (["demain", "aujourd'hui", "lundi", "vendredi"], ["dimanche"]),
  "es-ES": (["hoy", "el lunes", "el viernes", "pasado mañana"], ["el domingo"]),
  "de-DE": (["morgen", "heute", "Montag", "Freitag"], ["Sonntag"])},
 "place_name": {
  "en-US": (["chennai", "mumbai", "delhi", "london", "paris"], ["bengaluru"]),
  "ta-IN": (["சென்னை", "மும்பை", "டெல்லி", "லண்டன்"], ["பெங்களூரு"]),
  "hi-IN": (["चेन्नई", "मुंबई", "दिल्ली", "लंदन"], ["बेंगलुरु"]),
  "fr-FR": (["Chennai", "Mumbai", "Delhi", "Londres", "Paris"], ["Kolkata"]),
  "es-ES": (["Chennai", "Mumbai", "Delhi", "Londres", "Madrid"], ["Kolkata"]),
  "de-DE": (["Chennai", "Mumbai", "Delhi", "London", "Berlin"], ["Kolkata"])},
 "artist_name": {
  "en-US": (["a r rahman", "ilaiyaraaja", "arijit singh", "shakira"], ["adele"]),
  "ta-IN": (["ஏ. ஆர். ரகுமான்", "இளையராஜா", "அனிருத்"], ["சித் ஸ்ரீராம்"]),
  "hi-IN": (["ए. आर. रहमान", "अरिजीत सिंह", "श्रेया घोषाल"], ["लता मंगेशकर"]),
  "fr-FR": (["Stromae", "Édith Piaf", "Daft Punk"], ["Indila"]),
  "es-ES": (["Shakira", "Rosalía", "Enrique Iglesias"], ["Juanes"]),
  "de-DE": (["Rammstein", "Helene Fischer", "Nena"], ["Kraftwerk"])},
 "event_name": {
  "en-US": (["team meeting", "dentist appointment", "project review"], ["birthday party"]),
  "ta-IN": (["குழு கூட்டம்", "பல் மருத்துவர் சந்திப்பு"], ["பிறந்தநாள் விழா"]),
  "hi-IN": (["टीम मीटिंग", "डॉक्टर की अपॉइंटमेंट"], ["जन्मदिन पार्टी"]),
  "fr-FR": (["réunion d'équipe", "visite chez le dentiste"], ["fête d'anniversaire"]),
  "es-ES": (["reunión de equipo", "cita con el dentista"], ["fiesta de cumpleaños"]),
  "de-DE": (["Teammeeting", "Zahnarzttermin"], ["Geburtstagsfeier"])},
 "person": {
  "en-US": (["ravi", "priya", "anand", "john"], ["meena"]),
  "ta-IN": (["ரவி", "பிரியா", "ஆனந்த்"], ["மீனா"]),
  "hi-IN": (["रवि", "प्रिया", "आनंद"], ["मीना"]),
  "fr-FR": (["Marie", "Pierre", "Ravi"], ["Julien"]),
  "es-ES": (["María", "Carlos", "Ravi"], ["Lucía"]),
  "de-DE": (["Anna", "Thomas", "Ravi"], ["Lukas"])},
 "transport_type": {
  "en-US": (["train", "bus", "flight"], []), "ta-IN": (["ரயில்", "பேருந்து", "விமான"], []),
  "hi-IN": (["ट्रेन", "बस", "फ्लाइट"], []), "fr-FR": (["train", "bus", "avion"], []),
  "es-ES": (["tren", "autobús", "avión"], []), "de-DE": (["Zug", "Bus", "Flug"], [])},
 "food_type": {
  "en-US": (["pizza", "biryani", "dosa", "noodles"], ["burger"]),
  "ta-IN": (["பிரியாணி", "தோசை", "பீட்சா"], ["இட்லி"]),
  "hi-IN": (["बिरयानी", "डोसा", "पिज़्ज़ा"], ["समोसा"]),
  "fr-FR": (["pizza", "sushi", "biryani"], ["burger"]),
  "es-ES": (["pizza", "tacos", "paella"], ["burrito"]),
  "de-DE": (["Pizza", "Döner", "Sushi"], ["Burger"])},
 "house_place": {
  "en-US": (["kitchen", "bedroom", "living room"], ["garage"]),
  "ta-IN": (["சமையலறை", "படுக்கையறை", "வரவேற்பறை"], ["குளியலறை"]),
  "hi-IN": (["रसोई", "बेडरूम", "लिविंग रूम"], ["बाथरूम"]),
  "fr-FR": (["la cuisine", "la chambre", "le salon"], ["la salle de bain"]),
  "es-ES": (["la cocina", "el dormitorio", "el salón"], ["el baño"]),
  "de-DE": (["Küche", "Schlafzimmer", "Wohnzimmer"], ["Badezimmer"])},
 "currency_name": {
  "en-US": (["dollar", "euro", "pound", "yen"], ["dirham"]),
  "ta-IN": (["டாலர்", "யூரோ", "பவுண்ட்"], ["யென்"]),
  "hi-IN": (["डॉलर", "यूरो", "पाउंड"], ["येन"]),
  "fr-FR": (["dollar", "yen", "euro"], ["franc suisse"]),
  "es-ES": (["dólar", "euro", "yen"], ["peso"]),
  "de-DE": (["Dollar", "Euro", "Yen"], ["Franken"])},
 "news_topic": {
  "en-US": (["cricket", "politics", "technology", "stock market"], ["science"]),
  "ta-IN": (["கிரிக்கெட்", "அரசியல்", "தொழில்நுட்பம்"], ["அறிவியல்"]),
  "hi-IN": (["क्रिकेट", "राजनीति", "तकनीक"], ["विज्ञान"]),
  "fr-FR": (["politique", "technologie", "football"], ["science"]),
  "es-ES": (["política", "tecnología", "fútbol"], ["ciencia"]),
  "de-DE": (["Politik", "Technologie", "Fußball"], ["Wissenschaft"])},
}

# ---------------------------------------------------------------- templates
T = {
 "alarm_set": {
  "en-US": ["wake me up at {time} {date}", "set an alarm for {time}", "please set alarm at {time} {date}",
            "i need an alarm at {time}", "alarm for {time} {date} please"],
  "ta-IN": ["{date} {time} க்கு அலாரம் வை", "{time} அலாரம் வைக்கவும்", "{date} {time} எழுப்பி விடு",
            "{time} ஒரு அலாரம் செட் செய்", "அலாரம் {time} {date}"],
  "hi-IN": ["{date} {time} का अलार्म लगाओ", "{time} के लिए अलार्म सेट करो", "मुझे {date} {time} जगा देना",
            "{time} पर अलार्म लगा दो", "अलार्म सेट करो {time}"],
  "fr-FR": ["réveille-moi à {time} {date}", "mets une alarme à {time}", "règle le réveil pour {time} {date}",
            "j'ai besoin d'une alarme à {time}", "alarme à {time} {date} s'il te plaît"],
  "es-ES": ["despiértame a {time} {date}", "pon una alarma a {time}", "configura la alarma para {time} {date}",
            "necesito una alarma a {time}", "alarma a {time} {date} por favor"],
  "de-DE": ["weck mich {date} um {time}", "stell einen Wecker auf {time}", "Alarm für {time} {date} einstellen",
            "ich brauche einen Wecker um {time}", "bitte Wecker um {time} stellen"]},
 "alarm_remove": {
  "en-US": ["cancel my {time} alarm", "remove the alarm for {time}", "delete alarm at {time}",
            "turn off the {time} alarm", "i don't need the {time} alarm anymore"],
  "ta-IN": ["{time} அலாரத்தை ரத்து செய்", "{time} அலாரம் நீக்கு", "{time} வைத்த அலாரத்தை அழி",
            "{time} அலாரம் வேண்டாம்", "{time} அலாரத்தை நிறுத்து"],
  "hi-IN": ["{time} वाला अलार्म हटा दो", "{time} का अलार्म रद्द करो", "{time} का अलार्म डिलीट करो",
            "{time} वाला अलार्म बंद करो", "मुझे {time} का अलार्म नहीं चाहिए"],
  "fr-FR": ["annule mon alarme de {time}", "supprime l'alarme de {time}", "enlève le réveil de {time}",
            "désactive l'alarme de {time}", "je n'ai plus besoin de l'alarme de {time}"],
  "es-ES": ["cancela mi alarma de {time}", "elimina la alarma de {time}", "borra la alarma de {time}",
            "quita la alarma de {time}", "ya no necesito la alarma de {time}"],
  "de-DE": ["lösche meinen Wecker um {time}", "entferne den Alarm für {time}", "Wecker um {time} löschen",
            "schalte den Wecker um {time} aus", "ich brauche den Wecker um {time} nicht mehr"]},
 "weather_query": {
  "en-US": ["what's the weather in {place_name} {date}", "will it rain in {place_name} {date}",
            "how hot is it in {place_name}", "weather forecast for {place_name} {date}",
            "tell me the weather {date} in {place_name}"],
  "ta-IN": ["{date} {place_name} வானிலை எப்படி இருக்கும்", "{place_name} மழை பெய்யுமா {date}",
            "{place_name} வெப்பநிலை என்ன", "{place_name} வானிலை முன்னறிவிப்பு {date}", "{date} {place_name} காலநிலை சொல்"],
  "hi-IN": ["{date} {place_name} में मौसम कैसा रहेगा", "क्या {date} {place_name} में बारिश होगी",
            "{place_name} में कितनी गर्मी है", "{place_name} का मौसम पूर्वानुमान {date}", "{place_name} का मौसम बताओ {date}"],
  "fr-FR": ["quel temps fait-il à {place_name} {date}", "va-t-il pleuvoir à {place_name} {date}",
            "quelle est la température à {place_name}", "prévisions météo pour {place_name} {date}",
            "donne-moi la météo de {place_name} {date}"],
  "es-ES": ["qué tiempo hace en {place_name} {date}", "va a llover en {place_name} {date}",
            "qué temperatura hay en {place_name}", "pronóstico del tiempo para {place_name} {date}",
            "dime el clima de {place_name} {date}"],
  "de-DE": ["wie ist das Wetter in {place_name} {date}", "regnet es {date} in {place_name}",
            "wie warm ist es in {place_name}", "Wettervorhersage für {place_name} {date}",
            "sag mir das Wetter in {place_name} {date}"]},
 "play_music": {
  "en-US": ["play some songs by {artist_name}", "i want to listen to {artist_name}", "put on {artist_name} music",
            "play {artist_name}", "start playing {artist_name} songs"],
  "ta-IN": ["{artist_name} பாடல்களை போடு", "{artist_name} பாட்டு கேட்கணும்", "{artist_name} இசையை இயக்கு",
            "{artist_name} பாடல் ஒலிக்கட்டும்", "கொஞ்சம் {artist_name} பாட்டு போடு"],
  "hi-IN": ["{artist_name} के गाने चलाओ", "मुझे {artist_name} सुनना है", "{artist_name} का संगीत लगाओ",
            "{artist_name} के गीत बजाओ", "कुछ {artist_name} के गाने सुनाओ"],
  "fr-FR": ["joue des chansons de {artist_name}", "je veux écouter {artist_name}", "mets la musique de {artist_name}",
            "lance {artist_name}", "fais jouer {artist_name}"],
  "es-ES": ["pon canciones de {artist_name}", "quiero escuchar a {artist_name}", "reproduce música de {artist_name}",
            "toca algo de {artist_name}", "pon a {artist_name}"],
  "de-DE": ["spiel Lieder von {artist_name}", "ich möchte {artist_name} hören", "spiel Musik von {artist_name}",
            "starte {artist_name}", "leg {artist_name} auf"]},
 "calendar_set": {
  "en-US": ["add {event_name} to my calendar {date} at {time}", "schedule a {event_name} {date}",
            "remind me about the {event_name} {date} at {time}", "book {event_name} for {date}",
            "create an event {event_name} at {time} {date}"],
  "ta-IN": ["{date} {time} {event_name} காலண்டரில் சேர்", "{date} {event_name} திட்டமிடு",
            "{date} {event_name} பற்றி நினைவூட்டு", "{event_name} {date} பதிவு செய்",
            "{date} {time} {event_name} நிகழ்வை உருவாக்கு"],
  "hi-IN": ["{date} {time} {event_name} कैलेंडर में जोड़ो", "{date} {event_name} शेड्यूल करो",
            "{date} {time} {event_name} की याद दिलाना", "{date} के लिए {event_name} बुक करो",
            "{date} {time} {event_name} इवेंट बनाओ"],
  "fr-FR": ["ajoute {event_name} à mon agenda {date} à {time}", "planifie {event_name} {date}",
            "rappelle-moi {event_name} {date} à {time}", "réserve {event_name} pour {date}",
            "crée un événement {event_name} à {time} {date}"],
  "es-ES": ["añade {event_name} a mi calendario {date} a {time}", "programa {event_name} {date}",
            "recuérdame {event_name} {date} a {time}", "reserva {event_name} para {date}",
            "crea un evento {event_name} a {time} {date}"],
  "de-DE": ["trag {event_name} {date} um {time} in meinen Kalender ein", "plane {event_name} für {date}",
            "erinnere mich {date} um {time} an {event_name}", "buche {event_name} für {date}",
            "erstelle einen Termin {event_name} um {time} {date}"]},
 "email_sendemail": {
  "en-US": ["send an email to {person}", "email {person} that i will be late", "write a mail to {person}",
            "compose an email for {person}", "shoot {person} an email"],
  "ta-IN": ["{person} க்கு மின்னஞ்சல் அனுப்பு", "{person} க்கு நான் தாமதமாக வருவேன் என்று மெயில் செய்",
            "{person} க்கு ஒரு மெயில் எழுது", "{person} க்கு மின்னஞ்சல் தயார் செய்", "{person} க்கு ஈமெயில் அனுப்பு"],
  "hi-IN": ["{person} को ईमेल भेजो", "{person} को ईमेल करो कि मैं देर से आऊँगा", "{person} को एक मेल लिखो",
            "{person} के लिए ईमेल तैयार करो", "{person} को मेल भेज दो"],
  "fr-FR": ["envoie un e-mail à {person}", "écris à {person} que je serai en retard", "rédige un mail pour {person}",
            "compose un courriel pour {person}", "envoie un message électronique à {person}"],
  "es-ES": ["envía un correo a {person}", "escribe a {person} que llegaré tarde", "redacta un email para {person}",
            "manda un correo electrónico a {person}", "envíale un email a {person}"],
  "de-DE": ["schick eine E-Mail an {person}", "schreib {person}, dass ich später komme", "verfasse eine Mail an {person}",
            "sende {person} eine E-Mail", "maile {person}"]},
 "transport_ticket": {
  "en-US": ["book a {transport_type} ticket to {place_name} {date}", "i need a {transport_type} ticket for {place_name}",
            "get me a {transport_type} to {place_name} {date}", "reserve {transport_type} tickets to {place_name}",
            "buy a {transport_type} ticket to {place_name} for {date}"],
  "ta-IN": ["{date} {place_name} க்கு {transport_type} டிக்கெட் புக் செய்", "{place_name} க்கு {transport_type} டிக்கெட் வேண்டும்",
            "{date} {place_name} போக {transport_type} டிக்கெட் எடு", "{place_name} {transport_type} டிக்கெட் முன்பதிவு செய்",
            "{place_name} க்கு {date} {transport_type} டிக்கெட் வாங்கு"],
  "hi-IN": ["{date} {place_name} के लिए {transport_type} टिकट बुक करो", "मुझे {place_name} का {transport_type} टिकट चाहिए",
            "{date} {place_name} जाने के लिए {transport_type} टिकट लो", "{place_name} के {transport_type} टिकट आरक्षित करो",
            "{place_name} का {date} का {transport_type} टिकट खरीदो"],
  "fr-FR": ["réserve un billet de {transport_type} pour {place_name} {date}",
            "j'ai besoin d'un billet de {transport_type} pour {place_name}",
            "trouve-moi un {transport_type} pour {place_name} {date}", "achète des billets de {transport_type} pour {place_name}",
            "prends un billet de {transport_type} vers {place_name} pour {date}"],
  "es-ES": ["reserva un billete de {transport_type} a {place_name} {date}",
            "necesito un billete de {transport_type} para {place_name}",
            "consígueme un {transport_type} a {place_name} {date}", "compra billetes de {transport_type} a {place_name}",
            "saca un billete de {transport_type} a {place_name} para {date}"],
  "de-DE": ["buche eine Fahrkarte für den {transport_type} nach {place_name} {date}",
            "ich brauche ein Ticket für den {transport_type} nach {place_name}",
            "besorg mir einen {transport_type} nach {place_name} {date}",
            "reserviere Tickets für den {transport_type} nach {place_name}",
            "kauf ein Ticket für den {transport_type} nach {place_name} für {date}"]},
 "takeaway_order": {
  "en-US": ["order {food_type} for delivery", "i want to order some {food_type}", "get me {food_type} from a takeaway",
            "can you order {food_type} for dinner", "place an order for {food_type}"],
  "ta-IN": ["{food_type} ஆர்டர் செய்", "எனக்கு {food_type} ஆர்டர் பண்ணனும்", "{food_type} வீட்டுக்கு வரவழை",
            "இரவு உணவுக்கு {food_type} ஆர்டர் செய்ய முடியுமா", "{food_type} க்கு ஆர்டர் போடு"],
  "hi-IN": ["{food_type} ऑर्डर करो", "मुझे {food_type} ऑर्डर करना है", "{food_type} घर पर मंगवाओ",
            "क्या तुम डिनर के लिए {food_type} ऑर्डर कर सकते हो", "{food_type} का ऑर्डर दो"],
  "fr-FR": ["commande {food_type} en livraison", "je veux commander {food_type}", "fais-moi livrer {food_type}",
            "peux-tu commander {food_type} pour le dîner", "passe une commande de {food_type}"],
  "es-ES": ["pide {food_type} a domicilio", "quiero pedir {food_type}", "tráeme {food_type} para llevar",
            "puedes pedir {food_type} para la cena", "haz un pedido de {food_type}"],
  "de-DE": ["bestell {food_type} zum Liefern", "ich möchte {food_type} bestellen", "lass {food_type} liefern",
            "kannst du {food_type} zum Abendessen bestellen", "gib eine Bestellung für {food_type} auf"]},
 "iot_hue_lightoff": {
  "en-US": ["turn off the lights in the {house_place}", "switch off {house_place} lights", "lights off in the {house_place}",
            "kill the lights in the {house_place}", "make the {house_place} dark"],
  "ta-IN": ["{house_place} விளக்குகளை அணை", "{house_place} லைட்டை ஆஃப் செய்", "{house_place} விளக்கை நிறுத்து",
            "{house_place} வெளிச்சத்தை அணைத்து விடு", "{house_place} இருட்டாக்கு"],
  "hi-IN": ["{house_place} की लाइट बंद करो", "{house_place} की बत्ती बुझा दो", "{house_place} में लाइट ऑफ करो",
            "{house_place} की रोशनी बंद कर दो", "{house_place} में अंधेरा कर दो"],
  "fr-FR": ["éteins les lumières dans {house_place}", "coupe la lumière dans {house_place}",
            "lumières éteintes dans {house_place}", "éteins tout dans {house_place}", "plonge {house_place} dans le noir"],
  "es-ES": ["apaga las luces en {house_place}", "desconecta la luz en {house_place}", "luces apagadas en {house_place}",
            "apaga la lámpara en {house_place}", "deja a oscuras {house_place}"],
  "de-DE": ["{house_place}: Licht aus", "schalte die Lampen im Raum {house_place} aus", "Licht aus in {house_place}",
            "{house_place} Licht ausschalten", "mach es dunkel in {house_place}"]},
 "iot_hue_lighton": {
  "en-US": ["turn on the lights in the {house_place}", "switch on {house_place} lights", "lights on in the {house_place}",
            "brighten up the {house_place}", "make the {house_place} bright"],
  "ta-IN": ["{house_place} விளக்குகளை போடு", "{house_place} லைட்டை ஆன் செய்", "{house_place} விளக்கை எரியவிடு",
            "{house_place} வெளிச்சம் கொடு", "{house_place} பிரகாசமாக்கு"],
  "hi-IN": ["{house_place} की लाइट जलाओ", "{house_place} की बत्ती चालू करो", "{house_place} में लाइट ऑन करो",
            "{house_place} की रोशनी जला दो", "{house_place} में उजाला कर दो"],
  "fr-FR": ["allume les lumières dans {house_place}", "mets la lumière dans {house_place}",
            "lumières allumées dans {house_place}", "éclaire {house_place}", "illumine {house_place}"],
  "es-ES": ["enciende las luces en {house_place}", "prende la luz en {house_place}", "luces encendidas en {house_place}",
            "ilumina {house_place}", "da luz en {house_place}"],
  "de-DE": ["{house_place}: Licht an", "schalte die Lampen im Raum {house_place} ein", "Licht an in {house_place}",
            "{house_place} Licht einschalten", "mach es hell in {house_place}"]},
 "qa_currency": {
  "en-US": ["what is the exchange rate of the {currency_name}", "how much is one {currency_name} in rupees",
            "convert {currency_name} to rupees", "{currency_name} rate today", "tell me the current {currency_name} price"],
  "ta-IN": ["{currency_name} மாற்று விகிதம் என்ன", "ஒரு {currency_name} எத்தனை ரூபாய்", "{currency_name} ஐ ரூபாயாக மாற்று",
            "இன்று {currency_name} விலை", "தற்போதைய {currency_name} மதிப்பு சொல்"],
  "hi-IN": ["{currency_name} की विनिमय दर क्या है", "एक {currency_name} कितने रुपये का है", "{currency_name} को रुपये में बदलो",
            "आज {currency_name} का रेट", "{currency_name} का मौजूदा भाव बताओ"],
  "fr-FR": ["taux de change {currency_name}", "combien vaut un {currency_name} en roupies",
            "convertis le {currency_name} en roupies", "cours du {currency_name} aujourd'hui",
            "donne-moi la valeur actuelle du {currency_name}"],
  "es-ES": ["cuál es el tipo de cambio del {currency_name}", "cuánto vale un {currency_name} en rupias",
            "convierte el {currency_name} a rupias", "precio del {currency_name} hoy", "dime el valor actual del {currency_name}"],
  "de-DE": ["wie ist der Wechselkurs vom {currency_name}", "wie viel ist ein {currency_name} in Rupien",
            "rechne {currency_name} in Rupien um", "{currency_name} Kurs heute", "sag mir den aktuellen Kurs vom {currency_name}"]},
 "news_query": {
  "en-US": ["what's the latest news on {news_topic}", "show me {news_topic} headlines", "any updates about {news_topic}",
            "read the {news_topic} news", "tell me today's {news_topic} news"],
  "ta-IN": ["{news_topic} சமீபத்திய செய்திகள் என்ன", "{news_topic} தலைப்புச் செய்திகளைக் காட்டு",
            "{news_topic} பற்றி ஏதாவது புதிய தகவல்", "{news_topic} செய்திகளை படி", "இன்றைய {news_topic} செய்திகள் சொல்"],
  "hi-IN": ["{news_topic} की ताज़ा खबरें क्या हैं", "{news_topic} की सुर्खियाँ दिखाओ", "{news_topic} के बारे में कोई अपडेट",
            "{news_topic} की खबरें पढ़ो", "आज की {news_topic} खबरें बताओ"],
  "fr-FR": ["quelles sont les dernières infos en {news_topic}", "montre-moi les gros titres en {news_topic}",
            "des nouvelles en {news_topic}", "lis-moi les actualités en {news_topic}", "quoi de neuf en {news_topic} aujourd'hui"],
  "es-ES": ["cuáles son las últimas noticias de {news_topic}", "muéstrame los titulares de {news_topic}",
            "alguna novedad sobre {news_topic}", "lee las noticias de {news_topic}", "dime las noticias de {news_topic} de hoy"],
  "de-DE": ["was gibt es Neues zum Thema {news_topic}", "zeig mir Schlagzeilen zu {news_topic}",
            "gibt es Updates zu {news_topic}", "lies mir die Nachrichten über {news_topic} vor",
            "sag mir die heutigen Nachrichten zu {news_topic}"]},
}
SCENARIO = {"alarm_set": "alarm", "alarm_remove": "alarm", "weather_query": "weather", "play_music": "play",
            "calendar_set": "calendar", "email_sendemail": "email", "transport_ticket": "transport",
            "takeaway_order": "takeaway", "iot_hue_lightoff": "iot", "iot_hue_lighton": "iot",
            "qa_currency": "qa", "news_query": "news"}
PER_TEMPLATE = 6


def fill(template, locale, rng, unseen_prob):
    utt, annot = template, template
    for slot in re.findall(r"\{(\w+)\}", template):
        seen, unseen = F[slot][locale]
        pool = unseen if (unseen and rng.random() < unseen_prob) else seen
        v = rng.choice(pool)
        utt = utt.replace("{" + slot + "}", v, 1)
        annot = annot.replace("{" + slot + "}", f"[{slot} : {v}]", 1)
    return utt, annot


def main():
    rng = random.Random(2026)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    rid = 0
    for intent, by_loc in T.items():
        for loc in LOCALES:
            temps = by_loc[loc]
            assert len(temps) == 5, (intent, loc)
            for ti, tmpl in enumerate(temps):
                part = "train" if ti < 3 else "test"
                seen_utts = set()
                for _ in range(PER_TEMPLATE * 4):
                    if len(seen_utts) >= PER_TEMPLATE:
                        break
                    utt, annot = fill(tmpl, loc, rng, 0.0 if part == "train" else 0.35)
                    if utt in seen_utts:
                        continue
                    seen_utts.add(utt)
                    records.append({"id": str(rid), "locale": loc, "partition": part, "scenario": SCENARIO[intent],
                                    "intent": intent, "utt": utt, "annot_utt": annot, "template": ti})
                    rid += 1
    with open(OUT / "massive_style.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(records)} utterances -> {OUT / 'massive_style.jsonl'}")
    print(Counter((r["partition"]) for r in records), Counter(r["locale"] for r in records))


if __name__ == "__main__":
    main()
