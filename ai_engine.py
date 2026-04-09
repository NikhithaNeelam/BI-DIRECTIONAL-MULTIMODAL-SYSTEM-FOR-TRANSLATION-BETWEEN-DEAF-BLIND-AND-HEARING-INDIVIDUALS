"""
ai_engine.py — FIXED: Ollama via HTTP (works regardless of drive location)
Falls back to rule-based when Ollama is offline.
"""
import requests, json, re

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
TIMEOUT      = 8

def is_ollama_running():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except: return False

def _call_ollama(prompt):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False},
                          timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("response","").strip()
    except: pass
    return None

# ── Rule-based gloss→sentence ────────────────────────────
PHRASE_MAP = {
    "HAI":               {"English":"Hello!","Hindi":"नमस्ते!","Telugu":"హలో!"},
    "BYE":               {"English":"Goodbye!","Hindi":"अलविदा!","Telugu":"వీడ్కోలు!"},
    "GOOD":              {"English":"That is good.","Hindi":"यह अच्छा है।","Telugu":"అది మంచిది."},
    "FOOD":              {"English":"I need food.","Hindi":"मुझे खाना चाहिए।","Telugu":"నాకు ఆహారం కావాలి."},
    "FOOD NEEDED":       {"English":"Food is needed.","Hindi":"खाना चाहिए।","Telugu":"ఆహారం కావాలి."},
    "WELCOME":           {"English":"Welcome!","Hindi":"स्वागत है!","Telugu":"స్వాగతం!"},
    "DONE":              {"English":"It is done.","Hindi":"हो गया।","Telugu":"అయిపోయింది."},
    "YES":               {"English":"Yes.","Hindi":"हाँ।","Telugu":"అవును."},
    "NO":                {"English":"No.","Hindi":"नहीं।","Telugu":"కాదు."},
    "HELLO":             {"English":"Hello!","Hindi":"नमस्ते!","Telugu":"హలో!"},
    "THANK YOU":         {"English":"Thank you.","Hindi":"धन्यवाद।","Telugu":"ధన్యవాదాలు."},
    "GIVE ME":           {"English":"Please give me that.","Hindi":"कृपया दीजिए।","Telugu":"దయచేసి ఇవ్వండి."},
    "CAN I GO":          {"English":"Can I go?","Hindi":"क्या मैं जा सकता हूँ?","Telugu":"నేను వెళ్ళవచ్చా?"},
    "YOUR NAME":         {"English":"What is your name?","Hindi":"आपका नाम क्या है?","Telugu":"మీ పేరు ఏమిటి?"},
    "IM GOING":          {"English":"I am going.","Hindi":"मैं जा रहा हूँ।","Telugu":"నేను వెళ్తున్నాను."},
    "YOU ARE MY FRIEND": {"English":"You are my friend.","Hindi":"आप मेरे दोस्त हैं।","Telugu":"మీరు నా స్నేహితుడు."},
    "GO AHEAD":          {"English":"Please go ahead.","Hindi":"कृपया आगे बढ़ें।","Telugu":"దయచేసి ముందుకు వెళ్ళండి."},
    "PALM":              {"English":"Open palm.","Hindi":"खुली हथेली।","Telugu":"చేయి తెరవండి."},
    "FIST":              {"English":"Closed fist.","Hindi":"मुट्ठी।","Telugu":"పిడికిలి."},
    "THUMB_UP":          {"English":"Thumbs up!","Hindi":"शाबाश!","Telugu":"శభాష్!"},
    "OK":                {"English":"Okay!","Hindi":"ठीक है!","Telugu":"సరే!"},
    "THUMB_DOWN":        {"English":"Not good.","Hindi":"अच्छा नहीं।","Telugu":"మంచిది కాదు."},
    "INDEX_POINT":       {"English":"Look there.","Hindi":"वहाँ देखो।","Telugu":"అక్కడ చూడండి."},
}

REVERSE_MAP = {
    "HELLO":"HAI","HI":"HAI","GOODBYE":"BYE","BYE":"BYE",
    "THANK YOU":"THANK YOU","THANKS":"THANK YOU",
    "YES":"YES","NO":"NO","OKAY":"OK","OK":"OK","GOOD":"GOOD",
    "WELCOME":"WELCOME","FOOD":"FOOD","I NEED FOOD":"FOOD NEEDED",
    "CAN I GO":"CAN I GO","CAN I LEAVE":"CAN I GO",
    "WHAT IS YOUR NAME":"YOUR NAME","WHAT IS YOUR NAME?":"YOUR NAME",
    "YOU ARE MY FRIEND":"YOU ARE MY FRIEND",
    "I AM GOING":"IM GOING","I AM LEAVING":"IM GOING",
    "GIVE ME":"GIVE ME","PLEASE GIVE ME":"GIVE ME",
    "THUMBS UP":"THUMB_UP","THUMB UP":"THUMB_UP",
    "FIST":"FIST","PALM":"PALM",
}

STOP_WORDS = {"I","AM","IS","ARE","THE","A","AN","TO","OF","IN","AND","OR",
              "IT","THIS","THAT","MY","YOUR","PLEASE","WILL","CAN","DO",
              "BE","HAVE","HAS","WAS","WERE","FOR","WITH","ON","AT"}

def _rule_refine(words, lang):
    up = words.strip().upper()
    if up in PHRASE_MAP:
        return PHRASE_MAP[up].get(lang, PHRASE_MAP[up]["English"])
    parts = [w.capitalize() for w in up.split()]
    return " ".join(parts) + "."

def _rule_gloss(text):
    up = text.strip().upper()
    for phrase,gloss in REVERSE_MAP.items():
        if phrase in up: return gloss
    tokens = [w for w in re.findall(r'\b\w+\b',up) if w not in STOP_WORDS]
    return " ".join(tokens) if tokens else up

# ── Public API ───────────────────────────────────────────
def refine_sentence(words, target_lang="English"):
    """Sign glosses → fluent sentence."""
    if is_ollama_running():
        prompt = (f"Convert sign language glosses \"{words}\" into one natural "
                  f"sentence in {target_lang}. Output only the sentence.")
        result = _call_ollama(prompt)
        if result: return result
    return _rule_refine(words, target_lang)

def translate_text(text, target_lang="English"):
    """Sentence → ISL gloss sequence for avatar."""
    up = text.strip().upper()
    if up in PHRASE_MAP: return up
    if is_ollama_running():
        prompt = (f"Convert to simple sign language glosses (uppercase keywords only): "
                  f"\"{text}\". Example: 'I am going' -> 'ME GO'. Output only glosses.")
        result = _call_ollama(prompt)
        if result:
            glosses = " ".join(re.findall(r'\b[A-Z0-9]+\b', result.upper()))
            if glosses: return glosses
    return _rule_gloss(text)
