"""Tradução da interface. A língua base é o português; outras línguas vêm de ficheiros forja/lang/<código>_*.tsv
(uma linha por texto: original<TAB>tradução). Para acrescentar uma língua basta criar um ficheiro novo."""
import glob
import os

LANG_DIR = os.path.join(os.path.dirname(__file__), "lang")
NAMES = {"pt": "Português", "en": "English", "es": "Español", "fr": "Français", "de": "Deutsch", "it": "Italiano",
         "zh": "中文", "ja": "日本語", "ru": "Русский", "pl": "Polski", "nl": "Nederlands", "el": "Ελληνικά",
         "cs": "Čeština", "sk": "Slovenčina", "sl": "Slovenščina", "lt": "Lietuvių", "th": "ไทย", "ms": "Bahasa Melayu"}

_table = {}
_lang = "pt"


def available():
    codes = {"pt"}
    for f in glob.glob(os.path.join(LANG_DIR, "*.tsv")):
        codes.add(os.path.basename(f).split("_")[0].split(".")[0])
    return sorted(codes, key=lambda c: (c != "pt", c))


def load(lang):
    global _table, _lang
    _lang = lang
    _table = {}
    if lang == "pt":
        return
    for f in sorted(glob.glob(os.path.join(LANG_DIR, f"{lang}_*.tsv")) + glob.glob(os.path.join(LANG_DIR, f"{lang}.tsv"))):
        for line in open(f, encoding="utf-8"):
            line = line.rstrip("\n")
            if "\t" in line and not line.startswith("#"):
                a, b = line.split("\t", 1)
                _table[a] = b


def current():
    return _lang


def tr(text):
    if not _table or not isinstance(text, str):
        return text
    key = text.replace("&", "")
    if key in _table:
        return _table[key]
    # textos com prefixo variável: traduz a parte final conhecida
    return text
