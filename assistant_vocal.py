"""
Assistant vocal Claude - interrupteur ON/OFF + choix des périphériques audio.
Lancement : python assistant_vocal.py   (la clé API se saisit dans la fenêtre)
"""
import json, os, re, subprocess, sys, tempfile, threading, queue, webbrowser
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
import soundfile as sf
import pyttsx3
from anthropic import Anthropic

MODEL = "claude-sonnet-5-5"
WHISPER_SIZE = "small"          # "base" = plus rapide, "medium" = plus précis
LANGUE = "fr"

# ---- Liste blanche : seules ces applis peuvent être ouvertes (à personnaliser) ----
APPS = {
    "bloc-notes": "notepad", "calculatrice": "calc", "explorateur": "explorer",
    "chrome": "chrome", "firefox": "firefox", "edge": "msedge",
    "spotify": "spotify", "discord": "discord", "paint": "mspaint",
}

SYSTEM = f"""Tu es un assistant vocal sur le PC de l'utilisateur. Réponds en français, de façon courte (1-2 phrases), car ta réponse est lue à voix haute.
Réponds UNIQUEMENT avec un objet JSON, sans texte autour :
{{"say": "ce que tu dis à voix haute", "action": "none" | "open_app" | "open_url", "target": "..."}}
- open_app : target doit être l'un de ces noms exacts : {list(APPS)}
- open_url : target est une URL complète en https://
- Pour une simple question, action = "none"."""

SR_WHISPER = 16000
CONFIG = os.path.join(os.path.expanduser("~"), ".assistant_vocal.json")


def charger_cle():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f).get("api_key", "")
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def sauver_cle(cle):
    try:
        with open(CONFIG, "w", encoding="utf-8") as f:
            json.dump({"api_key": cle}, f)
    except Exception:
        pass


# ------------------------------------------------------------------ audio utils
def lister_peripheriques():
    """Retourne (entrées, sorties) de l'API audio par défaut : listes de (index, nom)."""
    api = sd.query_hostapis(sd.default.hostapi)
    ins, outs = [], []
    for i in api["devices"]:
        d = sd.query_devices(i)
        if d["max_input_channels"] > 0:
            ins.append((i, d["name"]))
        if d["max_output_channels"] > 0:
            outs.append((i, d["name"]))
    return ins, outs


def reechantillonner(audio, sr_src, sr_dst=SR_WHISPER):
    if sr_src == sr_dst:
        return audio
    n = int(len(audio) * sr_dst / sr_src)
    return np.interp(np.linspace(0, len(audio), n, endpoint=False),
                     np.arange(len(audio)), audio).astype(np.float32)


# ------------------------------------------------------------------ moteur
class Assistant:
    def __init__(self, log):
        self.log = log
        self.client = None                 # créé au démarrage avec la clé saisie
        self.whisper = None
        self.actif = threading.Event()     # = interrupteur
        self.thread = None
        self.parle = threading.Event()
        self.in_dev = None
        self.out_dev = None
        self.historique = []

    # --- interrupteur ---
    def demarrer(self, in_dev, out_dev, cle):
        self.in_dev, self.out_dev = in_dev, out_dev
        self.client = Anthropic(api_key=cle)
        self.actif.set()
        self.thread = threading.Thread(target=self.boucle, daemon=True)
        self.thread.start()

    def arreter(self):
        self.actif.clear()                 # coupe micro + plus aucune action possible
        try:
            sd.stop()
        except Exception:
            pass

    # --- boucle d'écoute ---
    def boucle(self):
        try:
            if self.whisper is None:
                self.log("Chargement de la reconnaissance vocale (1ère fois : un peu long)...")
                from faster_whisper import WhisperModel
                self.whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
            sr = int(sd.query_devices(self.in_dev)["default_samplerate"])
            bloc = int(sr * 0.03)
            q = queue.Queue()

            def cb(indata, frames, t, status):
                if not self.parle.is_set():          # on n'écoute pas pendant que l'assistant parle
                    q.put(indata[:, 0].copy())

            with sd.InputStream(device=self.in_dev, channels=1, samplerate=sr,
                                blocksize=bloc, callback=cb):
                self.log("🎤 À l'écoute...")
                self.ecouter(q, sr)
        except Exception as e:
            self.log(f"Erreur audio : {e}")
        self.log("⛔ Micro coupé.")

    def ecouter(self, q, sr):
        SEUIL, SILENCE_MAX = 0.015, 25            # ~0,75 s de silence = fin de phrase
        buf, parole, silence = [], False, 0
        while self.actif.is_set():
            try:
                bloc = q.get(timeout=0.2)
            except queue.Empty:
                continue
            fort = np.sqrt(np.mean(bloc ** 2)) > SEUIL
            if fort:
                parole, silence = True, 0
            elif parole:
                silence += 1
            if parole:
                buf.append(bloc)
            if parole and silence > SILENCE_MAX:
                audio = np.concatenate(buf)
                buf, parole, silence = [], False, 0
                if len(audio) > sr * 0.4:          # ignore les bruits très courts
                    self.traiter(audio, sr)
                    while not q.empty():
                        q.get_nowait()

    # --- traitement d'une phrase ---
    def traiter(self, audio, sr):
        audio = reechantillonner(audio.astype(np.float32), sr)
        segments, _ = self.whisper.transcribe(audio, language=LANGUE, vad_filter=True)
        texte = " ".join(s.text for s in segments).strip()
        if not texte or not self.actif.is_set():
            return
        self.log(f"👤 {texte}")
        self.historique.append({"role": "user", "content": texte})
        self.historique = self.historique[-10:]
        try:
            rep = self.client.messages.create(model=MODEL, max_tokens=300,
                                              system=SYSTEM, messages=self.historique)
            brut = rep.content[0].text
            self.historique.append({"role": "assistant", "content": brut})
            m = re.search(r"\{.*\}", brut, re.S)
            data = json.loads(m.group(0)) if m else {"say": brut, "action": "none"}
        except Exception as e:
            self.log(f"Erreur API : {e}")
            return
        if not self.actif.is_set():               # interrupteur coupé entre-temps : on n'exécute rien
            return
        self.executer(data.get("action", "none"), str(data.get("target", "")))
        self.dire(data.get("say", ""))

    # --- actions (liste blanche) ---
    def executer(self, action, cible):
        if action == "open_url" and cible.startswith(("https://", "http://")):
            self.log(f"🌐 Ouverture : {cible}")
            webbrowser.open(cible)
        elif action == "open_app" and cible.lower() in APPS:
            cmd = APPS[cible.lower()]
            self.log(f"🚀 Ouverture : {cible}")
            try:
                if sys.platform.startswith("win"):
                    subprocess.Popen(["cmd", "/c", "start", "", cmd])
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-a", cmd])
                else:
                    subprocess.Popen([cmd])
            except Exception as e:
                self.log(f"Impossible d'ouvrir {cible} : {e}")
        elif action != "none":
            self.log(f"Action refusée : {action} {cible}")

    # --- voix (jouée sur le périphérique de sortie choisi) ---
    def dire(self, texte):
        if not texte or not self.actif.is_set():
            return
        self.log(f"🤖 {texte}")
        self.parle.set()
        try:
            fichier = os.path.join(tempfile.gettempdir(), "assistant_tts.wav")
            moteur = pyttsx3.init()
            moteur.save_to_file(texte, fichier)
            moteur.runAndWait()
            data, sr_out = sf.read(fichier, dtype="float32")
            sd.play(data, sr_out, device=self.out_dev)
            sd.wait()
        except Exception as e:
            self.log(f"Erreur voix : {e}")
        finally:
            self.parle.clear()


# ------------------------------------------------------------------ interface
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Assistant vocal")
        self.geometry("520x530")
        self.resizable(False, False)
        self.ins, self.outs = lister_peripheriques()
        self.assistant = Assistant(self.log)

        ttk.Label(self, text="Assistant vocal", font=("Segoe UI", 16, "bold")).pack(pady=(14, 8))

        cadre = ttk.Frame(self)
        cadre.pack(fill="x", padx=20)
        ttk.Label(cadre, text="Entrée (micro) :").grid(row=0, column=0, sticky="w", pady=4)
        self.cb_in = ttk.Combobox(cadre, state="readonly", width=48,
                                  values=[n for _, n in self.ins])
        self.cb_in.grid(row=1, column=0, sticky="w")
        ttk.Label(cadre, text="Sortie (casque / haut-parleurs) :").grid(row=2, column=0, sticky="w", pady=(10, 4))
        self.cb_out = ttk.Combobox(cadre, state="readonly", width=48,
                                   values=[n for _, n in self.outs])
        self.cb_out.grid(row=3, column=0, sticky="w")
        ttk.Label(cadre, text="Clé API Anthropic :").grid(row=4, column=0, sticky="w", pady=(10, 4))
        self.cle = ttk.Entry(cadre, width=51, show="•")
        self.cle.grid(row=5, column=0, sticky="w")
        self.cle.insert(0, charger_cle())
        self.selection_par_defaut()

        self.btn = tk.Button(self, text="OFF", font=("Segoe UI", 16, "bold"), width=12,
                             bg="#c0392b", fg="white", relief="flat", command=self.basculer)
        self.btn.pack(pady=16)

        self.zone = tk.Text(self, height=11, state="disabled", wrap="word")
        self.zone.pack(fill="both", padx=20, pady=(0, 14))
        self.protocol("WM_DELETE_WINDOW", self.quitter)
        self.log("Choisis tes périphériques puis appuie sur le bouton.")

    def selection_par_defaut(self):
        di, do = sd.default.device
        for cb, liste, defaut in ((self.cb_in, self.ins, di), (self.cb_out, self.outs, do)):
            idx = next((k for k, (i, _) in enumerate(liste) if i == defaut), 0)
            if liste:
                cb.current(idx)

    def log(self, msg):
        def ecrire():
            self.zone.config(state="normal")
            self.zone.insert("end", msg + "\n")
            self.zone.see("end")
            self.zone.config(state="disabled")
        self.after(0, ecrire)

    def basculer(self):
        if self.assistant.actif.is_set():
            self.assistant.arreter()
            self.btn.config(text="OFF", bg="#c0392b")
            self.cb_in.config(state="readonly")
            self.cb_out.config(state="readonly")
            self.cle.config(state="normal")
        else:
            if self.cb_in.current() < 0 or self.cb_out.current() < 0:
                self.log("Sélectionne un micro et une sortie.")
                return
            cle = self.cle.get().strip()
            if not cle:
                self.log("Colle ta clé API Anthropic dans le champ prévu.")
                return
            sauver_cle(cle)
            self.assistant.demarrer(self.ins[self.cb_in.current()][0],
                                    self.outs[self.cb_out.current()][0], cle)
            self.cle.config(state="disabled")
            self.btn.config(text="ON", bg="#27ae60")
            self.cb_in.config(state="disabled")   # pas de changement de périphérique en cours d'écoute
            self.cb_out.config(state="disabled")

    def quitter(self):
        self.assistant.arreter()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
