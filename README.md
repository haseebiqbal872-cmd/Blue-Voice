# Blue Voice

**Sign. Speak. Be heard.** Blue Voice is a free sign-language AI. It turns hand signs into text and speech for
Deaf people and people who cannot speak, and lets them **talk to any AI model by signing**. The camera, tracking and
recognition all run on your own computer.


> ## Please read this first
> - **Blue Voice knows 250 words right now** (American Sign Language). We are still adding more.
> - **To use it every day, first record your own words.** Open *Settings → Save & record signs* and add the words you
>   use to communicate. Sign each one 3 times and Blue Voice learns it. Your own signs are checked before the built-in list.
> - It is early software. It has been tested with recorded clips and simulated cameras, **not yet with many real signers**.
>   Accuracy in real life is unmeasured. Please try it and open an issue.

## What it does

- **Sign words → text.** 250 built-in ASL words plus any signs you record yourself.
- **Welcome and sign-in.** A handwritten "Hello" in many languages on first start, then "Welcome to Blue Voice" and sign-in with **GitHub** or **Google** (or continue without signing in).
- **Record your own signs.** Show an open palm to start, sign 3 times, preview your own sign slowly, save. No video is ever stored.
- **Fingerspelling keyboard.** Pick letters from a hand-picture keyboard to type the word you want to record.
- **Palm to send.** Hold an open palm for about a second to send what you signed.
- **Talk to an AI, two ways**
  1. **Overlay mode** (`overlay.bat`): everything disappears except three buttons (Pause / Play / End conversation) and the message box,
     floating over the app in front of you (ChatGPT, Claude, Gemini, any chat). Sign, show your palm, and the text is **pasted into
     that app's message box**. *You* press its Send button; Blue Voice never does. **End conversation** brings Blue Voice back.
  2. **Built-in assistant:** connect any model in *Settings → AI models*: Claude, OpenAI, Gemini, Ollama (local and free),
     OpenRouter, Groq, or any OpenAI-compatible address. Replies stream in as they are written and chats are saved on your computer.
     The assistant can read, write, edit and delete files in a folder **you** choose, search the web, and (if you switch it on) run commands.
- **Share signs.** *Settings → Updates & signs* saves your signs as a small file and adds signs from someone else's file.
- **Updates.** The same page checks GitHub for a newer version.

## Install

You need Python 3.10+ and a webcam.

    Windows:   double-click start.bat        (overlay: overlay.bat)
    Mac/Linux: ./start.sh

By hand:

    pip install -r requirements.txt
    python main.py                       # opens http://127.0.0.1:8765
    pip install -r requirements-overlay.txt && python overlay.py

The first run downloads Google's MediaPipe tracking models (Apache 2.0) once into `assets/models/`.

## Privacy and safety

- Blue Voice only listens on `127.0.0.1` (this computer).
- Camera frames are processed in memory. The preview of a sign you record is kept in memory only and never saved as video.
- Sign-in with GitHub or Google only reads your name and email, and Blue Voice keeps no token afterwards. There is no Blue Voice cloud or password database. To switch it on you register the sign-in apps once: see [docs/SIGN_IN_SETUP.md](docs/SIGN_IN_SETUP.md).
- API keys are stored in `data/llm.json`, never sent to the web page. `data/` is git-ignored. **Never commit it.**
- The assistant's file tools stay inside the folder you choose. Web fetch refuses private and loopback addresses.
  File changes ask for your approval; **commands are off by default and always ask**. An approved command runs with your own user rights.

## Known limits

- 250 words plus your own recordings. Signs outside that list can be mis-read as a known word.
- Live recognition of fingerspelled letters from the camera is not connected yet (the on-screen keyboard works).
- The overlay and paste features are Windows-first and need more real-computer testing.
- Saved chats are plain text in `data/chats/`.

## Add words, help us grow

Blue Voice grows through people recording the words they really use. Record yours, share a sign pack, and open an issue to tell us what is missing. Signs from other sign languages will be added only when they can be verified with Deaf signers or official dictionaries.

## Tests

    pip install -r requirements.txt
    python -m pytest -q tests

## Project layout

`main.py` app · `overlay.py` overlay window · `server.py` local HTTP + WebSocket bridge · `web/` interface ·
`core/gs/` sign recognition · `core/word_session.py` live recogniser and recorder · `core/agent/` AI agent and tools ·
`core/inject.py` paste into the app in front · `core/oauth.py` GitHub and Google sign-in · `core/packs.py` sign packs · `tools/` training scripts (`requirements-train.txt`).

## Licence and credits

Copyright (c) 2026 Haseeb Iqbal. All rights reserved. Free for personal use; copying, modifying or redistributing needs written permission. See [LICENSE](LICENSE). Third-party models, data and fonts are credited in [CREDITS.md](CREDITS.md):
GestureSign (MIT), Google *Isolated Sign Language Recognition* (CC BY 4.0), MediaPipe (Apache 2.0), FSboard (CC BY 4.0), Great Vibes font (SIL OFL 1.1).

See also [SECURITY.md](SECURITY.md).
