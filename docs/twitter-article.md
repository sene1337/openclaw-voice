# OpenClaw Voice: Talk to Your AI Like You Talk to Alexa

**Free. Open-source. Self-hosted. No subscriptions.**

---

## The Problem

Voice AI is everywhere—Alexa, Siri, Google Assistant—but they're all locked ecosystems. Want to talk to YOUR AI? The one you've customized? The one with YOUR context?

You've got two options:
1. **Pay $0.08-0.15/minute** for hosted voice AI (ElevenLabs Agents, Retell.ai)
2. **Spend 3 days** configuring WebRTC, Whisper, VAD, and a dozen other acronyms

Neither is great.

## The Solution

**OpenClaw Voice** is a browser-based voice interface you can self-host in 5 minutes.

- 🎤 **Local STT** — Whisper runs on YOUR machine. Your voice never leaves your server.
- 🔊 **Premium TTS** — ElevenLabs integration for natural speech.
- 🌐 **Works in any browser** — Desktop, mobile, no app install.
- 🔌 **Connect any AI** — OpenAI, Claude, or your own custom agent.
- 🏠 **100% self-hosted** — Your data stays yours.

## How It Works

```
Browser → WebSocket → [Whisper STT] → [Your AI] → [ElevenLabs TTS] → Browser
```

That's it. Your voice gets transcribed locally, sent to your AI, and the response comes back as speech.

## Quick Start

If you're technical (or have an AI assistant that is):

```bash
git clone https://github.com/Purple-Horizons/openclaw-voice.git
cd openclaw-voice
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ELEVENLABS_API_KEY="your-key" OPENAI_API_KEY="your-key" \
  python -m src.server.main
```

Open http://localhost:8765. Start talking.

For mobile (HTTPS required), use Tailscale Funnel:
```bash
tailscale funnel 8765
```

## For OpenClaw Users

If you're already running OpenClaw, you can connect Voice directly to your agent. Same context, same memory, same tools—just voice.

Add to your `openclaw.json`:
```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  }
}
```

Now your voice conversations route through your full agent.

## Why Open Source?

Because voice AI shouldn't be a subscription. 

The models exist. The tools exist. The only thing missing was someone putting them together in a way that doesn't require a PhD in audio engineering.

OpenClaw Voice is MIT licensed. Fork it. Modify it. Run it on a Raspberry Pi. We don't care. Just build cool stuff.

## Links

- 🦞 **Website:** [openclawvoice.com](https://openclawvoice.com)
- 📦 **GitHub:** [github.com/Purple-Horizons/openclaw-voice](https://github.com/Purple-Horizons/openclaw-voice)
- 🔧 **OpenClaw:** [openclaw.ai](https://openclaw.ai)

---

*Built by [Purple Horizons](https://purplehorizons.io) in Miami. Part of the OpenClaw ecosystem.*
