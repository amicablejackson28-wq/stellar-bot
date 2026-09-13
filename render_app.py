"""
Render entry point — runs the Stellar trading bot in a background thread
and exposes a tiny web server so Render (a web-service host) considers
the deployment healthy and keeps it running.

This file does not contain trading logic itself — see bot_trend_rsi.py
for that. This just makes the bot "look like a website" to Render.
"""

import threading
import os
from flask import Flask

import bot_trend_rsi  # the actual trading bot logic

app = Flask(__name__)

bot_thread_started = False


@app.route("/")
def health_check():
    return "Stellar trading bot is running.", 200


def start_bot_in_background():
    global bot_thread_started
    if not bot_thread_started:
        bot_thread_started = True
        thread = threading.Thread(target=bot_trend_rsi.run_bot, daemon=True)
        thread.start()


start_bot_in_background()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
