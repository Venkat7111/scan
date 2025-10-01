import os
from flask import Flask, render_template, request, session
import google.generativeai as genai
from langdetect import detect

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

# Configure Gemini API
API_KEY = os.environ.get("AIzaSyCY_LA2oVPUPvHbmVmznCxqRZkuciG0N2U", "")
if not API_KEY:
	# Allow running with a hardcoded key via env for dev, but avoid committing keys
	pass

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
chat = model.start_chat()

SUPPORTED_LANGS = {
	"en": "English",
	"te": "Telugu",
	"ta": "Tamil",
	"hi": "Hindi",
	"kn": "Kannada",
	"ml": "Malayalam",
}


def translate_text(text: str, target_lang_code: str) -> str:
	if target_lang_code == "en":
		return text
	prompt = (
		f"Translate the following text into {SUPPORTED_LANGS.get(target_lang_code, target_lang_code)}.\n"
		"Only return the translation, no extra commentary.\n\n"
		f"Text: {text}"
	)
	resp = model.generate_content(prompt)
	return resp.text or ""


def bot_reply(user_message: str, target_lang_code: str) -> str:
	resp = chat.send_message(user_message)
	text = resp.text or ""
	translated = translate_text(text, target_lang_code)
	return translated


@app.route("/", methods=["GET", "POST"])
def index():
	if 'chat_history' not in session:
		session['chat_history'] = []
	selected_lang = request.form.get("lang", "en")

	if request.method == "POST":
		user_message = request.form.get("user_input", "").strip()
		if user_message:
			try:
				# Detect user input language (optional display)
				user_lang = ""
				try:
					user_lang = detect(user_message)
				except Exception:
					user_lang = ""

				bot_message = bot_reply(user_message, selected_lang)
				session['chat_history'].append(("YOU", user_message, user_lang))
				session['chat_history'].append(("BOT", bot_message, selected_lang))
				session.modified = True
			except Exception as e:
				session['chat_history'].append(("BOT", f"Error: {e}", selected_lang))
				session.modified = True

	return render_template("index.html", chat_history=session['chat_history'], langs=SUPPORTED_LANGS, selected_lang=selected_lang)


if __name__ == "__main__":
	app.run(debug=True)
