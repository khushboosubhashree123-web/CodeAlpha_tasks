from flask import Flask, render_template, request
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Popular languages
LANGUAGES = {
    'en': 'English', 
    'hi': 'Hindi', 
    'ta': 'Tamil', 
    'te': 'Telugu',
    'kn': 'Kannada', 
    'ml': 'Malayalam', 
    'mr': 'Marathi', 
    'bn': 'Bengali',
    'gu': 'Gujarati', 
    'fr': 'French', 
    'de': 'German', 
    'es': 'Spanish',
    'ja': 'Japanese', 
    'ko': 'Korean', 
    'zh-CN': 'Chinese (Simplified)',
    'ar': 'Arabic'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""
    original_text = ""
    source_lang = "auto"
    target_lang = "hi"

    if request.method == 'POST':
        original_text = request.form.get('text', '').strip()
        source_lang = request.form.get('source_lang', 'auto')
        target_lang = request.form.get('target_lang', 'hi')

        if original_text:
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                translated_text = translator.translate(original_text)
            except Exception as e:
                translated_text = f"❌ Translation Error: {str(e)}"

    return render_template('index.html', 
                         translated_text=translated_text,
                         original_text=original_text,
                         source_lang=source_lang,
                         target_lang=target_lang,
                         languages=LANGUAGES)

if __name__ == '__main__':
    app.run(debug=True)