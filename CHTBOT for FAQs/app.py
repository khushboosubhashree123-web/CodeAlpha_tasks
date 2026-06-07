from flask import Flask, render_template, request, jsonify
import string
import re

# Simple fallback tokenization 
def simple_tokenize(text):
    """Simple word tokenizer without NLTK"""
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into words
    tokens = text.split()
    # Remove short/stop words
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
                  "for", "of", "with", "by", "is", "are", "was", "were", "be", 
                  "this", "that", "it", "i", "you", "me", "my", "your"}
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(tokens)

app = Flask(__name__)

#  SAMPLE FAQS 
faqs = [
    {"question": "1. What is CodeAlpha?", 
     "answer": "CodeAlpha is a tech platform and company based in Lucknow, Uttar Pradesh, India, focused on empowering students and beginners in technology. It offers virtual internships, online compilers, tutorials, interview preparation resources, and project-based learning experiences."},
    {"question": "2. When was CodeAlpha founded and where is it located?",
      "answer": "CodeAlpha was founded in 2022 and is headquartered in Lucknow, India. It operates primarily as a remote/virtual platform for students worldwide."},
    {"question": "3. What kind of internships does CodeAlpha offer?", 
     "answer": "CodeAlpha provides free or low-barrier virtual (remote) internships in various tech domains, including Web Development, Python Programming, App Development, Data Science, Cyber Security, AI & Machine Learning, and more. Internships are typically 4–8 weeks long and project-based."},
    {"question": "4. Are CodeAlpha internships paid?",
      "answer": "Most CodeAlpha internships are unpaid/virtual programs designed for skill-building and resume experience. Some programs may offer cash prizes for top performers, but there is generally no fixed stipend."},
    {"question": "5. Do participants receive a certificate upon completion?",
      "answer": "GYes. Successful completion of internship tasks usually results in an official internship certificate, Letter of Recommendation (LOR), and sometimes other recognitions. Certificate verification is available on their platform."},
    {"question": "6. Is CodeAlpha legitimate or a scam?",
      "answer": "YCodeAlpha has a mixed reputation. Many students appreciate the project-based learning and certificate for resume building. However, there are numerous reports and Reddit discussions alleging that they request small payments (e.g., ₹100–200) for certificates at the end, minimal mentorship, and mass offer letters, leading some to label it as low-quality or potentially scammy. Always research thoroughly and avoid any unexpected payments."},
    {"question": "7. How can I apply for a CodeAlpha internship?",
      "answer": "You can apply through their official website (codealpha.tech) or registration forms (often Google Forms shared on their social channels). Applications are usually open year-round with rolling batches."},
]

faqs.append({
    "question": "8. What do interns actually do during the program?",
    "answer": "Interns receive tasks or projects related to their domain (e.g., building a website, Python scripts, or data analysis tools). They work independently or with provided resources and submit completed work for review. There is limited real-time mentorship reported by some participants."
})

# Prepare FAQ data
faq_questions = [faq["question"] for faq in faqs]
processed_faqs = [simple_tokenize(q) for q in faq_questions]

# Use TF-IDF for smart matching
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_faqs)

def get_best_match(user_question):
    if not user_question or not user_question.strip():
        return "Please ask me something! 😊"
   
    processed = simple_tokenize(user_question)
    if not processed.strip():
        return "Sorry, I didn't understand that. Try asking in simple words."
   
    user_vec = vectorizer.transform([processed])
    similarities = cosine_similarity(user_vec, faq_vectors)
    best_idx = similarities.argmax()
    score = similarities[0][best_idx]
   
    # Good match threshold
    if score > 0.25:
        return faqs[best_idx]["answer"]
    else:
        return "Sorry, I couldn't find an exact match. Try rephrasing your question!"

# ROUTES 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    message = data.get('message', '').strip()
    response = get_best_match(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    print("🚀 CodeAlpha FAQ Chatbot is running at http://127.0.0.1:5000")
    print("✅ Ready! Open the link in your browser.")
    app.run(debug=True, host='0.0.0.0', port=5000)