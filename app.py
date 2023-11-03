import json
from flask import Flask, render_template, request, jsonify
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the business guidelines from a JSON file
with open("business_guidelines.json", "r") as f:
    guidelines_data = json.load(f)

# Create a list of business ideas and their corresponding categories
data = []
for category, ideas_dict in guidelines_data.items():
    for sub_category, ideas in ideas_dict.items():
        for idea in ideas:
            data.append((category, sub_category, idea))

# Prepare the text data for classification using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([idea for _, _, idea in data])
y = [category for category, _, _ in data]

# Create and train an SVM classifier
clf = SVC(kernel='linear')
clf.fit(X, y)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json.get("user_input", "")
    result = {}

    if user_input in guidelines_data:
        result["category"] = user_input
        result["data"] = guidelines_data[user_input]
    elif user_input:
        X_test = vectorizer.transform([user_input])
        predicted_category = clf.predict(X_test)[0]
        if predicted_category in guidelines_data:
            result["category"] = predicted_category
            result["data"] = guidelines_data[predicted_category]
        else:
            result["error"] = f"No guidelines found for the predicted category '{predicted_category}'."

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
