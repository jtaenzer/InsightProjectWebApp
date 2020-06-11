from joblib import load
from sklearn.feature_extraction.text import TfidfTransformer
from flask import Flask, render_template, request

# Create the application object
app = Flask(__name__)

save_path = "./binaries/"
countvec = load(save_path + "count_vectorizer.joblib")
title_encoding = load(save_path + "title_encoding.joblib")
vocab_for_counts = load(save_path + "vocab_for_counts.joblib")
scaler = load(save_path + "scaler.joblib")
model = load(save_path + "model.joblib")

@app.route('/',methods=["GET","POST"]) #we are now using these methods to get user input
def home_page():
    return render_template('index.html')  # render a template


@app.route('/output')
def recommendation_output():
    # Pull input
    user_input = request.args.get('user_input')

    # Case if empty
    if user_input == "":
        return render_template("index.html",my_input=user_input,my_form_result="Empty")
    else:
        skills_list = user_input.split(", ")
        count_mat = countvec.transform([" ".join(skills_list)])
        tfidf = TfidfTransformer()
        tfidf_mat = tfidf.fit_transform(count_mat)
        test_data = scaler.transform(tfidf_mat.toarray())
        pred_job_title = title_encoding[int(model.predict(test_data))]
        return render_template("index.html", my_input=user_input, my_output=pred_job_title, my_form_result="NotEmpty")

# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True)  # will run locally http://127.0.0.1:5000/
