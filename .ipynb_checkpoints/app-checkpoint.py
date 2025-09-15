from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

vector = TfidfVectorizer(stop_words='english', max_df=0.7)

model = joblib.load(open("Finalized_model.pkl", 'rb'))


app= Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods= ['GET','POST'])
def prediction():
    if request.method == "POST":    
        news = request.form['news']
        predict = model.predict(vector.transform[news])
        print(predict)

        return render_template("prediction.html")
    else:
        return render_template("prediction.html")    


if __name__ == '__main__':
    app.run(debug=True)
