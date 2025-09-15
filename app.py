from flask import Flask, request, render_template
import joblib


vector = joblib.load(open("vectorizer.pkl", 'rb'))
model = joblib.load(open("Finalized_model.pkl", 'rb'))

app= Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods= ['GET','POST'])
def prediction():
    if request.method == "POST":    
        news = str(request.form['news'])
        print(news)
        predict = model.predict(vector.transform([news]))
        print(predict)

        return render_template("prediction.html", prediction_text = "News headlines is -> {}".format(predict))
    else:
        return render_template("prediction.html")    


if __name__ == '__main__':
    app.run(debug=True)
