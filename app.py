from flask import Flask, render_template, request, redirect
import pickle


app = Flask(__name__)

model = pickle.load(open("RF_Diabetes_Clf_.pkl", "rb"))


@app.route("/")
def hello_world():
    return render_template("Index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        pregnancies = request.form["Pregnancies"]
        glucose = request.form["Glucose"]
        bloodPressure = request.form["BloodPressure"]
        skinThickness = request.form["SkinThickness"]
        insulin = request.form["Insulin"]
        bMI = request.form["BMI"]
        diabetesPedigreeFunction = request.form["DiabetesPedigreeFunction"]
        age = request.form["Age"]

        prediction = model.predict(
            [
                [
                    pregnancies,
                    glucose,
                    bloodPressure,
                    skinThickness,
                    insulin,
                    bMI,
                    diabetesPedigreeFunction,
                    age,
                ]
            ]
        )

        return render_template("predict.html", result=prediction)


if __name__ == "__main__":
    app.run(debug=True)
