import json
import sys

from flask import Flask, render_template, request

sys.path.append("./")
sys.path.append("../../")

from src.config import *

N = 100
SEED = 2

app = Flask(__name__)


@app.route("/")
def index():
    with open(MTURK_PATH + f"{N}_{SEED}_sample.json", "r") as f:
        images_array = json.load(f)
    images_array = images_array
    return render_template("index.html", images_array=images_array)


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    name = data["name"]
    labels = data["labels"]

    with open(MTURK_PATH + "annotated/" + f"{N}_{SEED}_sample_{name}.json", "w") as f:
        json.dump(labels, f)
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
