import sys

import pandas as pd

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.taxonomy import Label, Taxonomy


def generate_column(root, df, name, level, i=""):
    """
    Generate HTML code for the column
    """
    return generate_label(
        list(filter(lambda x: x.name == name, root.children))[0], level, df, i
    )[:-1]


def generate_tooltip(df, name, level):
    """
    Generate HTML code for the tooltip
    """
    if name not in df:
        return ""

    return_str = ""
    return_str += (
        "\t" * level
        + '<div class="tooltip"><img src="https://upload.wikimedia.org/wikipedia/commons/a/ac/Aiga_information.png" class="question_mark">\n'
    )
    return_str += "\t" * (level + 1) + f'<span class="tooltiptext">{name}:<br>\n'

    for example in df[name]:
        return_str += (
            "\t" * (level + 2) + f'<img src="{example}" class="tooltip_image">\n'
        )

    return_str += "\t" * (level + 1) + "</span>\n"
    return_str += "\t" * level + "</div>\n"
    return return_str


def generate_label(label, level, df, i=""):
    """
    Generate HTML code for the label
    """
    emoji_mapping = {}

    name = label.name.replace(" & ", "_").replace(" ", "_")
    if not label.children:
        return_string = (
            "\t" * level
            + f'<crowd-checkbox id="{name}{i}" name="{name}{i}" value="{name}{i}"> {label.name.capitalize()}\n'
            + generate_tooltip(df, label.name, level + 1)
            + "\t" * level
            + "</crowd-checkbox><br>\n"
        )
    else:
        return_string = (
            "\t" * level
            + '<fieldset class="with_margin">\n'
            + "\t" * (level + 1)
            + f'<legend><b>{label.name}</b>{emoji_mapping.get(label.name, "")}</legend>\n'
        )

        for child in label.children:
            return_string += generate_label(child, level + 1, df, i)
        if label.name == "Places":
            return_string += generate_label(Label("places", []), level + 1, df, i)
        return_string += "\t" * level + "</fieldset>\n"

    return return_string


if __name__ == "__main__":
    taxonomy = Taxonomy(hierarchical=True)
    taxonomy.set_taxonomy(TAXONOMY_VERSION)
    root = taxonomy.taxonomy

    with open("templates/tooltip_examples.json", "r") as f:
        df = pd.read_json(f, typ="series")

    repetitions = 10

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Image Annotation task</title>
</head>
<body>


<!-- You must include this JavaScript file -->
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<!--<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-+0n0xVW2eSR5OomGNYDnhzAbDsOXxcvSN1TPprVMTNDbiYZCxYbOOl7+AMvyTG2x" crossorigin="anonymous">-->

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.1/dist/js/bootstrap.bundle.min.js" integrity="sha384-gtEjrD/SeCtmISkJkNUaaKMoLD0//ElJ19smozuHV6z3Iehds+3Ulb9Bn9Plx0x4" crossorigin="anonymous"></script>
<script src="https://code.jquery.com/jquery-3.6.0.min.js" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4=" crossorigin="anonymous"></script>

<style>
    body {{
        font-family: "Helvetica Neue", Helvetica, sans-serif;
    }}

.tooltip {{
  position: relative;
  display: inline-block;
}}

.question_mark {{
  width: 15px;
  opacity: 0.5;
  vertical-align: middle;
}}

.tooltip_image {{
  width: 130px;
}}

.tooltip .tooltiptext {{
  visibility: hidden;
  width: 120px;
  background-color: black;
  color: #fff;
  text-align: center;
  border-radius: 6px;
  padding: 5px 0;
  position: absolute;
  z-index: 1;
  top: 150%;
  left: 50%;
  margin-left: -60px;
}}

.tooltip .tooltiptext::after {{
  content: "";
  position: absolute;
  bottom: 100%;
  left: 50%;
  margin-left: -5px;
  border-width: 5px;
  border-style: solid;
  border-color: transparent transparent black transparent;
}}

.tooltip:hover .tooltiptext {{
  visibility: visible;
}}

.with_margin{{
    width: 220px;
}}

crowd-checkbox{{
    margin-bottom: 5px;
}}
</style>


<!-- For the full list of available Crowd HTML Elements and their input/output documentation,
      please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

<!-- You must include crowd-form so that your task submits answers to MTurk -->
<crowd-form answer-format="flatten-objects">

  <crowd-instructions link-text="View instructions" link-type="button">
    <short-summary>
      <p>Select all the topics that are relevant for the image</p>
    </short-summary>

  <!--  <detailed-instructions>-->
  <!--    <h3>Provide more detailed instructions here</h3>-->
  <!--    <p>Include additional information</p>-->
  <!--  </detailed-instructions>-->

  <!--  <positive-example>-->
  <!--    <p>Provide an example of a good answer here</p>-->
  <!--    <p>Explain why it's a good answer</p>-->
  <!--  </positive-example>-->

  <!--  <negative-example>-->
  <!--    <p>Provide an example of a bad answer here</p>-->
  <!--    <p>Explain why it's a bad answer</p>-->
  <!--  </negative-example>-->
  </crowd-instructions>

  <div class="options-container" style="width: 55%; float: left; text-align: center;">
    <div id="title"> <h3>Select all the topics that are relevant for the image</h3></div>"""

    for i in range(repetitions):
        doc += (
            f"""
    <table>
        <tr>
            <td style="vertical-align:top">
                <img src="${{url{i}}}" width="350" class="with_margin" id="main_image"><br><br>
            </td>

            <td style="vertical-align:top">
{generate_column(root, df, "STEM", 4, i)}<br>
            </td>

            <td style="vertical-align:top">
{generate_column(root, df, "Society", 4, i)}<br>
{generate_column(root, df, "Places", 4, i)}
            </td>

            <td style="vertical-align:top">
{generate_column(root, df, "Culture", 4, i)}<br>
                <fieldset class="with_margin">
                    <legend><b>Other</b> </legend>
{generate_label(Label("Logos & Symbols", []), 5, df, i)}"""
            + f"""                    <crowd-checkbox id="None{i}" name="None{i}" value="None{i}"> None of those</crowd-checkbox><br>
                </fieldset><br>
            </td>
        </tr>
    </table>
    <hr>

"""
        )

    doc += "</crowd-form>"

    with open("templates/index_mturk.html", "w", encoding="utf-8") as file:
        file.write(doc)

    print("Done.")
