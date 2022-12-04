import sys

import pandas as pd

sys.path.append("./")
sys.path.append("../../")

from src.config import *
from src.taxonomy.taxonomy import Taxonomy


def generate_column(root, name, level):
    """
    Generate HTML code for the column
    """
    return generate_label(
        list(filter(lambda x: x.name == name, root.children))[0], level
    )[:-1]


def generate_label(label, level):
    """
    Generate HTML code for the label
    """
    emoji_mapping = {
        "STEM": " 🌿 🦠 🔬 🧬 🚀 📡 💊",
        "Society": " 👥 🏅 🏟 ⚖️ 🚊",
        "Culture": " 🏺 🎭 🎼 🍽️",
        "Places": " 🏛️ 🏘️ 🇺🇳 🗺️",
    }

    name = label.name.replace(" & ", "_").replace(" ", "_")
    if not label.children:
        return_string = (
            "\t" * level
            + f'<input type="checkbox" id="{name}" name="{name}" value="{name}"> <label for="{name}"> {label.name}</label><br>\n'
        )
    else:
        return_string = (
            "\t" * level
            + '<fieldset class="with_margin">\n'
            + "\t" * (level + 1)
            + f'<legend><b>{label.name}</b>{emoji_mapping.get(label.name, "")}</legend>\n'
        )

        for child in label.children:
            return_string += generate_label(child, level + 1)
        return_string += "\t" * level + "</fieldset>\n"

    return return_string


if __name__ == "__main__":
    app_name = "app.html"
    taxonomy = Taxonomy(hierarchical=True)
    taxonomy.set_taxonomy(TAXONOMY_VERSION)
    root = taxonomy.taxonomy

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Image Annotation task</title>
</head>
<body>

<script>

    let data = {{{{images_array|safe}}}}

    let all_labels = [];

</script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js"></script>
<style>
    body {{
        font-family: "Helvetica Neue", Helvetica, sans-serif;
    }}

    .with_margin {{
        margin: 8px;
    }}

    .big_button {{
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        border: black solid;
    }}
    .big_button:hover{{
        color:#000000;
        background-color:#FFFFFF;
    }}
</style>

<form id="main_labels">
<table>

    <tr>
        <td style="vertical-align:top">
            <img src="" width="350" class="with_margin" id="main_image"><br><br>
            <div style="display:inline-block; padding: 8px; border: black solid">
                <p id="counter">0 / 0</p>
            </div>
        </td>
        
        
        <td style="vertical-align:top">
{generate_column(root, "STEM", 3)}<br>
        </td>

        <td style="vertical-align:top">
{generate_column(root, "Society", 3)}<br>
{generate_column(root, "Places", 3)}
        </td>

        <td style="vertical-align:top">
{generate_column(root, "Culture", 3)}<br>
            <fieldset class="with_margin">
                <legend><b>Other</b> </legend>
                <input type="checkbox" id="Logos" name="Logos" value="Logos"> <label for="Logos"> Logos</label><br>
                <input type="checkbox" id="Other" name="Other" value="Other"> <label for="Other"> Other</label><br>
                <input type="text" id="other_text" name="other_text"><br><br>
            </fieldset><br>
        </td>

        <td style="vertical-align:top">
            <input type="button" value="Next" id="next_button" class="big_button"/>
            <input type="text" placeholder="YOUR NAME" id="name_button" style="display: none"/><br><br>
            <input type="button" value="Submit" id="submit_button" style="display: none" class="big_button"/>
        </td>
    </tr>


</table>


</form>

<div id="log"></div>


<script>
    let current_index = -1;

    function push_labels() {{
        let labels = [];
        $.each($('#main_labels input:checkbox').serializeArray(), function (i, field) {{
            labels.push(field.value);
        }});
        $("input:checkbox").prop('checked', "");

        let other_text = $("#other_text").val();
        $("#other_text").val("");

        let item_labels = {{"id": data[current_index].id, "url": data[current_index].url, "labels": labels, "other_text": other_text}};

        all_labels.push(item_labels);
    }}

    function next_image() {{
        $("#main_image").attr("src", "https://upload.wikimedia.org/wikipedia/commons/b/b1/Loading_icon.gif")

        if (current_index>=0) {{
            push_labels();
        }}

        $("#main_image").attr("src", data[++current_index].url)
        $("#counter").text(current_index + 1 + "/" + data.length);

        if (current_index>=data.length-1) {{
            $("#next_button").hide()
            $("#submit_button").show();
            $("#name_button").show();
        }}
    }}

    $(function(){{
        next_image();
    }});

    $("#next_button").click(next_image);

    $("#submit_button").click(function() {{
        push_labels();
        let name = $("#name_button").val();

        $.ajax({{
          type: "POST",
          url: "/save",
          data: JSON.stringify({{"name": name, "labels": all_labels}}),
          success: function () {{
              alert("Thank you")
              $("#submit_button").hide();
          }},
          contentType : 'application/json',
        }});
    }});

</script>


</body>
</html>"""

    with open("templates/index.html", "w", encoding="utf-8") as file:
        file.write(doc)

    print("Done.")
