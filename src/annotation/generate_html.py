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


<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js"></script>
<style>
    body {{
        font-family: "Helvetica Neue", Helvetica, sans-serif;
    }}

    .with_margin {{
        margin: 8px;
    }}
</style>

<form>
<table>

    <tr>
        <td style="vertical-align:top">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/2nd_Battalion%2C_503rd_Infantry_Regiment%2C_173rd_Airborne_Brigade_depart_Aviano_Air_Base%2C_Italy%2C_Feb._24%2C_2022.jpg/1920px-2nd_Battalion%2C_503rd_Infantry_Regiment%2C_173rd_Airborne_Brigade_depart_Aviano_Air_Base%2C_Italy%2C_Feb._24%2C_2022.jpg"
                 alt="Girl in a jacket" width="350" class="with_margin">
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
                <input type="checkbox" id="Other" name="Other" value="Other"> <label for="Other"> Other</label><br>
                <input type="text" id="other_text" name="other_text"><br><br>
            </fieldset><br>
        </td>
    </tr>


</table>

    <input type="button" value="Next" />

</form>


<script>
    // You can use the console window at the bottom
    var h1 = document.querySelector('h1');
    console.log(h1.textContent);
</script>


</body>
</html>
        """

    with open(MTURK_PATH + app_name, "w") as file:
        file.write(doc)

    print("Done.")
