import os
import json
import time
import subprocess
from flask import Flask, request, jsonify, render_template
from shutil import copyfile  # ✅ THIS LINE
from datetime import datetime
import nbformat


app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebook')
NOTEBOOK_PATH = os.path.join(NOTEBOOK_DIR, 'hello_world.ipynb')
METADATA_PATH = os.path.join(NOTEBOOK_DIR, 'kernel-metadata.json')
KAGGLE_JSON_PATH = os.path.join(BASE_DIR, 'kaggle.json')

# Load Kaggle credentials once when app starts
os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')


@app.route("/")
def index():
    return render_template("index.html")

from flask import request  # Add this import at the top

@app.route("/train-line", methods=["POST"])
def train_line():
    kernel_slug = "yamansanghavi/hello-world-trigger"
    output_dir = os.path.join(BASE_DIR, "output")

    data = request.get_json()
    m = float(data.get("m", 1))
    b = float(data.get("b", 0))

    # ✅ Load and inject into notebook template
    import nbformat

    TEMPLATE_PATH = os.path.join(BASE_DIR, "notebook", "template_notebook.ipynb")
    NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebook", "hello_world.ipynb")

    with open(TEMPLATE_PATH) as f:
        notebook_json = nbformat.read(f, as_version=4)

    notebook_json.cells[0].source = f"m = {m}\nb = {b}"
    notebook_json.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
    }


    with open(NOTEBOOK_PATH, "w") as f:
        nbformat.write(notebook_json, f)

    try:
        import shutil
        import time

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print("🚀 Pushing notebook to Kaggle...")
        push_result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", NOTEBOOK_DIR],
            capture_output=True, text=True
        )
        print("STDOUT:", push_result.stdout)
        print("STDERR:", push_result.stderr)

        # 🕒 Wait until execution completes
        for i in range(10):
            status_check = subprocess.run(["kaggle", "kernels", "status", kernel_slug],
                                          capture_output=True, text=True)
            status_output = status_check.stdout
            if "COMPLETE" in status_output:
                print("✅ Kaggle notebook finished execution.")
                break
            print(f"⏳ Waiting for Kaggle... {i+1}/10")
            time.sleep(6)

        print("⬇️ Pulling notebook output from Kaggle...")
        for i in range(8):
            subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir],
                           capture_output=True, text=True)
            files = os.listdir(output_dir)
            print(f"⏳ Attempt {i+1}: Output files = {files}")
            if "plot.png" in files:
                print("✅ plot.png FOUND in output")
                break
            time.sleep(5)
        else:
            print("❌ plot.png NOT FOUND after 8 attempts")

        result = {"status": "Training complete ✅"}

        log_path = os.path.join(BASE_DIR, "output", "output.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                result["text"] = f.read()

        plot_path = os.path.join(BASE_DIR, "output", "plot.png")
        static_path = os.path.join(BASE_DIR, "static")

        if os.path.exists(plot_path):
            print("✅ Copying plot.png to static...")
            os.makedirs(static_path, exist_ok=True)
            static_plot_path = os.path.join(static_path, "plot.png")
            if os.path.exists(static_plot_path):
                os.remove(static_plot_path)
            copyfile(plot_path, static_plot_path)
            result["plot_url"] = "/static/plot.png"
            print("✅ plot.png saved")

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)})



@app.route("/check-status", methods=["GET"])
def check_status():
    kernel_slug = "yamansanghavi/hello-world-trigger"

    result = subprocess.run([
        "kaggle", "kernels", "status", kernel_slug
    ], capture_output=True, text=True)

    if result.returncode == 0:
        return jsonify({
            "status": "Status fetched successfully ✅",
            "output": result.stdout
        })
    else:
        return jsonify({
            "status": "Failed to get status ❌",
            "error": result.stderr
        })




@app.route("/get-output", methods=["GET"])
def get_output():
    kernel_slug = "yamansanghavi/hello-world-trigger"  
    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run([
        "kaggle", "kernels", "output", kernel_slug, "-p", output_dir
    ], capture_output=True, text=True)

    result = {"status": "Success ✅"}

    # Load text output
    log_path = os.path.join(output_dir, "output.txt")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            result["text"] = f.read()

    # Send image URL reference
    plot_path = os.path.join(output_dir, "plot.png")
    if os.path.exists(plot_path):
        result["plot_url"] = "/static/plot.png"
        # Copy plot to static directory for frontend
        from shutil import copyfile
        static_dir = os.path.join(BASE_DIR, "static")
        os.makedirs(static_dir, exist_ok=True)
        copyfile(plot_path, os.path.join(static_dir, "plot.png"))
    else:
        result["plot_url"] = None

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
