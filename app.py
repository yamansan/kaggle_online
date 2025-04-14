import os
import json
import time
import subprocess
from flask import Flask, request, jsonify, render_template
from shutil import copyfile  # ✅ THIS LINE
from datetime import datetime

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

    # NN training code injected into notebook
    new_source = f"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

m = {m}
b = {b}

# Generate data
x = torch.linspace(-1, 1, 100).unsqueeze(1)
y = m * x + b + 0.1 * torch.randn_like(x)

# Model
model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x, y = x.to(device), y.to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(200):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

# Prediction
with torch.no_grad():
    y_pred = model(x).cpu().numpy()
    x_cpu = x.cpu().numpy()
    y_cpu = y.cpu().numpy()

# Plot
plt.scatter(x_cpu, y_cpu, label='Noisy Data')
plt.plot(x_cpu, y_pred, color='green', label='NN Fit')
plt.plot(x_cpu, m * x_cpu + b, color='red', linestyle='--', label=f'True Line: y = {m}x + {b}')
plt.legend()
plt.title("NN Fit vs True Line")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
out_path = os.path.join(os.getcwd(), "plot.png")
plt.savefig(out_path)
print("✅ plot.png saved at:", out_path)

with open("output.txt", "w") as f:
    f.write(f"Final loss: {{loss.item()}}")
"""

    try:
        with open(NOTEBOOK_PATH, "r") as f:
            notebook_json = json.load(f)

        notebook_json['cells'][0]['source'] = [line + '\n' for line in new_source.strip().split('\n')]
        notebook_json["metadata"] = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        }

        with open(NOTEBOOK_PATH, "w") as f:
            json.dump(notebook_json, f)

        import shutil
        import time
        
        # Clear old output to prevent stale data
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Repeatedly try to pull Kaggle output and check for plot.png
        for i in range(8):
            subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir], capture_output=True, text=True)
            files = os.listdir(output_dir)
            print(f"⏳ Attempt {i+1}: Output files = {files}")
            if "plot.png" in files:
                print("✅ plot.png FOUND in output")
                break
            time.sleep(5)
        else:
            print("❌ plot.png NOT FOUND after 8 attempts")


        # Wait and recheck status until it's COMPLETE
        import time
        status = ""
        for i in range(10):  # Try for up to ~60 seconds
            status_check = subprocess.run(["kaggle", "kernels", "status", kernel_slug],
                                        capture_output=True, text=True)
            status_output = status_check.stdout
            if "COMPLETE" in status_output:
                print("✅ Kaggle notebook finished execution.")
                break
            print(f"⏳ Waiting for Kaggle... {i+1}/10")
            time.sleep(6)

        # Now fetch output
        print("⬇️ Pulling notebook output from Kaggle...")
        subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", output_dir], capture_output=True, text=True)

        result = {"status": "Training complete ✅"}

        log_path = os.path.join(BASE_DIR, "output", "output.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                result["text"] = f.read()

        plot_path = os.path.join(BASE_DIR, "output", "plot.png")
        static_path = os.path.join(BASE_DIR, "static")

        if os.path.exists(plot_path):
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
