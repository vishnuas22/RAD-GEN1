from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
from PIL import Image
from werkzeug.utils import secure_filename
import torch
from transformers import CLIPTokenizer
from sd import model_loader, pipeline


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Device configuration
DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.backends.mps.is_built() or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"

print(f"Using device: {DEVICE}")

# Model and tokenizer initialization
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model_file = "/Users/Dataghost/RAD_GEN1/data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    strength = float(request.form.get('strength', 0.9))
    cfg_scale = float(request.form.get('cfg_scale', 8))
    num_inference_steps = int(request.form.get('num_inference_steps', 50))
    seed = int(request.form.get('seed', 42))
    uncond_prompt = ""
    do_cfg = True
    sampler = "ddpm"
    input_image = None

    if 'input_image' in request.files:
        file = request.files['input_image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            input_image = Image.open(file_path)

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.png')
    Image.fromarray(output_image).save(output_image_path)

    return redirect(url_for('index'))

@app.route('/save_settings', methods=['POST'])
def save_settings():
    settings = {
        'strength': request.form.get('strength'),
        'cfg_scale': request.form.get('cfg_scale'),
        'num_inference_steps': request.form.get('num_inference_steps'),
        'seed': request.form.get('seed')
    }
    # Save settings to a file or database as needed
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
