# Cliniscan: Pneumonia X-ray Classifier with Grad-CAM

End-to-end pipeline:
- Use Kaggle API (`kaggle.json`) to download datasets or competitions
- Train ResNet18 (Normal vs Pneumonia)
- Visualize predictions with Grad-CAM
- Streamlit app to upload an X-ray, get prediction and confidence, and view heatmap overlay
- Flask + Gemini chatbot with language translation (English, Telugu, Tamil, Hindi, Kannada, Malayalam)

## 1) Prerequisites
- Windows 10/11 with PowerShell
- Python 3.10 or 3.11 (64-bit)
- Kaggle account and API token (`kaggle.json`). Get it from your Kaggle account settings.
- Google Gemini API key (`GEMINI_API_KEY` env var)

## 2) Project Setup
```powershell
cd "C:\Users\VENKATA SAI\Desktop\cliniscan"
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Configure Kaggle API Key
Place `kaggle.json` either in the project root or in `%USERPROFILE%\.kaggle\kaggle.json`.

If you put it in the project root, the downloader will copy it to the correct location.

## 4) Download Dataset
Default pneumonia dataset:
```powershell
python src\data\download_kaggle.py --type dataset --ref paultimothymooney/chest-xray-pneumonia --out data\raw
```

Download competition (example: vinbigdata):
```powershell
python src\data\download_kaggle.py --type competition --ref vinbigdata-chest-xray-abnormalities-detection --out data\raw\vinbigdata
```

## 5) Train the Model
```powershell
python src\models\train_resnet18.py --data_dir data\raw\chest_xray --epochs 5 --batch_size 32 --lr 1e-4 --out_dir models
```
The script saves `models/resnet18_pneumonia.pt` and `models/class_index.json`.

## 6) Run the Streamlit App
```powershell
streamlit run app.py
```
Upload an X-ray (PNG/JPG). The app shows prediction (Normal/Pneumonia), confidence, and a Grad-CAM heatmap overlay.

## 7) Run the Flask Gemini Chatbot
Set your API key and start the app:
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
$env:FLASK_SECRET_KEY="change-this-secret"
python chatbot_app.py
```
Open `http://127.0.0.1:5000/` and choose a language (English, Telugu, Tamil, Hindi, Kannada, Malayalam). Messages are answered by Gemini and translated to the selected language.

## Notes
- If the Kaggle CLI is not found, install it via `pip install kaggle` (already included) and restart the shell.
- For Kaggle competitions, ensure you have joined the competition and accepted rules on Kaggle.
- GPU is optional but recommended if available (PyTorch wheels included in requirements).
- Do not hardcode API keys in code; use environment variables.
