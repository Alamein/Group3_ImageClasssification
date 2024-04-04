from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import torch
from torchvision.models import mobilenet_v2
from torchvision.transforms import transforms
import torch.nn as nn


app = Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024


model_path = 'train_model.pth'
class_labels = [
    'Abacha and Ugba (African salad)',
    'Akara and Eko',
    'Amala and Gbegiri-Ewedu',
    'Asaro',
    'Boli (Bole)',
    'Chin Chin',
    'Egusi Soup',
    'Ewa-Agoyin',
    'Fried Plantains (Dodo)',
    'Jollof Rice',
    'Meat-pie',
    'Moin-Moin',
    'Nkwobi',
    'Okro Soup',
    'Pepper Soup',
    'Puff-Puff',
    'Suya',
    'Vegetable Soup'
]


loaded_model = mobilenet_v2(pretrained=False) 
num_features = loaded_model.classifier[1].in_features
loaded_model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(num_features, len(class_labels))
)

loaded_model.load_state_dict(torch.load(model_path))


loaded_model.eval()


transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_image(image_path, model):

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0) 

    # Perform inference
    with torch.no_grad():
        output = model(image)
        scores = torch.softmax(output, dim=1) 
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class, scores.squeeze().tolist()

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = 'static/images/' + filename
        file.save(file_path)

        
        predicted_class, scores = predict_image(file_path, loaded_model)
        product = class_labels[predicted_class]

       
        class_scores = {class_labels[i]: score for i, score in enumerate(scores)}

        return render_template('prediction.html', product=product, predicted_class=predicted_class, class_scores=class_scores, user_image=file_path)

if __name__ == "__main__":
    app.run(debug=True)
