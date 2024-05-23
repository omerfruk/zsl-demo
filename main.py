from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Modeli ve işlemciyi yükleme
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Görüntü URL'si örneği
image_path = "bird.jpg"
image = Image.open(image_path)

# Metin açıklaması
text = ["a photo of a dog", "a photo of a cat", "a photo of a bird", "a photo of a horse", "a photo of a car", "a photo of a tree", "a photo of a flower", "a photo of a building", "a photo of a person", "a photo of a landscape"]  # İlgili metin açıklamaları

# Görüntüyü işleme ve modelden öznitelik çıkarımı
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Çıktıları analiz etme ve gösterme
# Model çıktısını alın (varsayılan olarak logit olarak döner, softmax ile olasılığa çevirin)
probs = outputs.logits_per_image.softmax(dim=-1).squeeze().tolist()

# En yüksek puan alan sınıfın indexini bulun
max_index = np.argmax(probs)

# En yüksek puan alan sınıfı yazdır
predicted_class = text[max_index]
print("Predicted class:", predicted_class)

# Resmi ve tahmini sınıfı görselleştir
plt.imshow(image)
plt.title(f"Predicted: {predicted_class}")
plt.show()