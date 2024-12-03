import cv2
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

model_sinhala = tf.keras.models.load_model('model/sinhala_handwritten_cnn.h5')
# save the LB object to disk
filename = 'model/LB.sav'
# load the LB object from disk
loaded_LB = pickle.load(open(filename, 'rb'))

sinhala_classes=["අ","ආ","ඇ","ඈ","ඉ","ඊ","උ","එ","ඒ","ඔ","ඕ",
             "ක","කා","කැ","කෑ","කි","කී","කු","කූ","ක්","කෝ","ක්‍ර","ක්‍රි","ක්‍රී",
             "ග","ගා","ගැ","ගෑ","ගි","ගී","ගු","ගූ","ග්","ගෝ","ග්‍ර","ග්‍රි","ග්‍රී",
             "ච","චා","චැ","චෑ","චි","චී","චු","චූ","ච්","චෝ","ච්‍ර","ච්‍ර්","ච්‍රී",
             "ජ","ජා","ජැ","ජෑ","ජි","ජී","ජු","ජූ","ජ්","ජෝ","ජ්‍ර","ජ්‍රි","ජ්‍රී",
             "ට","ටා","ටැ","ටෑ","ටි","ටී","ටු","ටූ","ට්","ටෝ","ට්‍ර","ට්‍ර්","ට්‍රි"
             ,"ඩ","ඩා","ඩැ","ඩෑ","ඩි","ඩී","ඩු","ඩූ","ඩ්","ඩෝ","ඩ්‍ර","ඩ්‍ර්","ඩ්‍රි",
             "ණ","ණා","ණි",
             "ත","තා","ති","තී","තු","තූ","ත්","තෝ","ත්‍ර","ත්‍රා","ත්‍රි","ත්‍රී",
             "ද ","දා","දැ","දෑ","දි","දී","දු","දූ","ද්","දෝ","ද්‍ර","ද්‍රෝ","ද්‍රා","ද්‍රි","ද්‍රී",
             "න","නා","නැ","නෑ","නි","නී","නු","නූ","න්","නෝ","න්‍ර","න්‍රා","න්‍රි","න්‍රී",
             "ප","පා","පැ","පෑ","පි","පී","පු","පූ","ප්","ප්‍රෝ","පෝ","ප්‍ර","ප්‍රා","ප්‍රි","ප්‍රී",
             "බ","බා","බැ","බෑ","බි","බී","බු","බූ","බ්","බ්‍රෝ","බ්‍ර","බ්‍රා","බ්‍රි","බ්‍රී","බ්‍රෝ",
             "ම","මා","මැ","මෑ","මි","මී","මු","මූ","ම්","මෝ","ම්‍ර","ම්‍රා","ම්‍රි","ම්‍රී","ම්‍රෝ",
             "ය","යා","යැ","යෑ","යි","යී","යු","යූ","ෝ","ය්","hda",
             "ර","රා","රැ","රැ","රු","රූ","රි","රී",
             "ල","ලා","ලැ","ලෑ","ලි","ලී","ලු","ලූ","ල්",",da",
             "ව","වා","වැ","වෑ","වි","වී","වු","වූ","ව්","jda","ව්‍ර","ව්‍රා","ව්‍රැ","ව්‍රෑ","j%da",
             "ශ","ශා","ශැ","ශෑ","ශි","ශී","ශු","ශූ","ශ්","Yda","ශ්‍ර","ශ්‍රා","ශ්‍රැ","ශ්‍රෑ","ශ්‍රි","ශ්‍රී","Y%da",
             "ෂ","ෂා","ෂැ","ෂෑ","ෂි","ෂී","ෂු","ෂූ","ෂ්","Ida",
             "ස","සා","සැ","සෑ","සි","සී","සු","සූ","ida","ස්‍ර","ස්‍රා","ස්‍රි","ස්‍රී","ස්",
             "හ","හා","හැ","හෑ","හි","හී","හු","හූ","හ්","yda",
             "ළ","ළා","ළැ","ළෑ","ළි","ළී",
             "ළූ","ළූ",
             "ෆ","ෆා","ෆැ","ෆෑ","ෆි","ෆී","ෆූ","ෆූ","ෆ්‍ර","ෆ්‍රි","ෆ්‍රී","ෆ්‍රැ","ෆ්‍රෑ","ෆ්","*da",
             "ක්‍රා","ක්‍රැ","ක්‍රෑ","l%da",".%da",
             "ඛ","ඛා","ඛි","ඛී","ඛ්",
             "ඝ","ඝා","ඝැ","ඝෑ","ඝි","ඝී","ඝු","ඝූ",">da","ඝ්","ඝ්‍ර","ඝ්‍රා","ඝ්‍රි","ඝ්‍රී",
             "ඳ","ඳා","ඳැ","ෑ","ඳෑ","ඳි","ඳී","ඳු","ඳූ","|da "," ඳ්",
             "ඟ","ඟා","ඟැ"," ඟෑ"," ඟි","ඟී"," ඟු"," ඟූ","Õda","ඟ්",
             "ඬ","ැ","ඬා"," ඬැ", "ඬෑ"," ඬි","ඬී"," ඬු","ඬූ","ඬda "," ඬ්",
             "ඹ","ඹා"," ඹැ"," ඹෑ"," ඹි","ඹී"," ඹු","ඹූ","Uda","ඹ්",
             "භ","භා","භැ","භෑ","භි","භී","භු","භූ","Nda","භ්",
             "ධ","ධා","ධැ","ධෑ",",ධි",",ධී",",ධු",",ධූ","ධෝ","ධ්",
            "ඨ","ඨා","ඨැ","ඨි","ඨී","ඨු","ඨූ","ඨ්","ඪ","ඪා","ඪි","Vda",
             "ඵ","ඵා","ඵු","ඵි","Mda","ඵ් ","ථ","ථා","ථැ","ථ්","ා","ෟ","ණැ","ණෑ","ෘ","ණී","ණු","ණූ",
            "Kda","ණ්","ඥ","ඥා","{da","ඤ","ඤා","ඤු","[da","ඤ්","ඣ","ඣා","ඣු","COda",
             "ඣ්","ඦ","ඦා","ඦැ","ඦෑ","ඦි","ඦු","ඦූ","ඦෝ",
             "ඦ්","ඡ","ඡා","ඡැ","ඡෑ","ඡි","ඡේ","තැ","තෑ","ත්‍රැ","ත්‍රෑ",";%da",
             "ළු","ෲ","HQ","ff","f","H","Hq"
            
]

def predict_letter(img):
    # Remove alpha channel if it exists
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGB')

    # Resize the image with anti-aliasing to minimize quality loss
    resized_image = img.resize((32, 32), Image.LANCZOS)

    resized_image.save('data/sinhala_resized.png')  # Save resized and adjusted image

    # Convert the image to grayscale with OpenCV
    img_gray = cv2.imread('data/sinhala_resized.png', cv2.IMREAD_GRAYSCALE)

    # Convert the final processed image to a numpy array and scale it
    img_array = np.array(img_gray) / 255.0

    # Add a dimension to transform the array into a "batch" of size (1, 32, 32, 1)
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = np.expand_dims(img_batch, axis=-1)

    # Get predictions
    predictions = model_sinhala.predict(img_batch)

    # Get the indices of the top 40 probabilities
    top40_indices = np.argpartition(predictions[0], -40)[-40:]

    # Sort the indices by their corresponding probabilities
    top40_indices = top40_indices[np.argsort(predictions[0][top40_indices])][::-1]

    # Get the corresponding class names
    top40_classes = loaded_LB.classes_[top40_indices]

    # Display the predicted class
    predicted_letter = sinhala_classes[int(top40_classes[0]) - 1]
    return predicted_letter, predictions[0][top40_indices[0]]

# Folder path containing the images
folder_path = 'test_images'
# Supported image extensions
valid_extensions = ('.jpg', '.jpeg', '.png')
# Register the font with Matplotlib
prop = fm.FontProperties(fname="font/iskpota.ttf")
# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(valid_extensions):  # Check if file has a valid image extension
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        predicted_class, confidence = predict_letter(img)
        print(f"File Name: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence}")
        
        # Open and display the image
        with Image.open(image_path) as img:
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"File Name: {filename}, Predicted Class: {predicted_class}, Confidence: {confidence}", fontproperties=prop, fontsize=14)
            plt.show()