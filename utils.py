import numpy as np
from googletrans import Translator
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # Ensure consistent results by setting the seed

def convert_unicode_2_emoji(unicode_str):
    """
    This function take unicode string and return the emoji
    """
    emoji = chr(int(unicode_str[2:], 16))
    return emoji

def convert_emoji_2_unicode_str(emoji):
    unicode_code_point = f'U+{ord(emoji):X}'
    return unicode_code_point



def load_glove_embeddings(file_path):
    """Load GloVe embeddings from a file."""
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings


def sentence_to_vector(sentence, embeddings):
    """
    This function convert a sentence to vector using GloVe embeddings.
    We convert each word into embedding then calculate average of every word in a sentence.
    """
    vectors = []
    for word in sentence.split():
        vector = embeddings.get(word)
        if vector is not None:
            vectors.append(vector)
    
    if len(vectors) == 0:
        return np.zeros(50)  # Return a zero vector if no words are found
    
    return np.mean(vectors, axis=0)


def get_top_k_prediction(xgb_model, input_name, top_k, embedding, label_encoder):
    """
    This function take input as text and return its TOP-K predicted emotion
    """    

    # Add batch dimension for input text
    processed_name = sentence_to_vector(input_name, embedding)
    processed_name = np.expand_dims(processed_name, axis=0)

    # Get top_k predicted probability
    y_pred_proba = xgb_model.predict_proba(processed_name)[0]
    top_k_y_pred_label = np.argsort(y_pred_proba)[-top_k:][::-1]  # Just a trick to reverse an array

    # Loop through all top_k predicted 
    list_predcited_emotion = []
    for y_pred_label in top_k_y_pred_label:
        y_pred_unicode = label_encoder.inverse_transform([y_pred_label])[0]
        y_pred_emotion = convert_unicode_2_emoji(y_pred_unicode)
        list_predcited_emotion.append(y_pred_emotion)
    
    return list_predcited_emotion



def identify_language(text):
    """
    This function take input text and return the laguage
    """
    try:
        language = detect(text)
        return language
    except:
        return 'en'  # default language is English
    


def vietnamese_to_english(vietnamese_text):
    """
    This function take input as vietnamese text and return the translated english
    """

    translator = Translator()
    translation = translator.translate(vietnamese_text, src='vi', dest='en')
    return translation.text

