import numpy as np

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


def get_prediction_emotion(xgb_model, input_name, embedding, label_encoder):
    """
    This function take input as name and return its predicted emotion
    """
    processed_name = sentence_to_vector(input_name, embedding)
    processed_name = np.expand_dims(processed_name, axis=0)

    y_pred_label = xgb_model.predict(processed_name)  # Inference

    # convert from label (187) to emotion unicode -> (U+1F913)
    y_pred_unicode = label_encoder.inverse_transform(y_pred_label)[0]
    y_pred_emotion = convert_unicode_2_emoji(y_pred_unicode)

    return y_pred_emotion