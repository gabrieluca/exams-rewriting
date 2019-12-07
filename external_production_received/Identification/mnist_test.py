from keras.models import model_from_json


def model():
    json_file = open('train data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("train data/model.h5")
    print("Loaded model from disk")
    return loaded_model
