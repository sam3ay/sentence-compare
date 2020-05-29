import json
import logging
import scipy.spatial
from sentence_transformers import SentenceTransformer

#### Based on https://github.com/awslabs/amazon-sagemaker-examples/tree/training-scripts/pytorch-rnn-scripts

JSON_CONTENT_TYPE = "application/json"
logger = logging.getLogger(__name__)


def model_fn(model_dir: str) -> object:
    """Loads a pytorch model

    Args:
        model_dir (str): path to model directory
    
    Return:
        object: Pytorch Model Object
    """
    logger.info("Loading the model.")
    model = SentenceTransformer(model_dir)
    logger.info("Model loaded")
    return model


def input_fn(serialized_input: object, content_type: str = JSON_CONTENT_TYPE) -> dict:
    """Deserializes input data converting it to a dictionary

    Args:
        serialized_input (object): Json object
        content_type (str, optional): Object type. Defaults to JSON_CONTENT_TYPE.

    Raises:
        Exception: If content type is not JSON

    Return:
        dict: deserialized object
    """
    logger.info(f"Deserializing the input data. {serialized_input}")
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input)
        return input_data
    raise Exception(
        "Requested unsupported ContentType in content_type: " + content_type
    )


def output_fn(prediction_output: dict, accept: str = JSON_CONTENT_TYPE) -> object:
    """serialize dictionary to json

    Args:
        prediction_output (dict): inference from model
        accept (str, optional): Accepted as an output. Defaults to JSON_CONTENT_TYPE.

    Raises:
        Exception: If content type is not JSON

    Returns:
        object: JSON 
    """
    logger.info(f"Serializing output. {prediction_output}")
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception("Requested unsupported ContentType in Accept: " + accept)


def predict_fn(input_data: dict, model: object) -> dict:
    """compares a query sentence to other sentences 
    and returns a dictionary with the the most similar sentence to the query

    Args:
        input_data (dict): sentence and corpus keys with list as values
        model (object): pytorch model used to generate sentence embeddings

    Returns:
        dict: dictionary of prediction and similarity score 
            with values true and a sentence if similarity score > 85%
            or false and a sentence
    """
    logger.info(f"Generating String embeddings. {input_data}")
    corpus = input_data["corpus"]
    corpus_embeddings = model.encode(corpus)
    query = input_data["query"]
    query_embedding = model.encode(query)
    logger.info("Determining similarity score.")
    # determine cosine distance
    distances = scipy.spatial.distance.cdist(
        [query_embedding[0]], corpus_embeddings, "cosine"
    )[0]
    # map distances to index of corpus, sentences to distance
    results = zip(range(len(distances)), distances)
    # ascending sort
    results = sorted(results, key=lambda x: x[1])

    output_dict = {
        "predict": int(1 - results[0][1] > 0.80),
        "sentence": corpus[results[0][0]],
    }
    logger.info(f"Prediction completed, similarity score {results[0][1]}")
    return output_dict
