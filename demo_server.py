from __future__ import unicode_literals, print_function, division

from flask import Flask, render_template, request, url_for
from flask_cors import CORS, cross_origin
import torch
import torch.nn as nn
import numpy as np

from models import EncoderRNN, DecoderRNN, AttnDecoderRNN
from train import evaluate, normalizeString, prepareData

def sech(x):
    return 1./np.cosh(x)

class ImdbAutoEncoder(object):

    def __init__(self, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang

    def autoencoder(self, sentence):
        encoder = torch.load('test_encoder_imdb10000_0.0037651115254803288', map_location={'cuda:0': 'cpu'})
        decoder = torch.load('test_decoder_imdb10000_0.0037651115254803288', map_location={'cuda:0': 'cpu'})

        output_words, _, loss = evaluate(
            encoder, decoder, sentence, self.input_lang, self.output_lang)

        return output_words, loss


# Initialize the Flask application
print(" - Starting up application")

app = Flask(__name__)
CORS(app)


# Define a route for the default URL, which loads the form
@app.route('/', methods=['POST'])
def predict_with_ajax():
    """Returns prediction as string format"""
    total_anomaly_prob = 0
    review_texts = request.form['review_text']
    for review_text in review_texts.split('.'):
        review_text += '.'
        review_text = normalizeString(review_text)
        review_text = ' '.join(review_text.split()[:14])
        decoded_text, decoded_loss = imdb_autoencoder.autoencoder(review_text)
        print('Original text sequence: ', review_text.split())
        print('Decoded text sequence: ', decoded_text)
        anomaly_prob = sech(decoded_loss / len(review_text.split()))
        print(anomaly_prob)
        print(type(anomaly_prob))
        total_anomaly_prob += anomaly_prob[0]
    return str(total_anomaly_prob/len(review_texts.split('.')))



@app.route('/', methods=['GET'])
def main():
    return render_template('demo.html')

# Run the app :)
if __name__ == '__main__':
    input_lang, output_lang, pairs = prepareData('eng', 'eng', False)
    imdb_autoencoder = ImdbAutoEncoder(input_lang, output_lang)
    criterion = nn.NLLLoss()

    app.run(
        host="0.0.0.0",
        port=int("8888")
    )

