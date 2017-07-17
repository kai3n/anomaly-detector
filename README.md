# Anomaly Detector project with imdb

The goal of this project is to analys your sentiment through the review typed on the page.
If the background turns into green, that means you have the positive review. Red is the opposite. This project acheived `91%` accuracy.

## Getting Started

Demo server is based on Flask. To run the server, execute demo_server.py
```python
python demo_server.py
```

Next, once you go to the root page of your server, you will get the simple page.
You can just type the review of what you watched recently for the test. Credibility will tell you your review is in the same domain related to movies or anomaly. 

| ![good](pos.gif "test1") | ![bad](neg.gif "test2") |
|:---:|:---:|
             

### Prerequisites

This open source is based on Python 3.5

```
pip install -r requirement.txt
```

IMDb Dataset: `http://ai.stanford.edu/~amaas/data/sentiment/`

GloVe pre-trained vector: `https://nlp.stanford.edu/projects/glove/`

### Training


```shell
$python train.py
```

### Training with loaded model
```shell
$python train.py encoder_model decoder_model
```


### Using pre-trained model

```python
class ImdbAutoEncoder(object):

    def __init__(self, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang

    def autoencoder(self, sentence):
        encoder = torch.load('trained_model/encoder_imdb100000_max16_glove_0.3367996503444526_2.0', map_location={'cuda:0': 'cpu'})
        decoder = torch.load('trained_model/decoder_imdb100000_max16_glove_0.3367996503444526_2.0', map_location={'cuda:0': 'cpu'})

        output_words, _, loss = evaluate(
            encoder, decoder, sentence, self.input_lang, self.output_lang)

        return output_words, loss
```

## Running the tests

Coming soon.


## Contributing

Welcome!


## Authors

* **James Pak**


## License

This project is licensed under Gridspace.

## Acknowledgments

* Word embedding
* LSTM
* Attention
* Auto Encoder

