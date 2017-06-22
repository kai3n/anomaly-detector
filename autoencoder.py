
import numpy as np

from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM, RepeatVector, Dense, Dropout
from keras.models import Model

from gensim.models import Word2Vec

def refine(text):
    """Remove impurities from the text"""
    import re
    text = re.sub(r"<br />", "", text)
    text = re.sub(r"[^A-Za-z0-9!?\'\`]", " ", text)
    text = re.sub(r"it's", " it is", text)
    text = re.sub(r"that's", " that is", text)
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", " will not", text)
    text = re.sub(r"don't", " do not", text)
    text = re.sub(r"can't", " can not", text)
    text = re.sub(r"cannot", " can not", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.lower()

MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 10
HIDDEN_DIM = 10

def main():


    # all_sentences must receive a list of senteces/strings that will be used on training
    import re

    # text = """a congressional staffer and members of the Capitol police force were shot Wednesday in Alexandria, Virginia, during Republicans' early-morning practice ahead of a charity baseball game. President Donald Trump said the alleged gunman had been killed. Federal law enforcement officials identified the alleged shooter as James Hodgkinson, 66, of Belleville, Illinois. At least five people including Scalise, the third ranking member of House Republican leadership as the majority whip, were hospitalized. Scalise was in critical condition after surgery, according to So Young Pak, spokeswoman for MedStar Washington Hospital Center. Scalise is out of his first surgery, according to a Scalise aide. It is not clear if he will have a second surgery. His wife Jennifer and their two young children are traveling up from New Orleans to Washington now to be with him. A congressional staffer, Zach Barth, was also injured. Matt Mika, a lobbyist for Tyson Foods who sometimes practices with the team, was also identified as one of the victims. As of Wednesday afternoon, Mika was in surgery and in critical condition, according to a statement from his family. Who is Steve Scalise? Who is Steve Scalise? House Speaker Paul Ryan also identified two members of the Capitol Police who were injured, Crystal Griner and David Bailey. In a statement, Capitol Police said Griner was in "good condition in the hospital having been shot in the ankle," and that Bailey "was treated and released having sustained a minor injury during the incident." As of Wednesday afternoon, Mika was in surgery and in critical condition, according to a statement from his family. Lawmakers who were there described a chaotic scene, with many members of the congressional GOP baseball team huddled in a dugout while Capitol police who were part of Scalise's security detail and local Alexandria police engaged in a shoot-out with Hodgkinson. Congressional and law enforcement sources described it as a "deliberate attack." Traces are still being done on the two firearms recovered at the scene -- an SKS rifle 7.62 (which is a Chinese-made AK variant) and a .9 mm pistol, a law enforcement source tells CNN. Later, House Republican and Democratic leaders called for unity and praised the police, while Trump in an address from the White House said that the prayers of the nation and world are with Scalise. Trump calls for unity after GOP baseball shooting Trump calls for unity after GOP baseball shooting "Congressman Scalise is a friend, and a very good friend," Trump said. "He's a patriot. And he's a fighter. He will recover from this assault -- and Steve, I want you to know that you have the prayers not only of the entire city behind you, but of an entire nation, and frankly the entire world. America is praying for you and America is praying for all of the victims of this terrible shooting." Ryan condemned the shooting and praised the Capitol police on the House floor later Wednesday. "We are united. We are united in our shock. We are united in our anguish. An attack on one of us is an attack on all of us," Ryan said, drawing a bipartisan standing ovation. Members practicing for traditional baseball game Members of Congress were practicing for a game that was scheduled for Thursday night at Nationals Park. Rep. Martha McSally, R-Arizona, said the game will still go on as scheduled. The annual game has been played since 1909, and McSally said lawmakers applauded the announcement at an all-members meeting. RELATED: A short history of the Congressional Baseball Game Rep. Joe Barton, who manages the Republican team, said at a news conference that the game has added the Fallen Officers Fund as a charity it's sponsoring. Rep. Mike Doyle, who manages the Democratic team, said at the same conference that they would like to host the entire Republican team at the Democratic club for dinner to reflect Wednesday night. "Some of them have probably never set foot in that building," he joked. Lawmakers describe a terrifying scene Lawmakers who spoke at the scene to reporters described a normal morning practice, at a field where they've practiced for years, when suddenly shots rang out. Lawmakers, staff members and even the young son of one of the members ran for cover, jumping into dugouts and over fences to avoid the gunshots. Members described Scalise dragging himself roughly 15 yards away from second base, where he had been playing, and lying there until the shooter was neutralized, at which point some of them ran to assist him and apply pressure to the wound until he could be evacuated. Once they were able, Sen. Jeff Flake said he and Rep. Brad Wenstrup, who is a physician, went to where Scalise was lying to apply pressure to the wound. Scalise was coherent the whole time, Flake said. Kentucky Sen. Rand Paul told CNN "it would have been a massacre" without the Capitol Hill Police officers present. "Nobody would have survived without the Capitol Hill police," Paul said on CNN. "It would have been a massacre without them." What GOP lawmakers saw at congressional baseball attack What GOP lawmakers saw at congressional baseball attack "We had nothing but baseball bats to fight back against a rifle with," Alabama Rep. Mo Brooks said. Flake added that he saw a member of Scalise's security detail return fire on the gunman for what felt like 10 minutes, even though the police officer was wounded in the leg. RELATED: GOP pitcher chokes back tears as he reflects on teammates' camaraderie "Fifty (shots) would be an understatement, I'm quite sure," Flake said when asked about the total amount of gunfire, including police returning fire. Brooks said the shooter was behind the third base dugout and didn't say anything. "The gun was a semiautomatic," Brooks said, adding that he was sure it was a rifle but unsure what kind. "It continued to fire at different people. You can imagine, all the people on the field scatter." Gunman opens fire on GOP baseball practice 지도 데이터 ©2017 Google 이용약관 지도 오류 신고 2 km Shooter identified Hodgkinson's Facebook page is largely political, his profile picture is a caricature of Bernie Sanders as Uncle Sam. The Facebook feed is filled with anti-Trump sentiments such as "Trump is guilty and should go to prison for treason." He also "liked" a political cartoon that suggested Scalise should be fired. On March 22, he posted "Trump is a Traitor. Trump Has Destroyed Our Democracy. It's Time to Destroy Trump & Co." Congressmen say they spoke to suspect moments before he opened fire Congressmen say they spoke to suspect moments before he opened fire Vermont Sen. Bernie Sanders said both in a statement and on the Senate floor that he had learned the shooter volunteered on his presidential campaign -- and that he condemned the shooting "in the strongest possible terms." "I have just been informed that the alleged shooter at the Republican baseball practice is someone who apparently volunteered on my presidential campaign," Sanders said. "I am sickened by this despicable act. Let me be as clear as I can be. Violence of any kind is unacceptable in our society and I condemn this action in the strongest possible terms," he said. "Real change can only come about through nonviolent action, and anything else runs against our most deeply held American values." Capitol Hill increases security, cancels events The news of the shooting reverberated on Capitol Hill, where security was increased and regular proceedings were canceled or postponed. He also said he'd seen an image of members of the Democratic congressional baseball team huddled in prayer after learning of the shooting. "Every day we come here to test and to challenge each other. We feel so deeply about the things that we fight for and the things that we believe in. At times, our emotions can clearly get the best of us. We are all imperfect. But we do not shed our humanity when we enter this chamber. For all the noise and all the fury, we are one family," Ryan said. The House decided to not hold any votes on legislation Wednesday, and many hearings were delayed. The House Natural Resources Federal Lands Subcommittee also canceled a scheduled meeting to debate gun legislation. Capitol Hill Police notified congressional offices that the security presence on the Hill would be increased "out of an abundance of caution." Members of Congress targeted in the past by violence Scalise is the first member of Congress to be shot since former Arizona Rep. Gabby Giffords in January 2011. Giffords was shot in the head by Jared Lee Loughner at a "Congress On Your Corner" event at a Tucson grocery store. Giffords, who authorities said was the main target of the shooting, survived the attack but six others were killed and an additional 12 were injured. RELATED: Steve Scalise is the 9th sitting member of Congress to be shot Loughner pleaded guilty in 2012 and was sentenced to life in prison without the possibility of parole. This story has been updated and will continue to update as news develops. CNN's Ashley Killough, Evan Perez, Manu Raju, Phil Mattingly, Dana Bash, Deirdre Walsh, Eugene Scott, Peter Morris, Karl deVries and Noa Yadidi contributed to this report."""
    # text = re.sub(r"[^A-Za-z0-9!?\'\`.]", " ", text)
    # # all_sentences = ['list of sentences/string', 'list of sentences/string']
    # all_sentences = text.split('.')

    import os

    # read in the train data
    path_train = 'dataset/train/pos/'
    path_test = 'dataset/test/pos/'
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    all_sentences = []
    for path_train in ['dataset/train/pos/', 'dataset/train/neg/', 'dataset/test/pos/', 'dataset/test/neg/']:
        X_train.extend([open(path_train + f).read().split('.') for f in os.listdir(path_train) if f.endswith('.txt')])
        for i in X_train:
            for j in i:
                if len(refine(j)) > 0:
                    all_sentences.append(refine(j))

    content = ''
    for sentences in all_sentences:
        content += sentences + '\n'
    with open('imdb-eng-eng.txt', 'wt') as fo:
        fo.write(content[:-1])


    # tokenizer = Tokenizer(num_words=50) # nb_words=MAX_NB_WORDS
    # tokenizer.fit_on_texts(all_sentences)
    # sequences = tokenizer.texts_to_sequences(all_sentences)
    # #print(sequences)
    #
    # word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))
    # #print(word_index.items())
    #
    # x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # y_train = tokenizer.texts_to_matrix(all_sentences, mode='binary')
    #
    # print('Shape of data tensor:', x_train.shape)
    #
    # # Loading Word2Vec
    # # model = Word2Vec.load_word2vec_format('/home/edilson/GoogleNews-vectors-negative300.bin', binary=True)
    # #
    # # embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    # # for word, i in word_index.items():
    # #     if word in model:
    # #         embedding_matrix[i] = model[word]
    # #     else:
    # #         embedding_matrix[i] = np.random.rand(1, EMBEDDING_DIM)[0]
    #
    #
    # embedding_layer = Embedding(len(word_index) + 1,
    #                             EMBEDDING_DIM,
    #                             # weights=[embedding_matrix],
    #                             input_length=MAX_SEQUENCE_LENGTH
    #                             )
    #
    # inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    # embedded_sequences = embedding_layer(inputs)
    # encoded = LSTM(HIDDEN_DIM)(embedded_sequences)
    #
    # decoded = RepeatVector(MAX_SEQUENCE_LENGTH)(encoded)
    # decoded = LSTM(len(word_index))(decoded)
    #
    # #decoded = Dropout(0.5)(decoded)
    # decoded = Dense(y_train.shape[1], activation='softmax')(decoded)
    #
    # sequence_autoencoder = Model(inputs, decoded)
    #
    # encoder_s2s_attention = Model(inputs, encoded)
    #
    # sequence_autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # print(sequence_autoencoder.summary())
    #
    # sequence_autoencoder.fit(x_train, y_train,
    #                          epochs=2,
    #                          batch_size=32,
    #                          shuffle=True,
    #                          validation_data=(x_train, y_train))

if __name__ == '__main__':
    main()