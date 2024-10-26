def build_vocab(dataframe, threshold):
    """Build a simple vocabulary wrapper."""
    counter = {}
    for i, row in dataframe.iterrows():
        caption = row['caption']
        for word in caption.split(' '):
            if word not in counter:
                counter[word] = 0
            counter[word] += 1

    words = [word for word in counter if counter[word] >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab