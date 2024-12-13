def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels =[]
    tokenized_sentence.append("[CLS]")
    labels.append("O")

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label]*n_subwords)
    
    tokenized_sentence.append("[SEP]")
    labels.append("O")
    return tokenized_sentence, labels