from itertools import chain
import torch, math
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def read_json_file(file_path):

    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError in line: {line.strip()}")
                    print(f"Error message: {e}")
        return data
    except FileNotFoundError:
        print(f"The file at {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_distinct(candidate, num):
    def pad_sequence(
        sequence,
        n,
        pad_left=False,
        pad_right=False,
        left_pad_symbol=None,
        right_pad_symbol=None,
    ):
        """
        Returns a padded sequence of items before ngram extraction.

            >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
            ['<s>', 1, 2, 3, 4, 5, '</s>']
            >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
            ['<s>', 1, 2, 3, 4, 5]
            >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
            [1, 2, 3, 4, 5, '</s>']

        :param sequence: the source data to be padded
        :type sequence: sequence or iter
        :param n: the degree of the ngrams
        :type n: int
        :param pad_left: whether the ngrams should be left-padded
        :type pad_left: bool
        :param pad_right: whether the ngrams should be right-padded
        :type pad_right: bool
        :param left_pad_symbol: the symbol to use for left padding (default is None)
        :type left_pad_symbol: any
        :param right_pad_symbol: the symbol to use for right padding (default is None)
        :type right_pad_symbol: any
        :rtype: sequence or iter
        """
        sequence = iter(sequence)
        if pad_left:
            sequence = chain((left_pad_symbol,) * (n - 1), sequence)
        if pad_right:
            sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
        return sequence


    def ngrams(
        sequence,
        n,
        pad_left=False,
        pad_right=False,
        left_pad_symbol=None,
        right_pad_symbol=None,
    ):
        """
        Return the ngrams generated from a sequence of items, as an iterator.
        For example:

            >>> from nltk.util import ngrams
            >>> list(ngrams([1,2,3,4,5], 3))
            [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

        Wrap with list for a list version of this function.  Set pad_left
        or pad_right to true in order to get additional ngrams:

            >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
            >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
            >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
            [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
            >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
            [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


        :param sequence: the source data to be converted into ngrams
        :type sequence: sequence or iter
        :param n: the degree of the ngrams
        :type n: int
        :param pad_left: whether the ngrams should be left-padded
        :type pad_left: bool
        :param pad_right: whether the ngrams should be right-padded
        :type pad_right: bool
        :param left_pad_symbol: the symbol to use for left padding (default is None)
        :type left_pad_symbol: any
        :param right_pad_symbol: the symbol to use for right padding (default is None)
        :type right_pad_symbol: any
        :rtype: sequence or iter
        """
        sequence = pad_sequence(
            sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
        )

        history = []
        while n > 1:
            history.append(next(sequence))
            n -= 1
        for item in sequence:
            history.append(item)
            yield tuple(history)
            del history[0]


    def distinct_n_sentence_level(sentence, n):
        """
        Compute distinct-N for a single sentence.
        :param sentence: a list of words.
        :param n: int, ngram.
        :return: float, the metric value.
        """
        if len(sentence) == 0:
            return 0.0  # Prevent a zero division
        distinct_ngrams = set(ngrams(sentence, n))
        return len(distinct_ngrams) / len(sentence)


    def distinct_n_corpus_level(sentences, n):
        """
        Compute average distinct-N of a list of sentences (the corpus).
        :param sentences: a list of sentence.
        :param n: int, ngram.
        :return: float, the average value.
        """
        return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(
            sentences
        )
    
    return distinct_n_corpus_level(candidate, num)

def get_ppl(text, model, tokenizer, device):

    # Encode the input sentence
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Get the output logits
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Calculate the loss (cross entropy)
    loss = outputs.loss.item()

    # Convert the loss to perplexity
    perplexity = math.exp(loss)

    return perplexity

def get_bleu(reference, candidate):
    
    # reference : Correct Answer
    # candidate : Responsed Answer

    reference = [reference.split()]
    candidate = candidate.split()

    weights_unigram = (1, 0, 0, 0)
    bleu_unigram = sentence_bleu(
        reference,
        candidate,
        weights=weights_unigram,
        smoothing_function=SmoothingFunction().method1,
    )

    weights_bigram = (0.5, 0.5, 0, 0)
    bleu_bigram = sentence_bleu(
        reference,
        candidate,
        weights=weights_bigram,
        smoothing_function=SmoothingFunction().method1,
    )
    
    return bleu_unigram, bleu_bigram

def get_result(result):
    mean_distinct_1 = get_distinct([i.strip() for i in result['responses']],1)
    mean_distinct_2 = get_distinct([i.strip() for i in result['responses']],2)

    mean_bert = np.mean([i['precision']['precision'] for i in result['bert_scores']])

    mean_rouge1 = np.mean([i["rouge1"] for i in result['rouge_scores']])

    mean_rouge2 = np.mean([i["rouge2"] for i in result['rouge_scores']])

    mean_rougeL = np.mean([i["rougeL"] for i in result['rouge_scores']])

    mean_rougeLsum = np.mean([i["rougeLsum"] for i in result['rouge_scores']])

    mean_bleu_1 = np.mean(result['bleu_scores']['unigram'])

    mean_bleu_2 = np.mean(result['bleu_scores']['bigram'])

    mean_bleu_3 = np.mean(result['bleu_scores']['trigram'])

    mean_bleu_4 = np.mean(result['bleu_scores']['fourgram'])

    mean_ppl = np.mean(result['ppl_values'])

    return (
        mean_bert,
        mean_rouge1,
        mean_rouge2,
        mean_rougeL,
        mean_rougeLsum,
        mean_bleu_1,
        mean_bleu_2,
        mean_bleu_3,
        mean_bleu_4,        
        mean_distinct_1,
        mean_distinct_2,
        mean_ppl,
    )