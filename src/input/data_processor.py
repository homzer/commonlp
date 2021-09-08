from src.input.tokenizer import Tokenizer
from random import randint


def padding(ids, fixed_length):
    while len(ids) < fixed_length:
        ids.append(0)
    return ids


def truncating(ids, fixed_length):
    if len(ids) > fixed_length:
        ids = ids[:fixed_length]
    return ids


class DataProcessor:
    def __init__(self, vocab_file='config/vocab.txt'):
        self._tokenizer = Tokenizer(vocab_file)

    def texts2ids(self, texts, fixed_length: int, add_start_id=False):
        """
        Texts -> Tokens -> Ids.
        :param texts: `list` of `str`
        :param fixed_length: `int` the fixed length of text.
        :param add_start_id: whether to add start id.
        :return: 2-D `list` the ids of tokens.
        """
        return self.tokens2ids(
            self.texts2tokens(texts), fixed_length, add_start_id)

    def texts2tokens(self, texts):
        """
        Tokenize texts.
        :param texts: can be either `list` or `str`.
        :return: 2D `list` if type(texts) is `list`,
        1D `list` if type(texts) is `str`
        """
        tokens = []
        if type(texts) == list:
            for text in texts:
                tokens.append(self._tokenizer.tokenize(text))
        elif type(texts) == str:
            tokens = self._tokenizer.tokenize(texts)
        else:
            raise ValueError(
                "type(texts) must be either `list` or `str`. got %s"
                % type(texts))
        return tokens

    def tokens2ids(self, tokens: list, fixed_length: int, add_start_id=False):
        """
        Convert tokens to ids.
        :param tokens: can be either 2D or 1D `list`.
        :param fixed_length: padding or Truncating according to the fixed length.
        :param add_start_id: whether to add start id.
        :return: `list` same shape of tokens
        """
        tokens_ids = []
        if type(tokens[0]) == list:
            for token in tokens:
                if add_start_id:
                    token_ids = [self._tokenizer.get_start_id()]
                    token_ids.extend(self._tokenizer.tokens2ids(token))
                else:
                    token_ids = self._tokenizer.tokens2ids(token)
                token_ids = padding(token_ids, fixed_length)
                token_ids = truncating(token_ids, fixed_length)
                assert len(token_ids) == fixed_length
                tokens_ids.append(token_ids)
        elif type(tokens[0]) == str:
            if add_start_id:
                tokens_ids = [self._tokenizer.get_start_id()]
                tokens_ids.extend(self._tokenizer.tokens2ids(tokens))
            else:
                tokens_ids = self._tokenizer.tokens2ids(tokens)
            tokens_ids = padding(tokens_ids, fixed_length)
            tokens_ids = truncating(tokens_ids, fixed_length)
            assert len(tokens_ids) == fixed_length
        else:
            raise ValueError(
                "type(tokens[0]) must be either `list` or `str`. got %s"
                % type(tokens[0]))
        return tokens_ids

    def ids2tokens(self, ids: list):
        """
        Convert ids to tokens.
        :param ids: can be either 2D or 1D `list`.
        :return: `list` same shape of input.
        """
        tokens = []
        if type(ids[0]) == str:
            tokens = self._tokenizer.ids2tokens(ids)
        else:
            for i in range(len(ids)):
                tokens.append(self._tokenizer.ids2tokens(ids[i]))
        return tokens

    def mask(self, tokens, max_num_mask=2):
        def __masking(_tokens):
            length = len(_tokens)
            begin_idx = randint(1, length)
            size = randint(0, max_num_mask)
            for idx in range(size):
                index = idx + begin_idx
                if index < length:
                    _tokens[index] = self._tokenizer.MASK_TOKEN
            return _tokens

        if type(tokens[0]) == list:
            tokens_count = len(tokens)
            for i in range(tokens_count):
                tokens[i] = __masking(tokens[i])
        elif type(tokens[0]) == str:
            tokens = __masking(tokens)
        else:
            raise ValueError(
                "type(tokens[0]) expected to be either `list` or `str`. got %s"
                % type(tokens[0]))
        return tokens
