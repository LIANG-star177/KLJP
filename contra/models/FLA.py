import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.nn.functional as F


class FLA(nn.Module):
    """
    The overarching Hierarchial Attention Network (HAN).
    """

    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, details={}):
        super(FLA, self).__init__()

        # n_classes = len(maps["a2i"])
        vocab_size = vocab_size
        self.emb_dim = emb_dim
        emb_size = emb_dim
        word_rnn_size = emb_dim
        sentence_rnn_size = emb_dim
        word_rnn_layers = 2
        sentence_rnn_layers = 2
        word_att_size = emb_dim
        sentence_att_size = emb_dim
        dropout=0.5
        self.details = details

        self.article_embedding = nn.Embedding(vocab_size, emb_dim)
        self.article_word_lstm = nn.LSTM(emb_dim, emb_dim, bidirectional = True)
        self.article_sent_lstm = nn.LSTM(2 * emb_dim, emb_dim, bidirectional = True)

        # Sentence-level attention module (which will, in-turn, contain the word-level attention module)
        self.fact_han = SentenceAttention(vocab_size, emb_size, word_rnn_size, sentence_rnn_size,
                                                    word_rnn_layers, sentence_rnn_layers, word_att_size,
                                                    sentence_att_size, dropout)

        self.article_han = SentenceAttention(vocab_size, emb_size, word_rnn_size, sentence_rnn_size,
                                                    word_rnn_layers, sentence_rnn_layers, word_att_size,
                                                    sentence_att_size, dropout)

        self.uad_fc = nn.Linear(2 * emb_dim, 2 * emb_dim)
        self.uas_fc = nn.Linear(2 * emb_dim, 2 * emb_dim)
        self.uaw_fc = nn.Linear(2 * emb_dim, 2 * emb_dim)

        # Classifier
        self.fc_ar1 = nn.Linear(4 * sentence_rnn_size, sentence_rnn_size)
        self.fc_ch1 = nn.Linear(4 * sentence_rnn_size, sentence_rnn_size)
        self.fc_pt1 = nn.Linear(4 * sentence_rnn_size, sentence_rnn_size)

        self.fc_ar2 = nn.Linear(sentence_rnn_size, len(maps["a2i"]))
        self.fc_ch2 = nn.Linear(sentence_rnn_size, len(maps["c2i"]))
        self.fc_pt2 = nn.Linear(sentence_rnn_size, maps["pt_cls_len"])

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        documents = data["fact"]["input_ids"].cuda() # [batch, sent, word]
        batch = documents.shape[0]
        words_per_sentence = (documents != 0).sum(-1)
        sentences_per_document = (words_per_sentence != 0).sum(-1)
        df, _, _ = self.fact_han(documents, sentences_per_document, words_per_sentence)  # (n_documents, 2 * sentence_rnn_size), (n_documents, max(sentences_per_document), max(words_per_sentence)), (n_documents, max(sentences_per_document))
        
        articles = self.details["a_details"]["input_ids"].cuda() # [ar_num, sent, word]
        article_emb = self.article_embedding(articles)

        shape = [-1] + list(article_emb.shape)[-2:]
        ar_flat = article_emb.view(shape)
        ar_flat, _ = self.article_word_lstm(ar_flat)
        # ar_flat = ar_flat[:,:,:300] + ar_flat[:,:,300:]
        shape = list(article_emb.shape)[:-1] + [self.emb_dim * 2]
        ar_word = ar_flat.view(shape) # [ar, sent, word, dim] [70, 5, 50, 300]

        uaw = self.uaw_fc(df) # [b, dim] [16, 300]
        uas = self.uas_fc(df) # [b, dim] [16, 300]
        uad = self.uad_fc(df) # [b, dim] [16, 300]
        
        # word-level attention
        M_word = torch.matmul(ar_word, uaw.T).permute(3,0,1,2) # [batch, ar, sent, word] [16, 70, 5, 50]
        M_word = F.softmax(M_word, dim = -1)

        ar_sent = []
        for bid in range(batch):
            alpha = M_word[bid].unsqueeze(-1)
            cur_emb = alpha * ar_word
            cur_emb = cur_emb.sum(2).squeeze(2)
            ar_sent.append(cur_emb)
        ar_sent = torch.stack(ar_sent, dim = 0) # [batch, ar, sent, dim] [16, 70, 5, 300]

        shape = [-1] + list(ar_sent.shape)[-2:]
        ar_flat = ar_sent.view(shape)
        ar_flat, _ = self.article_sent_lstm(ar_flat)
        # ar_flat = ar_flat[:,:,:300] + ar_flat[:,:,300:]
        ar_sent = ar_flat.view(ar_sent.shape) # [batch, ar, sent, dim] [16, 70, 5, 300]

        # sent-level attention
        uas = uas.unsqueeze(1).unsqueeze(1)
        M_sent = ar_sent * uas
        M_sent = M_sent.sum(-1)
        M_sent = F.softmax(M_sent, dim = -1).unsqueeze(-1)

        ar_doc = M_sent * ar_sent
        ar_doc = ar_doc.sum(2).squeeze(1)

        # article aggregator
        uad = uad.unsqueeze(1)
        M_doc = ar_doc * uad
        M_doc = M_doc.sum(-1)
        M_doc = F.softmax(M_doc, dim = -1).unsqueeze(-1)

        da = M_doc * ar_doc
        da = da.sum(1).squeeze(1)

        embedding = torch.cat([df, da], dim = -1)
        out_ar = self.fc_ar2(nn.ReLU()(self.fc_ar1(embedding)))
        out_ch = self.fc_ch2(nn.ReLU()(self.fc_ch1(embedding)))
        out_pt = self.fc_pt2(nn.ReLU()(self.fc_pt1(embedding)))

        return {
            "article": out_ar,
            "charge": out_ch,
            "penalty": out_pt,
            "cl_loss": torch.tensor(0).cuda(),
            "meta": {}
        }



class SentenceAttention(nn.Module):
    """
    The sentence-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers,
                 word_att_size, sentence_att_size, dropout):
        super(SentenceAttention, self).__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size,
                                            dropout)

        # Bidirectional sentence-level RNN
        self.sentence_rnn = nn.GRU(2 * word_rnn_size, sentence_rnn_size, num_layers=sentence_rnn_layers,
                                   bidirectional=True, dropout=dropout, batch_first=True)

        # Sentence-level attention network
        self.sentence_attention = nn.Linear(2 * sentence_rnn_size, sentence_att_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Linear(sentence_att_size, 1,
                                                 bias=False)  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector
        # You could also do this with:
        # self.sentence_context_vector = nn.Parameter(torch.FloatTensor(1, sentence_att_size))
        # self.sentence_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence, uas = None, uaw = None):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """

        # Re-arrange as sentences by removing sentence-pads (DOCUMENTS -> SENTENCES)
        packed_sentences = pack_padded_sequence(documents,
                                                lengths=sentences_per_document.tolist(),
                                                batch_first=True,
                                                enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentences (n_sentences, word_pad_len)

        # Re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        packed_words_per_sentence = pack_padded_sequence(words_per_sentence,
                                                         lengths=sentences_per_document.tolist(),
                                                         batch_first=True,
                                                         enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentence lengths (n_sentences)
        

        # Find sentence embeddings by applying the word-level attention module
        sentences, word_alphas = self.word_attention(packed_sentences.data,
                                                     packed_words_per_sentence.data,
                                                     uaw)  # (n_sentences, 2 * word_rnn_size), (n_sentences, max(words_per_sentence))
        sentences = self.dropout(sentences)

        # Apply the sentence-level RNN over the sentence embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_sentences, _ = self.sentence_rnn(PackedSequence(data=sentences,
                                                               batch_sizes=packed_sentences.batch_sizes,
                                                               sorted_indices=packed_sentences.sorted_indices,
                                                               unsorted_indices=packed_sentences.unsorted_indices))  # a PackedSequence object, where 'data' is the output of the RNN (n_sentences, 2 * sentence_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_s = self.sentence_attention(packed_sentences.data)  # (n_sentences, att_size)
        att_s = torch.tanh(att_s)  # (n_sentences, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)

        if uas is not None:
            att_s = torch.matmul(att_s, uas)
        else:
            att_s = self.sentence_context_vector(att_s).squeeze(1)  # (n_sentences)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over sentences in the same document

        # First, take the exponent
        max_value = att_s.max()  # scalar, for numerical stability during exponent calculation
        att_s = torch.exp(att_s - max_value)  # (n_sentences)

        # Re-arrange as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        att_s, _ = pad_packed_sequence(PackedSequence(data=att_s,
                                                      batch_sizes=packed_sentences.batch_sizes,
                                                      sorted_indices=packed_sentences.sorted_indices,
                                                      unsorted_indices=packed_sentences.unsorted_indices),
                                       batch_first=True)  # (n_documents, max(sentences_per_document))

        # Calculate softmax values as now sentences are arranged in their respective documents
        sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)  # (n_documents, max(sentences_per_document))

        # Similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        documents, _ = pad_packed_sequence(packed_sentences,
                                           batch_first=True)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)

        # Find document embeddings
        documents = documents * sentence_alphas.unsqueeze(
            2)  # (n_documents, max(sentences_per_document), 2 * sentence_rnn_size)
        documents = documents.sum(dim=1)  # (n_documents, 2 * sentence_rnn_size)

        # Also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas, _ = pad_packed_sequence(PackedSequence(data=word_alphas,
                                                            batch_sizes=packed_sentences.batch_sizes,
                                                            sorted_indices=packed_sentences.sorted_indices,
                                                            unsorted_indices=packed_sentences.unsorted_indices),
                                             batch_first=True)  # (n_documents, max(sentences_per_document), max(words_per_sentence))

        return documents, word_alphas, sentence_alphas


class WordAttention(nn.Module):
    """
    The word-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param word_att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super(WordAttention, self).__init__()

        # Embeddings (look-up) layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # Bidirectional word-level RNN
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)

        # Word-level attention network
        self.word_attention = nn.Linear(2 * word_rnn_size, word_att_size)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)
        # You could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        self.dropout = nn.Dropout(dropout)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.embeddings.parameters():
            p.requires_grad = fine_tune

    def forward(self, sentences, words_per_sentence, uaw = None):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """

        # Get word embeddings, apply dropout
        sentences = self.dropout(self.embeddings(sentences))  # (n_sentences, word_pad_len, emb_size)

        # Re-arrange as words by removing word-pads (SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(sentences,
                                            lengths=words_per_sentence.tolist(),
                                            batch_first=True,
                                            enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened words (n_words, word_emb)



        # Apply the word-level RNN over the word embeddings (PyTorch automatically applies it on the PackedSequence)
        packed_words, _ = self.word_rnn(
            packed_words)  # a PackedSequence object, where 'data' is the output of the RNN (n_words, 2 * word_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(packed_words.data)  # (n_words, att_size)
        att_w = torch.tanh(att_w)  # (n_words, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)

        if uaw is not None:
            att_w = torch.matmul(att_w, uaw)
        else:
            att_w = self.word_context_vector(att_w).squeeze(1)  # (n_words)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = torch.exp(att_w - max_value)  # (n_words)

        # Re-arrange as sentences by re-padding with 0s (WORDS -> SENTENCES)
        att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                      batch_sizes=packed_words.batch_sizes,
                                                      sorted_indices=packed_words.sorted_indices,
                                                      unsorted_indices=packed_words.unsorted_indices),
                                       batch_first=True)  # (n_sentences, max(words_per_sentence))

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)  # (n_sentences, max(words_per_sentence))

        # Similarly re-arrange word-level RNN outputs as sentences by re-padding with 0s (WORDS -> SENTENCES)
        sentences, _ = pad_packed_sequence(packed_words,
                                           batch_first=True)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_sentences, max(words_per_sentence), 2 * word_rnn_size)
        sentences = sentences.sum(dim=1)  # (n_sentences, 2 * word_rnn_size)

        return sentences, word_alphas


