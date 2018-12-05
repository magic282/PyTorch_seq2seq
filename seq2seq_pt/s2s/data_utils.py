import s2s


def src_to_ids(words, vocab):
    ids = []
    oovs = []
    for w in words:
        w_id = vocab.lookup(w, default=s2s.Constants.UNK)
        if w_id == s2s.Constants.UNK:
            if w not in oovs:
                oovs.append(w)
            oov_id = vocab.size() + oovs.index(w)
            ids.append(oov_id)
        else:
            ids.append(w_id)
    return ids, oovs


def tgt_to_ids(words, vocab, src_oovs, bosWord=None, eosWord=None):
    ids = []
    if bosWord is not None:
        ids.append(vocab.lookup(bosWord))
    for w in words:
        w_id = vocab.lookup(w, default=s2s.Constants.UNK)
        if w_id == s2s.Constants.UNK:
            if w in src_oovs:
                idx = vocab.size() + src_oovs.index(w)
                ids.append(idx)
            else:
                ids.append(s2s.Constants.UNK)
        else:
            ids.append(w_id)
    if eosWord is not None:
        ids.append(vocab.lookup(eosWord))
    return ids
