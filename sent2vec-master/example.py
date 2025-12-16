import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('my_model.bin')
emb = model.embed_sentence("once upon a time")
print(type(emb))
print(emb.shape)
embs =model.embed_sentences(["first sentence .","another sentence"])
print(type(embs))
print(embs.shape)