import nltk
doc = '''What types of reports will I receive on a daily basis for custom indices? '''
# tokenize doc
tokenized_doc = nltk.word_tokenize(doc)
 
# tag sentences and use nltk's Named Entity Chunker
tagged_sentences = nltk.pos_tag(tokenized_doc)
ne_chunked_sents = nltk.ne_chunk(tagged_sentences)
 
# extract all named entities
named_entities = []
for tagged_tree in ne_chunked_sents:
    if hasattr(tagged_tree, 'label'):
        entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) #
        entity_type = tagged_tree.label() # get NE category
        named_entities.append((entity_name, entity_type))
print(named_entities)
