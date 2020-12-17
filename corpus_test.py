from vecto.corpus import ViewCorpus

path = "./tests/data/corpora/"
corpus = ViewCorpus(path)
corpus.load_dir_strucute()
search = 9
print("searching ", search, "in", corpus.tree)
pos, offset = corpus.get_file_and_offset(search, jump_to_next=True)
print("final pos", pos, ", offset", offset)
# rank 0 creates corpus from dir
# corpus has inside all file list and sizes
# use manually splits sends metadata of corpus : tree of dirs and files with uncompressed sizes to all workers
# otehr workers create corpora from that metadata using special service method like __from_metadata
# to avoid exessive file IO

# for time being - everybody just reads from FS

# # view = corpus.view(start_percent, end_pecent)
# print(corpus)
# iter_token = corpus.get_line_iterator()
# for s in iter_token:
#     print(s)

