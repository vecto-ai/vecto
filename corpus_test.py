from vecto.corpus import ViewCorpus

path = "./tests/data/corpora/multiple_small"
corpus = ViewCorpus(path)
corpus.load_dir_strucute()
print("three is ", corpus.tree)

# TODO: move all this to unittests 
print("testing get offset")
for q in [9, 11]:
    for start in [True, False]:
        print("search ", q, " with start=", start)
        pos, offset = corpus.get_file_and_offset(q, start_of_range=start, epsilon=2)
        print("pos", pos, ", offset", offset, "\n")
pos, offset = corpus.get_file_and_offset(15, start_of_range=False, epsilon=2)
print("pos", pos, ", offset", offset, "\n")


print("testing get get_line_iterator")
cnt_workers = 2
for i in range(cnt_workers):
    iterator = corpus.get_line_iterator(i, cnt_workers)
    print("worker", i, iterator)
    for line in iterator:
        print("line", line)

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

