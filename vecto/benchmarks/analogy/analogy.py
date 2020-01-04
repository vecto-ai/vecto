import datetime
import os
import uuid
import logging
import progressbar
# from tqdm import tqdm
import sklearn
from ..base import Benchmark
from .io import get_pairs
from .solvers import LinearOffset, LRCos, PairDistance
from .solvers import ThreeCosAvg, ThreeCosMul, ThreeCosMul2
from .solvers import SimilarToAny, SimilarToB


logger = logging.getLogger(__name__)


def select_method(key):
    if key == "3CosAvg":
        method = ThreeCosAvg
    elif key == "SimilarToAny":
        method = SimilarToAny
    elif key == "SimilarToB":
        method = SimilarToB
    elif key == "3CosMul":
        method = ThreeCosMul
    elif key == "3CosMul2":
        method = ThreeCosMul2
    elif key == "3CosAdd":
        method = LinearOffset
    elif key == "PairDistance":
        method = PairDistance
    elif key == "LRCos" or key == "SVMCos":
        method = LRCos
    else:
        raise RuntimeError("method name not recognized")
    return method


class Analogy(Benchmark):

    def __init__(self,
                 method="3CosAdd",
                 normalize=True,
                 ignore_oov=True,
                 do_top5=True,
                 # need_subsample=False,
                 size_cv_test=1,
                 set_aprimes_test=None,
                 exclude=True,
                 name_classifier='LR',
                 name_kernel="linear"):
        self.normalize = normalize
        self.method = method
        self.ignore_oov = ignore_oov
        self.do_top5 = do_top5
        # self.need_subsample = need_subsample
        self.normalize = normalize
        self.size_cv_test = size_cv_test
        self.set_aprimes_test = set_aprimes_test
#        self.inverse_regularization_strength = inverse_regularization_strength
        self.exclude = exclude
        self.name_classifier = name_classifier
        self.name_kernel = name_kernel

        self.stats = {}

        # this are some hard-coded bits which will be implemented later
        self.result_miss = {
            "rank": -1,
            "reason": "missing words"
        }


    # def is_at_least_one_word_present(self, words):
    #     for w in words:
    #         if self.embs.vocabulary.get_id(w) >= 0:
    #             return True
    #     return False


    # def gen_vec_single_nonoise(self, pairs):
    #     a, a_prime = zip(*pairs)
    #     a_prime = [i for sublist in a_prime for i in sublist]
    #     a_prime = [i for i in a_prime if self.embs.vocabulary.get_id(i) >= 0]
    #     x = list(a_prime) + list(a)
    #     X = np.array([self.embs.get_vector(i) for i in x])
    #     Y = np.hstack([np.ones(len(a_prime)), np.zeros(len(x) - len(a_prime))])
    #     return X, Y

    # def create_list_test_right(self, pairs):
    #     global set_aprimes_test
    #     a, a_prime = zip(*pairs)
    #     a_prime = [i for sublist in a_prime for i in sublist]
    #     set_aprimes_test = set(a_prime)

    # def get_distance_closest_words(self, center, cnt_words=1):
    #     scores = self.get_most_similar_fast(center)
    #     ids_max = np.argsort(scores)[::-1]
    #     distances = np.zeros(cnt_words)
    #     for i in range(cnt_words):
    #         distances[i] = scores[ids_max[i + 1]]
    #     return distances.mean()

    def run_category(self, pairs):
        details = []
        kfold = sklearn.model_selection.KFold(n_splits=len(pairs) // self.size_cv_test)
        cnt_splits = kfold.get_n_splits(pairs)
        loo = kfold.split(pairs)
        # if self.need_subsample:
        #    file_out = open("/dev/null", "a", errors="replace")
        #    loo = sklearn.cross_validation.KFold(
        #        n=len(pairs), n_folds=len(pairs) // self.size_cv_test)
        #    for max_size_train in range(10, 300, 5):
        #        finished = False
        #        my_prog = tqdm(0, total=len(loo), desc=name_category + ":" + str(max_size_train))
        #        for train, test in loo:
        #            p_test = [pairs[i] for i in test]
        #            p_train = [pairs[i] for i in train]
        #            p_train = [x for x in p_train if not self.is_pair_missing(x)]
        #            if len(p_train) <= max_size_train:
        #                finished = True
        #                continue
        #            p_train = random.sample(p_train, max_size_train)
        #            my_prog.update()
        #            self.do_test_on_pairs(p_train, p_test, file_out)
        #        if finished:
        #            break

        # my_prog = tqdm(0, total=cnt_splits, desc=name_category)
        my_prog = progressbar.ProgressBar(max_value=cnt_splits)
        cnt = 0
        for train, test in loo:
            p_test = [pairs[i] for i in test]
            p_train = [pairs[i] for i in train]
            # p_train = [x for x in p_train if not is_pair_missing(x)]
            cnt += 1
            my_prog.update(cnt)
            details += self.solver.do_test_on_pairs(p_train, p_test)

        out = dict()
        out["details"] = details
        results = {}
        # TODO: move this logic to solver
        results["cnt_questions_correct"] = self.solver.cnt_total_correct
        results["cnt_questions_total"] = self.solver.cnt_total_total
        if self.solver.cnt_total_total == 0:
            results["accuracy"] = -1
        else:
            results["accuracy"] = self.solver.cnt_total_correct / self.solver.cnt_total_total
        out["result"] = results
        # str_results = json.dumps(jsonify(out), indent=4, separators=(',', ': '), sort_keys=True)
        return out

    def run(self, embs, dataset):  # group_subcategory
        self.embs = embs
        # self.solver = select_method(self.method)(self.embs, exclude=self.exclude)


        if self.normalize:
            self.embs.normalize()
        self.embs.cache_normalized_copy()

        results = []
        for filename in dataset.file_iterator():
            self.solver = select_method(self.method)(self.embs, exclude=self.exclude) # initialize the solve for every analogy dataset file
            logger.info("processing " + filename)
            pairs = get_pairs(filename)
            name_category = os.path.basename(os.path.dirname(filename))
            name_subcategory = os.path.basename(filename)
            experiment_setup = dict()
            experiment_setup["dataset"] = dataset.metadata
            experiment_setup["embeddings"] = self.embs.metadata
            experiment_setup["category"] = name_category
            experiment_setup["subcategory"] = name_subcategory
            experiment_setup["task"] = "word_analogy"
            experiment_setup["default_measurement"] = "accuracy"
            experiment_setup["method"] = self.method
            experiment_setup["uuid"] = str(uuid.uuid4())
            if not self.exclude:
                experiment_setup["method"] += "_honest"
            experiment_setup["timestamp"] = datetime.datetime.now().isoformat()
            result_for_category = self.run_category(pairs)
            result_for_category["experiment_setup"] = experiment_setup
            results.append(result_for_category)
        # if group_subcategory:
            # results.extend(self.group_subcategory_results(results))
        return results

    # def group_subcategory_results(self, results):  # todo: figure out if we need this
        # group analogy results, based on the category
    #    new_results = {}
    #    for result in results:
    #        cnt_correct = 0
    #        cnt_total = 0
    #        for t in result['details']:
    #            if t['rank'] == 0:
    #                cnt_correct += 1
    #            cnt_total += 1

    #        k = result['experiment_setup']['category']

    #        if k in new_results:
    #            new_results[k]['experiment_setup']['cnt_questions_correct'] += cnt_correct
    #            new_results[k]['experiment_setup']['cnt_questions_total'] += cnt_total
    #            new_results[k]['details'] += result['details']
    #        else:
    #            new_results[k] = result.copy()
    #            del new_results[k]['experiment_setup']['category']
    #           new_results[k]['experiment_setup']['dataset'] = k
    #            # new_results[k]['experiment_setup'] = r['experiment_setup'].copy()
    #            new_results[k]['experiment_setup']['category'] = k
    #            new_results[k]['experiment_setup']['subcategory'] = k
    #            new_results[k]['experiment_setup']['cnt_questions_correct'] = cnt_correct
    #            new_results[k]['experiment_setup']['cnt_questions_total'] = cnt_total
    #    for k, v in new_results.items():
    #        new_results[k]['result'] = new_results[k]['experiment_setup']['cnt_questions_correct'] * 1.0 / new_results[k]['experiment_setup']['cnt_questions_total']
    #    out = []
    #    for k, v in new_results.items():
    #        out.append(new_results[k])
    #    return out

    #def subsample_dims(self, newdim):
        #self.embs.matrix = self.embs.matrix[:, 0:newdim]
        #self.embs.name = re.sub("_d(\d+)", "_d{}".format(newdim), self.embs.name)

    # def get_result(self, embeddings, path_dataset):  # , group_subcategory=False
    #     if self.normalize:
    #         embeddings.normalize()
    #     dataset = Dataset(path_dataset)
    #     results = self.run(embeddings, dataset)  #group_subcategory
    #     return results
