import datetime
import os
import random
import scipy
import uuid
import numpy as np
import logging
import progressbar
# from tqdm import tqdm
import sklearn
from sklearn.linear_model import LogisticRegression
from itertools import product
from vecto.data import Dataset
from ..base import Benchmark

logger = logging.getLogger(__name__)


def get_pairs(fname):  # todo: optional lower-casing, move to some io module
    pairs = []
    with open(fname) as file_in:
        id_line = 0
        for line in file_in:
            if line.strip() == '':
                continue
            try:
                id_line += 1
                if "\t" in line:
                    parts = line.lower().split("\t")
                else:
                    parts = line.lower().split()
                left = parts[0]
                right = parts[1]
                right = right.strip()
                if "/" in right:
                    right = [i.strip() for i in right.split("/")]
                else:
                    right = [i.strip() for i in right.split(",")]
                pairs.append([left, right])
            except:
                print("error reading pairs")
                print("in file", fname)
                print("in line", id_line, line)
                exit(-1)
    return pairs


class Analogy(Benchmark):

    def __init__(self, normalize=True,
                 ignore_oov=True,
                 do_top5=True,
                 # need_subsample=False,
                 size_cv_test=1,
                 set_aprimes_test=None,
                 inverse_regularization_strength=1.0,
                 exclude=True,
                 name_classifier='LR',
                 name_kernel="linear"):
        self.normalize = normalize
        self.ignore_oov = ignore_oov
        self.do_top5 = do_top5
        # self.need_subsample = need_subsample
        self.normalize = normalize
        self.size_cv_test = size_cv_test
        self.set_aprimes_test = set_aprimes_test
        self.inverse_regularization_strength = inverse_regularization_strength
        self.exclude = exclude
        self.name_classifier = name_classifier
        self.name_kernel = name_kernel

        self.stats = {}
        self.cnt_total_correct = 0
        self.cnt_total_total = 0

        # this are some hard-coded bits which will be implemented later
        self.result_miss = {
            "rank": -1,
            "reason": "missing words"
        }

    @property
    def method(self):
        return type(self).__name__

    def normed(self, v):
        if self.normalize:
            return v
        else:
            return v / np.linalg.norm(v)

    def get_most_similar_fast(self, v):
        scores = self.normed(v) @ self.embs._normalized_matrix.T
        scores = (scores + 1) / 2
        return scores

    def get_most_collinear_fast(self, a, ap, b):
        scores = np.zeros(self.embs.matrix.shape[0])
        offset_target = ap - a
        offset_target = offset_target / np.linalg.norm(offset_target)
        m_diff = self.embs.matrix - b
        norm = np.linalg.norm(m_diff, axis=1)
        norm[norm == 0] = 100500
        m_diff /= norm[:, None]
        scores = m_diff @ offset_target
        return scores

    # def is_at_least_one_word_present(self, words):
    #     for w in words:
    #         if self.embs.vocabulary.get_id(w) >= 0:
    #             return True
    #     return False

    def is_pair_missing(self, pairs):
        for pair in pairs:
            if self.embs.vocabulary.get_id(pair[0]) < 0:
                return True
            if self.embs.vocabulary.get_id(pair[1][0]) < 0:
                return True
            # if not is_at_least_one_word_present(pair[1]):
            # return True
        return False

    def gen_vec_single(self, pairs):
        a, a_prime = zip(*pairs)
        a_prime = [i[0] for i in a_prime]
        # a_prime=[i for sublist in a_prime for i in sublist]
        a_prime = [i for i in a_prime if self.embs.vocabulary.get_id(i) >= 0]
        a = [i for i in a if self.embs.vocabulary.get_id(i) >= 0]
        cnt_noise = len(a)
        noise = [random.choice(self.embs.vocabulary.lst_words) for i in range(cnt_noise)]

        if len(a_prime) == 0:
            a_prime.append(random.choice(self.embs.vocabulary.lst_words))
        train_vectors = list(a_prime) + list(a) + list(a) + list(a) + list(a) + noise
        train_vectors = np.array([self.embs.get_vector(i) for i in train_vectors])
        labels = np.hstack([np.ones(len(a_prime)), np.zeros(len(train_vectors) - len(a_prime))])
        return train_vectors, labels

    # def gen_vec_single_nonoise(self, pairs):
    #     a, a_prime = zip(*pairs)
    #     a_prime = [i for sublist in a_prime for i in sublist]
    #     a_prime = [i for i in a_prime if self.embs.vocabulary.get_id(i) >= 0]
    #     x = list(a_prime) + list(a)
    #     X = np.array([self.embs.get_vector(i) for i in x])
    #     Y = np.hstack([np.ones(len(a_prime)), np.zeros(len(x) - len(a_prime))])
    #     return X, Y

    def get_crowndedness(self, vector):
        scores = self.get_most_similar_fast(vector)
        scores.sort()
        return (scores[-11:-1][::-1]).tolist()

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

    def get_rank(self, source, center):
        if isinstance(center, str):
            center = self.embs.get_vector(center)
        if isinstance(source, str):
            source = [source]
        scores = self.get_most_similar_fast(center)
        ids_max = np.argsort(scores)[::-1]
        for i in range(ids_max.shape[0]):
            if self.embs.vocabulary.get_word_by_id(ids_max[i]) in source:
                break
        rank = i
        return rank

    @staticmethod
    def get_verbose_question(pair_test, pairs_train):
        extr = ""
        if len(pairs_train) == 1:
            extr = "as {} is to {}".format(pairs_train[0][1], pairs_train[0][0])
        res = "What is to {} {}".format(pair_test[0], extr)
        return res

    def process_prediction(self, p_test_one, scores, score_reg, score_sim, p_train=[]):
        ids_max = np.argsort(scores)[::-1]
        result = dict()
        cnt_answers_to_report = 6
        set_exclude = set()
        if len(p_train) == 1:
            set_exclude.update(set([p_train[0][0]]) | set(p_train[0][1]))

        set_exclude.add(p_test_one[0])
        result["question verbose"] = self.get_verbose_question(p_test_one, p_train)
        result["b"] = p_test_one[0]
        result["expected answer"] = p_test_one[1]
        result["predictions"] = []
        result['set_exclude'] = [e for e in set_exclude]

        cnt_reported = 0
        for i in ids_max[:10]:
            prediction = dict()
            ans = self.embs.vocabulary.get_word_by_id(i)
            if self.exclude and (ans in set_exclude):
                continue
            cnt_reported += 1
            prediction["score"] = float(scores[i])
            prediction["answer"] = ans
            if ans in p_test_one[1]:
                prediction["hit"] = True
            else:
                prediction["hit"] = False
            result["predictions"].append(prediction)
            if cnt_reported >= cnt_answers_to_report:
                break
        rank = 0
        for i in range(ids_max.shape[0]):
            ans = self.embs.vocabulary.get_word_by_id(ids_max[i])
            if self.exclude and (ans in set_exclude):
                continue
            if ans in p_test_one[1]:
                break
            rank += 1
        result["rank"] = rank
        if rank == 0:
            self.cnt_total_correct += 1
        self.cnt_total_total += 1
        # vec_b_prime = self.embs.get_vector(p_test_one[1][0])
        # result["closest words to answer 1"] = get_distance_closest_words(vec_b_prime,1)
        # result["closest words to answer 5"] = get_distance_closest_words(vec_b_prime,5)
        # where prediction lands:
        ans = self.embs.vocabulary.get_word_by_id(ids_max[0])
        result["landing_b"] = (ans == p_test_one[0])
        result["landing_b_prime"] = (ans in p_test_one[1])
        all_a = [i[0] for i in p_train]
        all_a_prime = [item for sublist in p_train for item in sublist[1]]
        result["landing_a"] = (ans in all_a)
        result["landing_a_prime"] = (ans in all_a_prime)
        return result

    def run_category(self, pairs):
        self.cnt_total_correct = 0
        self.cnt_total_total = 0
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
            details += self.do_test_on_pairs(p_train, p_test)

        out = dict()
        out["details"] = details
        results = {}
        if self.cnt_total_total == 0:
            results["accuracy"] = -1
        else:
            results["accuracy"] = self.cnt_total_correct / self.cnt_total_total
            results["cnt_questions_correct"] = self.cnt_total_correct
            results["cnt_questions_total"] = self.cnt_total_total
        out["result"] = results
        # str_results = json.dumps(jsonify(out), indent=4, separators=(',', ': '), sort_keys=True)
        return out

    def run(self, embs, path_dataset):  # group_subcategory
        self.embs = embs

        if self.normalize:
            self.embs.normalize()
        self.embs.cache_normalized_copy()

        results = []
        dataset = Dataset(path_dataset)
        for filename in dataset.file_iterator():
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

    def get_result(self, embeddings, path_dataset):  # , group_subcategory=False
        if self.normalize:
            embeddings.normalize()
        results = self.run(embeddings, path_dataset)  #group_subcategory
        return results


class PairWise(Analogy):
    def do_test_on_pairs(self, pairs_train, pairs_test):
        results = []
        for p_train, p_test in product(pairs_train, pairs_test):
            if self.is_pair_missing([p_train, p_test]):
                self.cnt_total_total += 1
                result = {}
                result["rank"] = -1
                result["question verbose"] = self.get_verbose_question(p_test, [p_train])
                # todo: report which exaclt words are missing
            else:
                result = self.do_on_two_pairs(p_train, p_test)
                result["b in neighbourhood of b_prime"] = self.get_rank(p_test[0], p_test[1][0])
                result["b_prime in neighbourhood of b"] = self.get_rank(p_test[1], p_test[0])
            results.append(result)
        return results

    def do_on_two_pairs(self, p_train, p_test):
        vec_a = self.embs.get_vector(p_train[0])
        vec_a_prime = self.embs.get_vector(p_train[1][0])
        vec_b = self.embs.get_vector(p_test[0])
        vec_b_prime = self.embs.get_vector(p_test[1][0])
        if scipy.sparse.issparse(self.embs.matrix):
            vec_a = vec_a.toarray()[0]
            vec_a_prime = vec_a_prime.toarray()[0]
            vec_b = vec_b.toarray()[0]

        scores, vec_b_prime_predicted = self.compute_scores(vec_a, vec_a_prime, vec_b)
        # ids_max = np.argsort(scores)[::-1]
        result = self.process_prediction(p_test, scores, None, None, [p_train])
        self.collect_stats(result, vec_a, vec_a_prime, vec_b, vec_b_prime, vec_b_prime_predicted)
        return result

    def collect_stats(self, result, vec_a, vec_a_prime, vec_b, vec_b_prime, vec_b_prime_predicted):
        if vec_b_prime_predicted is not None:
            result["similarity predicted to b_prime cosine"] = float(
                self.embs.cmp_vectors(vec_b_prime_predicted, vec_b_prime))

        result["similarity a to a_prime cosine"] = float(self.embs.cmp_vectors(vec_a, vec_a_prime))
        result["similarity a_prime to b_prime cosine"] = float(self.embs.cmp_vectors(vec_a_prime, vec_b_prime))
        result["similarity b to b_prime cosine"] = float(self.embs.cmp_vectors(vec_b, vec_b_prime))
        result["similarity a to b_prime cosine"] = float(self.embs.cmp_vectors(vec_a, vec_b_prime))

        result["distance a to a_prime euclidean"] = float(scipy.spatial.distance.euclidean(vec_a, vec_a_prime))
        result["distance a_prime to b_prime euclidean"] = float(
            scipy.spatial.distance.euclidean(vec_a_prime, vec_b_prime))
        result["distance b to b_prime euclidean"] = float(scipy.spatial.distance.euclidean(vec_b, vec_b_prime))
        result["distance a to b_prime euclidean"] = float(scipy.spatial.distance.euclidean(vec_a, vec_b_prime))

        result["crowdedness of b_prime"] = self.get_crowndedness(vec_b_prime)


class LinearOffset(PairWise):
    def compute_scores(self, vec_a, vec_a_prime, vec_b):
        vec_b_prime_predicted = vec_a_prime - vec_a + vec_b
        vec_b_prime_predicted = self.normed(vec_b_prime_predicted)
        scores = self.get_most_similar_fast(vec_b_prime_predicted)
        return scores, vec_b_prime_predicted


class PairDistance(PairWise):
    def compute_scores(self, vec_a, vec_a_prime, vec_b):
        scores = self.get_most_collinear_fast(vec_a, vec_a_prime, vec_b)
        return scores, None


class ThreeCosMul(PairWise):
    def compute_scores(self, vec_a, vec_a_prime, vec_b):
        epsilon = 0.001
        sim_a = self.get_most_similar_fast(vec_a)
        sim_a_prime = self.get_most_similar_fast(vec_a_prime)
        sim_b = self.get_most_similar_fast(vec_b)
        scores = (sim_a_prime * sim_b) / (sim_a + epsilon)
        return scores, None


class ThreeCosMul2(PairWise):
    def compute_scores(self, vec_a, vec_a_prime, vec_b):
        epsilon = 0.001
        # sim_a = get_most_similar_fast(vec_a)
        # sim_a_prime = get_most_similar_fast(vec_a_prime)
        # sim_b = get_most_similar_fast(vec_b)
        # scores = (sim_a_prime * sim_b) / (sim_a + epsilon)
        predicted = (((vec_a_prime + 0.5) / 2) * ((vec_b + 0.5) / 2)) / (((vec_a + 0.5) / 2) + epsilon)
        scores = self.get_most_similar_fast(predicted)
        return scores, predicted


# class SimilarToAny(PairWise):
#     def compute_scores(self, vectors):
#         scores = self.get_most_similar_fast(vectors)
#         best = scores.max(axis=0)
#         return best
#
#
# class SimilarToB(Analogy):
#     def do_test_on_pairs(self, pairs_train, pairs_test):
#         results = []
#         for p_test in pairs_test:
#             if self.is_pair_missing([p_test]):
#                 continue
#             result = self.do_on_two_pairs(p_test)
#             result["b in neighbourhood of b_prime"] = self.get_rank(p_test[0], p_test[1][0])
#             result["b_prime in neighbourhood of b"] = self.get_rank(p_test[1], p_test[0])
#             results.append(result)
#         return results
#
#     def do_on_two_pairs(self, pair_test):
#         if self.is_pair_missing([pair_test]):
#             result = self.result_miss
#         else:
#             vec_b = self.embs.get_vector(pair_test[0])
#             vec_b_prime = self.embs.get_vector(pair_test[1][0])
#             scores = self.get_most_similar_fast(vec_b)
#             result = self.process_prediction(pair_test, scores, None, None)
#             result["similarity to correct cosine"] = self.embs.cmp_vectors(vec_b, vec_b_prime)
#         return result


class ThreeCosAvg(Analogy):

    def do_test_on_pairs(self, p_train, p_test):
        vecs_a = []
        vecs_a_prime = []
        for pair in p_train:
            if self.is_pair_missing([pair]):
                continue
            vecs_a_prime_local = []
            for token in pair[1]:
                if self.embs.vocabulary.get_id(token) >= 0:
                    vecs_a_prime_local.append(self.embs.get_vector(token))
                break
            if len(vecs_a_prime_local) > 0:
                vecs_a.append(self.embs.get_vector(pair[0]))
                vecs_a_prime.append(np.vstack(vecs_a_prime_local).mean(axis=0))
        if len(vecs_a_prime) == 0:
            print("AAAA SOMETHIGN MISSING")
            return ([])

        vec_a = np.vstack(vecs_a).mean(axis=0)
        vec_a_prime = np.vstack(vecs_a_prime).mean(axis=0)

        results = []
        for p_test_one in p_test:
            if self.is_pair_missing([p_test_one]):
                continue
            vec_b_prime = self.embs.get_vector(p_test_one[1][0])
            vec_b = self.embs.get_vector(p_test_one[0])
            vec_b_prime_predicted = vec_a_prime - vec_a + vec_b
            # oh crap, why are we not normalizing here?
            scores = self.get_most_similar_fast(vec_b_prime_predicted)
            result = self.process_prediction(p_test_one, scores, None, None)
            result["distances to correct cosine"] = self.embs.cmp_vectors(vec_b_prime_predicted, vec_b_prime)
            results.append(result)
        return results


class LRCos(Analogy):

    def do_test_on_pairs(self, p_train, p_test):
        results = []
        X_train, Y_train = self.gen_vec_single(p_train)
        if self.name_classifier.startswith("LR"):
            # model_regression = LogisticRegression(class_weight = 'balanced')
            # model_regression = Pipeline([('poly', PolynomialFeatures(degree=3)), ('logistic', LogisticRegression(class_weight = 'balanced',C=C))])
            model_regression = LogisticRegression(
                class_weight='balanced',
                C=self.inverse_regularization_strength)
        if self.name_classifier == "SVM":
            model_regression = sklearn.svm.SVC(
                kernel=self.name_kernel,
                cache_size=1000,
                class_weight='balanced',
                probability=True)
        model_regression.fit(X_train, Y_train)
        score_reg = model_regression.predict_proba(self.embs.matrix)[:, 1]
        for p_test_one in p_test:
            if self.is_pair_missing([p_test_one]):
                # file_out.write("{}\t{}\t{}\n".format(p_test_one[0],p_test_one[1],"MISSING"))
                continue
            vec_b = self.embs.get_vector(p_test_one[0])
            vec_b_normed = vec_b / np.linalg.norm(vec_b)
            score_sim = vec_b_normed @ self.embs._normalized_matrix.T
            scores = score_sim * score_reg
            result = self.process_prediction(p_test_one, scores, score_reg, score_sim)
            vec_b_prime = self.embs.get_vector(p_test_one[1][0])
            result["similarity b to b_prime cosine"] = float(self.embs.cmp_vectors(vec_b, vec_b_prime))
            results.append(result)
        return results
