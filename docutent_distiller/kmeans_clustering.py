from distiller.data_snapshot import DataSnapshot
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from importlib_resources import files
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import re
import seaborn as sns


class RelevantExpressionFinder:
    def __init__(self, X):
        """
        :param X: List of documents
        """


regex_filter_pat = re.compile(
    r"1997\.lxxx\.törvény|2019\.cxxii\.törvény|1975\.ii\.törvény|1997\.lxxxiii\.törvény|2011\.cxci\.törvény|1997\.lxxxi\.törvény")
# r"2012\.i\.törvény\.(?:16[6-9]|1[78][0-9]|19[01])\.|1992\.xxii\.törvény\.(?:16[6-9]|17[0-9]|18[0-7])\.")
positives = DataSnapshot.load_stack(
    "/home/csanyig/PycharmProjects/distiller-wk-2021-1/resources/munkaügy_preprocessed_munkaviszony_megszüntetése_positive_samples_lower_law_refs_broadened_regex.zip")
negatives = DataSnapshot.load_stack(
    "/home/csanyig/PycharmProjects/distiller-wk-2021-1/resources/munkaügy_preprocessed_munkaviszony_megszüntetése_negative_samples_lower_law_refs_broadened_regex.zip")

all_data = positives + negatives

# working on all documents
X = [doc.get("text") for doc in all_data]

# keeping lawrefs only
# X = []
# for doc in all_data:
#     X.append(" ".join([word for word in doc.get("text").split(" ") if "." in word]))

X = [doc for doc in X if regex_filter_pat.search(doc)]

RANDOM_STATE = 1
FEATURE_COUNT = 1000
topic_nums = [11]#[elem for elem in range(1, 30, 2)]
n_top_words = 50
use_filter_reduction = False
nltk_stopwords = ['után', 'ilyenkor', 'amíg', 'utolsó', 'olyan', 'maga', 'elég', 'ill.', 'milyen', 'mivel', 'akik',
                  'amit', 'ezen', 'vissza', 'ami', 'amelyek', 'lett', 'ezzel', 'sokkal', 'újra', 'össze', 'amelyekben',
                  'ebben', 'ezek', 'egyetlen', 'mint', 'saját', 'új', 'szinte', 'abban', 'mellett', 'mi', 'lehet',
                  'volna', 'sem', 'hiszen', 'nem', 'aztán', 'és', 'majd', 'általában', 'hanem', 'néhány', 'persze',
                  'az', 'amelynek', 'de', 'azután', 'közül', 'vagyis', 'szemben', 'amikor', 'át', 'meg', 'még',
                  'legyen', 'által', 'egyik', 'bár', 'vagy', 'azzal', 'valamint', 'voltam', 'keresztül', 'között',
                  'mindent', 'egész', 'belül', 'ill', 'vannak', 'ahhoz', 'emilyen', 'amely', 'mikor', 'valaki', 'egy',
                  'lenne', 'egyéb', 'volt', 'mit', 'nélkül', 'nincs', 'több', 'mintha', 'úgy', 'igen', 'jó', 'legalább',
                  'kívül', 'tovább', 'jól', 'pedig', 'itt', 'ez', 'erre', 'kellett', 'ne', 'vele', 'nekem', 'nagy',
                  'ison', 'rá', 'annak', 'a', 'benne', 'melyek', 'eddig', 'azt', 'e', 'mert', 'azok', 'cikkek',
                  'cikkeket', 'voltunk', 'viszont', 'azonban', 'csak', 'való', 'én', 'sok', 'egyes', 'továbbá', 'már',
                  'elsõ', 'valami', 'így', 'fel', 'magát', 'talán', 'más', 'tehát', 'amelyet', 'ezért', 'aki', 'ehhez',
                  'ugyanis', 'õ', 'ellen', 'néha', 'õk', 'mely', 'lesz', 'ilyen', 'számára', 'van', 'neki', 'lehetett',
                  'elõ', 'sokat', 'most', 'nagyon', 'teljes', 'mindig', 'lenni', 'jobban', 'ennek', 'keressünk',
                  'arról', 'hogy', 'ott', 'be', 'õket', 'miért', 'hogyan', 'másik', 'illetve', 'ezt', 'elõször',
                  'ekkor', 'újabb', 'szerint', 'el', 'voltak', 'míg', 'minden', 'egyre', 's', 'felé', 'utána', 'alatt',
                  'semmi', 'azon', 'nagyobb', 'ahogy', 'kell', 'vagyok', 'elõtt', 'amolyan', 'ismét', 'éppen',
                  'amelyeket', 'ahol', 'akkor', 'arra', 'ki', 'mindenki', 'azért', "alperes", "felperes", "bíróság",
                  "is", "r"]

VECTORIZER_SETTINGS = {"min_df": 10, "token_pattern": r"\s?(.+?)\s",
                       "stop_words": nltk_stopwords, "ngram_range": (1, 2)}  # , "max_features": FEATURE_COUNT


def vectorize_training_data(X, **kwargs):
    tfidf_vectorizer = TfidfVectorizer(**kwargs)
    # vectorizing training data
    X = tfidf_vectorizer.fit_transform(X)
    return X, tfidf_vectorizer


def get_top_features_cluster(tf_idf_array, prediction, n_feats, tf_idf_vectorizer):
    labels = np.unique(prediction)
    dfs = []
    best_features_list = []
    for label in labels:
        id_temp = np.where(prediction == label)  # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis=0)  # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top 20 scores
        features = tf_idf_vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns=['features', 'score'])
        dfs.append(df)
        best_features_list.append(best_features)
    return dfs, best_features_list


def plot_entropies(results_list):
    for idx, entropy_words in enumerate(results_list):
        # reversing order
        entropy_words = entropy_words[::-1]

        words = [word[0] for word in entropy_words]

        entropies = [word[1] for word in entropy_words]
        y_pos = np.arange(len(words))
        max_entropy = max(entropies)

        entropies = [max_entropy - word[1] for word in entropy_words]
        plt.barh(y_pos, entropies, tick_label=words)
        plt.title("Most relevant words for Cluster Nr. {}".format(idx+1))
        plt.xlabel("Relevance score")
        plt.ylabel("Words")
        plt.show()


def plot_features(dfs):
    fig = plt.figure(figsize=(14, 12))
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i + 1)
        ax.set_title("Cluster: " + str(i), fontsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.barh(x, df.score, align='center', color='#40826d')
        yticks = ax.set_yticklabels(df.features)
    plt.show();


def plot_sns(dfs, nr_features):
    for i in range(len(dfs)):
        plt.figure(figsize=(8, 6))
        sns.barplot(x='score', y='features', orient='h', data=dfs[i][:nr_features])


X_vectorized, tfidf_vectorizer = vectorize_training_data(X, **VECTORIZER_SETTINGS)


def plot_clusters(kmeans, Y_sklearn, use_colors=False):
    if use_colors:
        predicted_values = kmeans.predict(Y_sklearn)

        plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=50, cmap='viridis')

        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.6)
    else:
        plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], s=50)
    plt.show()


def get_entropy_based_best_candidates(results_list):
    topic_union_set = set([elem[0] for res in results_list for elem in res])
    topic_sets = []
    for res in results_list:
        topic_sets.append(set([elem[0] for elem in res]))
    res_dict = {}
    res_lst = []
    for elem in topic_union_set:
        res_dict[elem] = []
        for res in results_list:
            if not elem in [wordscore[0] for wordscore in res]:
                res_dict[elem].append(0)
            else:
                for word, score in res:
                    if word == elem:
                        res_dict[elem].append(score)
    for topic_set in topic_sets:
        topic_entropy = [(word, entropy(res_dict.get(word))) for word in topic_set]
        topic_entropy = sorted(topic_entropy, key=lambda tup: tup[1])
        res_lst.append(topic_entropy)
    return res_lst


print(X_vectorized.shape)

inertias = []
sklearn_pca = PCA(n_components=200)
Y_sklearn = sklearn_pca.fit_transform(X_vectorized.toarray())
# Y_sklearn_tsne = TSNE(n_components=200, init='random').fit_transform(X_vectorized.toarray())
for topic_num in topic_nums:
    print("Number of clusters: {}".format(topic_num))
    kmeans = KMeans(n_clusters=topic_num, n_jobs=6, max_iter=600, random_state=RANDOM_STATE).fit(Y_sklearn)
    # kmeans_tsne = KMeans(n_clusters=topic_num, n_jobs=6, max_iter=600, random_state=RANDOM_STATE).fit(Y_sklearn_tsne)
    inertia = kmeans.inertia_
    inertias.append(inertia)
    print("Inertia: {}".format(inertia))
    # PCA
    dfs, best_features_list = get_top_features_cluster(X_vectorized.toarray(), kmeans.predict(Y_sklearn),
                                                       n_top_words,
                                                       tfidf_vectorizer)
    entropy_features = get_entropy_based_best_candidates(best_features_list)
    for elem in entropy_features:
        print(elem)
    plot_entropies(entropy_features)
    # #TSNE
    # dfs, best_features_list = get_top_features_cluster(X_vectorized.toarray(), kmeans_tsne.predict(Y_sklearn_tsne),
    #                                                    n_top_words,
    #                                                    tfidf_vectorizer)
    # entropy_features = get_entropy_based_best_candidates(best_features_list)
    # for elem in entropy_features:
    #     print(elem)
    # # plot_sns(dfs, n_top_words)
    #
    plot_clusters(kmeans, Y_sklearn, use_colors=True)
    # #
    # plot_clusters(kmeans_tsne, Y_sklearn_tsne, use_colors=True)

print("Inertias: {}".format(inertias))
fig, (ax1) = plt.subplots(1, 1)
ax1.plot(topic_nums, inertias, label="Inertia")
ax1.legend()
ax1.set_xlabel("Number of clusters")
ax1.set_ylabel("Sum of squared distances of samples to their closest cluster center")
plt.show()
