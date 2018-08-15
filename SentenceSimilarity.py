import io
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

class SentenceSimilarity:

    def __init__(self, model_path):
        self.model_path = model_path

    def load_vectors(self):
        fin = io.open(self.model_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        cpt = 0
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
            cpt += 1
            if cpt >= 30000:
                break
        return data

    def word_vector(self, word, debug=False):
        model = self.load_vectors()
        if debug:
            print("word :", word)
        try:
            data = list(model[word])
            vector = []
            for i in data:
                vector.append(i)
            return np.asarray(vector)
        except KeyError:
            return 0

    def word_weight(self, dic, word, question, len_json):
        dic_question = Counter(re.findall(r'\w+', question))
        tf = dic_question[word]
        if dic[word] != 0:
            weight = tf * np.log(len_json/dic[word])
            return weight
        return 0

    def sentence_weight(self, sentence, definitions):
        stopWords = set(stopwords.words('english'))
        sentence = sentence.replace("_", " ").replace("-", " ")
        question = sentence.lower()
        questionFiltered = []
        question1 = word_tokenize(question, language='english')
        corpus = ''
        for w in definitions:
            if w is not None:
                corpus = corpus + ' ' + w
        if corpus is not None:
            word_count = Counter(re.findall(r'\w+', corpus.lower()))
        weight_list = []
        for i, j in enumerate(question1):
            weight_list.append(self.word_weight(word_count, question1[i], question1, len(definitions)))
        return weight_list

    def sentence_vector_array(self, sentence, table_name=None):
        stopWords = set(stopwords.words('english'))
        sentence = sentence.replace("_", " ").replace("-", " ").lower()
        words = word_tokenize(sentence)
        wordsFiltered = []
        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)
        data = [self.word_vector(v) for v in wordsFiltered]
        return data

    def vector_similarity(self, vector1, vector2, weight1, weight2):
        assert len(vector1) == len(weight1)
        assert len(vector2) == len(weight2)
        for k, j in enumerate(weight1):
            if vector1[k] is not None:
                vector1[k] *= weight1[k]
            else:
                vector1[k] = 0
        for w, i in enumerate(weight2):
            if vector2[w] is not None:
                vector2[w] *= weight2[w]
            else:
                vector2[w] = 0
        vector1_sum = 0
        vector2_sum = 0
        for i in vector1:
            vector1_sum += i
        for j in vector2:
            vector2_sum += j
        if len(weight1) != 0 and len(weight2) != 0:
            vector1_sum = vector1_sum / len(weight1)
            vector2_sum = vector2_sum / len(weight2)
        else:
            return 0

        if not isinstance(vector2_sum, float):
            sum_vec1_vect2 = [vector1_sum[i]*vector2_sum[i] for i, j in enumerate(vector1_sum)]
            sum_vec1_vect2 = np.sum(sum_vec1_vect2)

            norm_vec1 = [np.power(vector1_sum[i], 2) for i, j in enumerate(vector1_sum)]
            norm_vec1 = np.sqrt(np.sum(norm_vec1))
            norm_vec2 = [np.power(vector2_sum[i], 2) for i, j in enumerate(vector2_sum)]
            norm_vec2 = np.sqrt(np.sum(norm_vec2))
            if norm_vec1 != 0 and norm_vec2 != 0:
                cosinus_similarity = sum_vec1_vect2 / (norm_vec1*norm_vec2)
                return cosinus_similarity
            return 0
        else:
            return 0

    def vectorize_for_picklelize(self, labels, definitions):
        vector_array = []
        for i, j in enumerate(labels):
            if definitions[i] is not None:
                vector2 = self.sentence_vector_array(str(definitions[i]))
                weight2 = self.sentence_weight(str(definitions[i]), definitions)
                vector_array.append([labels[i], weight2, vector2])
            else:
                vector2 = np.zeros((1, 300))
                weight2 = np.zeros((1, 1))
                vector_array.append([labels[i], weight2, vector2])
        return vector_array

    def classify_text(self, labels, sentence, definitions, path, debug=False):
        weight1 = self.sentence_weight(sentence, definitions)
        vector1 = self.sentence_vector_array(sentence)
        similarity_score = 0
        similarity_sorted = []
        indice = 0
        vector_array = []
        try:
            with open(path, 'rb') as handle:
                vectors = pickle.load(handle)
            for i, j in enumerate(vectors):
                if definitions[i] is not None:
                    similarity = self.vector_similarity(vector1, vectors[i][2], weight1, vectors[i][1])
                    similarity_sorted.append([np.sqrt(np.power(similarity, 2)), labels[i], definitions[i]])
                else:
                    similarity = 0
                if similarity > similarity_score:
                    similarity_score = similarity
                    indice = i
                if debug:
                    print(similarity, labels[i])
        except (FileExistsError, FileNotFoundError):
            for i, j in enumerate(labels):
                if definitions[i] is not None:
                    weight2 = self.sentence_weight(definitions[i], definitions)
                    vector2 = self.sentence_vector_array(definitions[i])
                    vector_array.append([labels[i], weight2, vector2])
                    similarity = self.vector_similarity(vector1, vector2, weight1, weight2)
                else:
                    similarity = 0
                if similarity > similarity_score:
                    similarity_score = similarity
                    indice = i
                if debug:
                    print(similarity, labels[i])
            with open(path, 'wb') as handle:
                pickle.dump(vector_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Result column : ", labels[indice])
        similarity_sorted.sort(reverse=True)
        return labels[indice], indice, similarity_sorted

model = SentenceSimilarity('GoogleNews-vectors-negative300.bin')
print(model.word_vector('work'))

