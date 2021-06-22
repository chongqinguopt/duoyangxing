class MMRModel(object):
    # def __init__(self,item_score_dict,similarity_matrix,lambda_constant,topN):
    #     self.item_score_dict = item_score_dict
    #     self.similarity_matrix = similarity_matrix
    #     self.lambda_constant = lambda_constant
    #     self.topN = topN
    def __init__(self, **kwargs):
        self.lambda_constant = kwargs['lambda_constant']
        self.topN = kwargs['topN']

    def build_data(self):
        sorce = np.random.random(size=(self.topN))
        item = np.random.randint(1, 1000, size=self.topN)
        self.item_score_dict = dict()
        for i in range(len(item)):
            self.item_score_dict[i] = sorce[i]
        item_embedding = np.random.randn(self.topN, self.topN)  # item的embedding
        item_embedding = item_embedding / np.linalg.norm(item_embedding, axis=1, keepdims=True)
        sim_matrix = np.dot(item_embedding, item_embedding.T)  # item之间的相似度矩阵
        self.similarity_matrix = sorce.reshape((self.topN, 1)) * sim_matrix * sorce.reshape((1, self.topN))

    def mmr(self):
        s, r = [], list(self.item_score_dict.keys())
        while len(r) > 0:
            score = 0
            select_item = None
            for i in r:
                sim1 = self.item_score_dict[i]
                sim2 = 0
                for j in s:
                    if self.similarity_matrix[i][j] > sim2:
                        sim2 = self.similarity_matrix[i][j]
                equation_score = self.lambda_constant * sim1 - (1 - self.lambda_constant) * sim2
                if equation_score > score:
                    score = equation_score
                    select_item = i
            if select_item == None:
                select_item = i
            r.remove(select_item)
            s.append(select_item)
        return (s, s[:self.topN])[self.topN > len(s)]

if __name__ == "__main__":
    kwargs = {
        'lambda_constant': 0.5,
        'topN': 5,
    }
    dpp_model = MMRModel(**kwargs)
    dpp_model.build_data()
    print(dpp_model.mmr())
