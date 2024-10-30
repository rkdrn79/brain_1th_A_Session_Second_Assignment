import numpy as np
from collections import Counter

class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self,max_depth=4, min_samples_leaf=1,min_information_gain=0.0,
                 numb_of_features_splitting=None,amount_of_say=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say

    def _entropy(self,class_probability):
        # TODO: Step 1. 엔트로피 계산을 구현하세요
        # - 수식: -Σ(p * log2(p)), p > 0인 경우에만 계산
        # - np.log2와 np.sum을 사용하세요
        pass

    def _data_entropy(self,labels):
        total_count = len(labels)
        probability = [label_count/total_count for label_count in Counter(labels).values()]
        return self._entropy(probability)

    def _partition_entropy(self,labels):
        # TODO: Step 1. 전체 데이터 수를 계산하세요
        # TODO: Step 2. 각 부분집합의 엔트로피와 가중치를 계산하여 총합을 구하세요
        # Hint: 가중치는 부분집합의 크기 / 전체 크기
        pass

    def _split(self,data,idx,feature_val):
        mask_below_threshold = data[:,idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2

    def _select_features_to_use(self,data):
        min_part_entropy = 1e9
        feature_idx = list(range(data.shape[1]-1))

        if self.numb_of_features_splitting == 'sqrt':
            feature_idx_to_use = np.random.choice(feature_idx,size=int(np.sqrt(len(feature_idx))))
        elif self.numb_of_features_splitting == 'log':
            feature_idx_to_use = np.random.choice(feature_idx,size=int(np.log(len(feature_idx))))
        else:
            feature_idx_to_use = feature_idx
        return feature_idx_to_use

    def _find_best_split(self,data):
        # TODO: Step 1. 최소 엔트로피 초기값과 사용할 특성 인덱스를 가져오세요
        # TODO: Step 2. 각 특성에 대해 가능한 분할 지점을 찾으세요 (25%, 50%, 75% 지점 사용)
        # TODO: Step 3. 각 분할 지점에 대한 엔트로피를 계산하고 최소값을 찾으세요
        # TODO: Step 4. 최적의 분할 결과를 반환하세요
        pass

    def _find_label_probs(self,data):
        labels_as_integers = data[:,-1].astype(int)
        total_labels = len(labels_as_integers)
        label_probability = np.zeros(len(self.labels_in_train), dtype=float)

        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers==i)[0]
            if len(label_index)>0:
                label_probability[i] = len(label_index)/total_labels
        return label_probability

    def _create_tree(self,data,current_depth):
        # TODO: Step 1. 최대 깊이 체크
        # TODO: Step 2. 최적 분할 지점 찾기
        # TODO: Step 3. 현재 노드의 레이블 확률과 정보 이득 계산
        # TODO: Step 4. 노드 생성
        # TODO: Step 5. 종료 조건 검사 (min_samples_leaf, min_information_gain)
        # TODO: Step 6. 재귀적으로 서브트리 생성
        pass

    def train(self,x,y):
        self.labels_in_train = np.unique(y)
        train_data = np.concatenate((x,np.reshape(y,(-1,1))),axis=1)
        self.tree = self._create_tree(train_data,current_depth=0)

    def predict(self,x):
        # TODO: Step 1. 결과를 저장할 리스트를 초기화하세요
        # TODO: Step 2. 각 데이터 포인트에 대해 트리를 순회하며 예측하세요
        # TODO: Step 3. 최종 예측 결과를 반환하세요
        # Hint: np.argmax를 사용하여 가장 높은 확률의 클래스를 선택하세요
        pass