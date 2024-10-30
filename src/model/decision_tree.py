import numpy as np
from collections import Counter

#decision tree
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
    return np.sum([-p*np.log2(p) for p in class_probability if p>0])

  def _data_entropy(self,labels):
    total_count=len(labels)
    probability=[label_count/total_count for label_count in Counter(labels).values()]
    return self._entropy(probability)

  def _partition_entropy(self,labels):
    total_count=np.sum([len(subset) for subset in labels])
    res=np.sum([self._data_entropy(subset)*(len(subset)/total_count) for subset in labels])
    return res


  def _split(self,data,idx,feature_val):
    mask_below_threshold=data[:,idx]<feature_val
    group1=data[mask_below_threshold]
    group2=data[~mask_below_threshold]

    return group1, group2

  def _select_features_to_use(self,data):
    min_part_entropy=1e9
    feature_idx=list(range(data.shape[1]-1))

    if self.numb_of_features_splitting=='sqrt':
      feature_idx_to_use=np.random.choice(feature_idx,size=int(np.sqrt(len(feature_idx))))
    elif self.numb_of_features_splitting=='log':
      feature_idx_to_use=np.random.choice(feature_idx,size=int(np.log(len(feature_idx))))
    else:
      feature_idx_to_use=feature_idx
    return feature_idx_to_use


  def _find_best_split(self,data):
    min_part_entropy = 1e9
    feature_idx_to_use=self._select_features_to_use(data)

    for idx in feature_idx_to_use:
      feature_vals=np.percentile(data[:,idx],q=np.array([25,100,25]))
      for feature_val in feature_vals:
        g1,g2=self._split(data,idx,feature_val)
        part_entropy=self._partition_entropy([g1[:,-1],g2[:,-1]])
        if part_entropy<min_part_entropy:
          min_part_entropy= part_entropy
          min_entropy_feature_val=feature_val
          min_entropy_feature_idx=idx
          g1_min,g2_min=g1,g2
    return g1_min,g2_min,min_entropy_feature_idx,min_entropy_feature_val,min_part_entropy

  def _find_label_probs(self,data):
    labels_as_integers=data[:,-1].astype(int)
    total_labels=len(labels_as_integers)
    label_probability=np.zeros(len(self.labels_in_train), dtype=float)

    for i, label in enumerate(self.labels_in_train):
      label_index=np.where(labels_as_integers==i)[0]
      if len(label_index)>0:
        label_probability[i]=len(label_index)/total_labels
    return label_probability



  def _create_tree(self,data,current_depth):
    if current_depth>self.max_depth:
      return None

    split_g1_data,split_g2_data,split_feature_idx,split_feature_val, split_entropy=self._find_best_split(data)
    label_probability=self._find_label_probs(data)

    node_entropy=self._entropy(label_probability)
    information_gain=node_entropy-split_entropy

    node=TreeNode(data,split_feature_idx,split_feature_val,label_probability,information_gain)

    if self.min_samples_leaf>split_g1_data.shape[0] or self.min_samples_leaf>split_g2_data.shape[0]:
      return node
    elif information_gain<self.min_information_gain:
      return node

    current_depth+=1
    node.left=self._create_tree(split_g1_data,current_depth)
    node.right=self._create_tree(split_g2_data,current_depth)

    return node




  def train(self,x,y):
    self.labels_in_train=np.unique(y)
    train_data=np.concatenate((x,np.reshape(y,(-1,1))),axis=1)

    self.tree=self._create_tree(train_data,current_depth=0)


  def predict(self,x):
    self.res=[]
    for data in x:
      node=self.tree

      while node:
        pred_probs=node.prediction_probs
        if data[node.feature_idx]<node.feature_val:
          node=node.left
        else:
          node=node.right
      self.res.append(pred_probs)
    return np.argmax(self.res,axis=1)
