import operator
import math
import time

class Decision_Tree():
    def __init__(self):
        pass
    
    def load_data(self):
        #  [4 features]
        #    weather: (2) sunny (1) cloudy (0) rainy
        #    temperature: (2) hot (1) moderate (0) cold
        #    humidity: (1) humid (0) moderate
        #    wind speed: (1) heavy (0) weak
        #  [label]
        #    hold event? ("yes") yes ("no") no
        data = [
            [2,2,1,0,"yes"],
            [2,2,1,1,"no"],
            [1,2,1,0,"yes"],
            [0,0,0,0,"yes"],
            [0,0,0,1,"no"],
            [1,0,0,1,"yes"],
            [2,1,1,0,"no"],
            [2,0,0,0,"yes"],
            [0,1,0,0,"yes"],
            [2,1,0,1,"yes"],
            [1,2,0,0,"no"],
            [0,1,1,1,"no"]
            ]
        feature_names = ["weather", "temperature", "humidity", "wind speed"]
        return data, feature_names
    
    def cal_entropy(self, data):
        labels = [record[-1] for record in data]
        probs = [labels.count(label)/len(data) for label in set(labels)]
        entropy = sum([-p*math.log2(p) for p in probs])
        return entropy
    
    def split_data(self, data, axis, feature_type):
        # Split data by specific-axis feature
        splitted_data = []
        for record in data:
            if record[axis] == feature_type:
                splitted_record = record[:axis] + record[axis+1:]
                splitted_data.append(splitted_record)
        return splitted_data
    
    # Determine which feature is the most worth to divide data
    def choose_bestWay_toSplit(self, data):
        best_entropy = self.cal_entropy(data)
        best_IG = 0.0
        best_feature_axis = -1
        
        num_features = len(data[0]) - 1
        # Go through each feature
        for axis in range(num_features):
            # 1. Count the entropy of each feature
            entropy_of_thisFeature = 0.0
            feature_list = [record[axis] for record in data]
            for type_ in set(feature_list):
                splitted_data = self.split_data(data, axis, type_)
                entropy_of_type = (len(splitted_data)/len(data)) * self.cal_entropy(splitted_data)
                entropy_of_thisFeature += entropy_of_type
            # 2. And then, get the Information Gain (of each feature)
            IG = best_entropy - entropy_of_thisFeature
            
            # 3. Finally, judge whether this feature has the best one (also has the best IG) to "divide" data
            if IG > best_IG: 
                best_IG = IG
                best_feature_axis = axis
        return best_feature_axis
    
    def get_most_freq_label(self, labels):
        best_label = None
        max_count = 0
        for type_ in labels:
            cnt = labels.count(type_)
            if cnt > max_count:
                max_count = cnt
                best_label = type_
        return best_label
        
    def create_tree(self, data, feature_names):
        # To avoid to affect original variable
        feature_names = list(feature_names)
        labels = [record[-1] for record in data]
        
        # 1. If "all labels are same", return this type of label
        if labels.count(labels[0]) == len(labels):
            return labels[0]
        # 2. If "all features (to divide data) is out of use", means records only have labels, 
        #    return the most frequent label
        if len(data[0]) == 1:
            return self.get_most_freq_label(labels)
        
        # 3. Else
        best_feature_axis = self.choose_bestWay_toSplit(data)
        # Get the "feature name" of the feature which is the most worth to split data,
        # and delete it from "feature name list"
        best_feature_names = feature_names.pop(best_feature_axis)
        
        d_tree = {best_feature_names: {}}
        
        # Iteratively execute method 'create_tree'
        # on child nodes of decision tree whose root node determined by 'best_feature_axis'
        best_feature_types = set([record[best_feature_axis] for record in data])
        for type_ in best_feature_types:
            d_tree[best_feature_names][type_] = self.create_tree(
                self.split_data(data, best_feature_axis, type_), feature_names[:])
        
        return d_tree
    
    def predict(self, tree, feature_names, test_features):
        for key_1 in tree.keys():
            subDict = tree[key_1]
            feature_axis = feature_names.index(key_1)
            for key_2 in subDict.keys():
                if test_features[feature_axis] == key_2:
                    # case 1: the value is a dict
                    # --> it's not leaf node
                    value = subDict[key_2]
                    if type(value).__name__ == "dict":
                        candidate_label = self.predict(subDict[key_2], feature_names, test_features)
                    # case 2: the value is the leaf node
                    else:
                        candidate_label = value
        return candidate_label
    
if __name__ == "__main__":
    dt = Decision_Tree()
    data, feature_names = dt.load_data()
    # Generate a decision tree
    d_tree = dt.create_tree(data, feature_names)
    print("decision tree:\n", d_tree, "\n")
    
    # Use the decision tree to predict for test features
    test_features = [1,1,1,0]    
    label = dt.predict(d_tree, feature_names, test_features)
    result = "hold" if label=="yes" else "not hold"
    print("If the features of {} is {},".format(tuple(feature_names), tuple(test_features)))
    print("according to the prediction of decision tree,")
    print("it's most probably \"{}\" the event.".format(result))
    