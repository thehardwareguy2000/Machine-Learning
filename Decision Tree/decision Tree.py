import argparse
import pandas as pd
import numpy as np


def entropy( target_col ):
    #print(target_col)
    elements,counts = np.unique(target_col,return_counts=True)
    entropy = - np.sum([(counts[i]/np.sum(counts))*(1/np.log2(log_base))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])  #multiplied with 1/log(log_base) according to formula
    #print(entropy)
    return entropy


def InfoGain(data,split_attr,target_name):
    total_entropy = entropy(data[target_name])
    vals,counts = np.unique(data[split_attr],return_counts=True)
    #cal the weighted entropy
    Entropy_wt = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attr]==vals[i]).dropna()[target_name])for i in range(len(vals))])
    #formula for information gain
    Information_Gain = total_entropy-Entropy_wt
    #print(Information_Gain)
    return Information_Gain


def attr_sel(data, target_name):
    attri = [col for col in data.columns if col != target_name]
    info_gain = [InfoGain(data, attribute, target_name) for attribute in attri]
    right_attri = attri[np.argmax(info_gain)]
    return right_attri


def ID3(data, target_name, main_node_cls=None, d=0, prev_att=None, prev_val=None):

    if d == 0: #starting condition where we print only whole entropy and rest is hardcoded
        ent = entropy(data[target_name])
        print(f"0,root,{ent},no_leaf")
    
    elif len(np.unique(data[target_name])) == 1: #we have only 1 att remaining so we prit it 
        ent = entropy(data[target_name])
        print(f"{d},{prev_att}={prev_val},{abs(ent)},{np.unique(data[target_name])[0]}")
        return np.unique(data[target_name])[0]
    
    elif entropy(data[target_name]) == 1: # impure subset high entropy
        majority_class = data[target_name].mode().iloc[0]
        print(f"{d},{prev_att}={prev_val},1,{majority_class}")
        return majority_class

    elif len(data) == 0 or len(data.columns) == 1:
        ent = ent(target_name)
        print(f"{d},{prev_att}={prev_val},{ent},{main_node_cls}")
        return main_node_cls
    
    right_attri = attr_sel(data, target_name)

    if d > 0:#depth is not yet 0 so we iterate forward and pint no_leaf as we can still iterate further
        print(f"{d},{prev_att}={prev_val},{entropy(data[target_name])},no_leaf")

    tree = {right_attri: {}}

    for value in np.unique(data[right_attri]):
        sub_data = data.where(data[right_attri] == value).dropna()
        subtree = ID3(sub_data, target_name, np.unique(sub_data[target_name])[0], d + 1, right_attri, value) #recurssive function
        tree[right_attri][value] = subtree
    return tree


par = argparse.ArgumentParser()
par.add_argument("--data", required=True)
file =par.parse_args()
dataset = pd.read_csv(file.data, header=None)
dataset.columns = [f'att{i}' for i in range(len(dataset.columns))]
log_base = dataset[dataset.columns[-1]].nunique() #settup log base to inversely multiply to convert entropy fromula from base 2 to base c
ID3(dataset, dataset.columns[-1])