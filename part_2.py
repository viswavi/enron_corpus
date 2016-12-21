import mysql.connector
import email
import os
import re
from dateutil import parser
from collections import defaultdict, Counter
import cPickle as pickle
import numpy as np
from scipy.spatial.distance import cosine as cos_distance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


def db_to_memory():
    cnx = mysql.connector.connect(user='root',database='enron',password='Cheerio627')
    cursor = cnx.cursor()
    query = ("select fileID, sender, recipients, subject, content from messages")
    cursor.execute(query)
    records = []
    #(fileid, sender, recipients, subject, content)
    for row in cursor:
        records.append(row)
    return records

def top_receivers(rows, num_labels=60):
    receivers = []
    for r in rows:
        receivers.extend(r[2].split(','))
    #produce list of recipients, ordered by frequency of receiving mail, descending
    counts = list(reversed(Counter(receivers).most_common()))
    top = [name for (name,counts) in counts][:num_labels]
    return top

# labels is the set of users we wish to predict
# we remove any documents that are not addressed to any of these top recipients
def build_text_dataset(rows, labels):
    labeled_documents = []
    i = 0
    recips = []
    i = 0
    tot = 0
    for r in rows:
        [fileid, sender, recipient_string, subject, body] = r
        recipients = recipient_string.split(',')
        i += 1
        tot += len(recipients)
        recips.extend(recipients)
        content = subject + " " + body
        recipients = set(recipients).intersection(labels)
        if recipients:
            labeled_documents.append((sender, content, recipients))
    print tot / float(i)
    (senders, documents, recipients) = zip(*labeled_documents)
    return ((senders, documents), recipients)



# train_text is a list of strings that we wish to classify
# train_targets is a list of the "label" email ID
def train_text_model(train_text, train_targets):
    vectorized_targets = MultiLabelBinarizer().fit_transform(train_targets)
    text_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', OneVsRestClassifier(LinearSVC(random_state=0)))])
    text_svm = text_svm.fit(train_text, vectorized_targets)
    return text_svm

def edge_list(train):
    pairs = defaultdict(int)
    nodes = []
    for (_, sender, recipients, _, _) in train:
        recipient_list = recipients.split(',')
        nodes.append(sender)
        for recipient in recipient_list:
            nodes.append(recipient)
            #make each pair undirected, so standardize the order
            pair = sender, recipient
            pairs[pair] += 1
    unique_nodes = list(set(nodes))
    return (pairs, unique_nodes)

# only need to run this function once on the train split
# which has already been done, with 'train.edgelist' written out
def write_edges_to_file(train):
    if('node2id.p' in os.listdir('.')):
        node2id = pickle.load(open('node2id.p'))
        id2node = dict((v,u) for (u,v) in node2id.items())
        return (node2id, id2node)
    (edges, nodes) = edge_list(train)
    node2id = {}
    id2node = {}
    for x in range(len(nodes)):
        node2id[nodes[x]] = x
        id2node[x] = nodes[x]
    out = open('train.edgelist','w')
    for (sender, recipient) in edges:
        send_id, recip_id = node2id[sender], node2id[recipient]
        out.write("{} {} {}\n".format(send_id, recip_id, edges[sender, recipient]))
    out.close()
    pickle.dump(node2id, open('node2id.p','w'))
    return (node2id, id2node)

def read_node_embeddings(emb_file):
    if('graph_embeddings.p' in os.listdir('.')):
        return pickle.load(open('graph_embeddings.p'))
    contents = open(emb_file).readlines()
    # first line just contains number of nodes and dimension size, so skip it
    embeddings = {}
    for line in contents[1:]:
        entries= line.split()
        node_id = int(entries[0])
        vec = np.asarray(map(float, entries[1:]))
        embeddings[id2node[node_id]] = vec
    #pickle.dump(embeddings, open('graph_embeddings.p','w'))
    return embeddings

def train_joint_model(node2vec, train_data):
    ((senders, train_text), train_targets) = train_data
    text_svm = train_text_model(train_text, train_targets)
    text_features = text_svm.decision_function(train_text)
    assert(len(senders) == len(train_text))
    augmented_train_set = []
    for sender, decision_vector, target in zip(senders, text_features, train_targets):
        if(sender in node2vec):
            graph_vector = node2vec[sender]
            vec = np.concatenate((decision_vector, graph_vector), 0)
            augmented_train_set.append((vec, target))
    (vecs, targets)= zip(*augmented_train_set)
    # format the email addresses as an indicator vector
    one_hot_targets =  MultiLabelBinarizer().fit_transform(targets)
    joint_svm = OneVsRestClassifier(LinearSVC(random_state=0)).fit(vecs, one_hot_targets)
    return text_svm, joint_svm

def predict_email_address(textsvm, joint_svm, node2vec, labels, sender, email, default_num_to_return=1):
    text_vector = text_svm.decision_function([email])[0]
    # if the sender of this email is not in our social network graph_vector
    # use the decision from text alone
    if(sender not in node2vec or True):
        decision_vector = text_vector
    else:
        graph_vector = node2vec[sender]
        vec = np.concatenate((text_vector, graph_vector), 0)
        joint_vector = joint_svm.decision_function([vec])
        decision_vector = joint_vector[0]
    indices = [i for i in range(len(decision_vector)) if decision_vector[i] >= 0]
    if(len(indices) < default_num_to_return):
        #print sender + "\n" + email
        indices = np.argsort(decision_vector)[-default_num_to_return:]
    predicted_recipients = [labels[i] for i in indices]
    return predicted_recipients


def evaluate(text_svm, joint_svm, node2vec, test_data, labels):
    ((test_senders, test_emails), test_recipients) = test_data
    precision = []
    recall = []
    errors = []
    i=0
    print len(test_senders)
    print"\n\n\n\n\n\n\n\n\n\n\n\n\n"
    print test_senders
    for (sender, email, recipients) in zip(test_senders, test_emails, test_recipients):
        # if none of the true recipients are in our list of 'labelled' recipients
        # then this is a region of the dataset that we are not attempting to model
        if(set(labels).intersection(recipients) == 0):
            continue
        emails = predict_email_address(text_svm, joint_svm, node2vec, labels, sender, email)
        relevant_and_retrieved = len(set(recipients).intersection(emails))
        if(relevant_and_retrieved == 0):
            errors.append((sender, email, recipients))
        if(len(recipients) != 0):
            precision.append((relevant_and_retrieved/float(len(emails))))
            recall.append((relevant_and_retrieved/float(len(recipients))))
    print "len(errors): {}".format(len(errors))
    print len(precision)
    return (precision, recall)

if __name__ == "__main__":
    # first, run:
    # python node2vec/src/main.py --input train.edgelist --output train.emb --weighted --directed --iter 25 --dimensions 60
    #   you can tinker with the output dimension and number of iterations
    # this will output a file called 'train.emb'
    # choose the top num_labels recipients as the labels for classification
    node2vec = read_node_embeddings('train.emb')
    rows = db_to_memory()
    # choose the top num_labels recipients as the labels for classification
    total_precs, total_recs = [], []
    for i in range(5):
        [train, test] = train_test_split(rows, test_size=0.2)
        labels = top_receivers(train, num_labels=60)
        build_data = lambda split: build_text_dataset(split, labels)
        (train_data, test_data) = map(build_data, (train,test))
        print len(train_data[1])
        ((senders, train_text), train_targets) = train_data
        text_svm, joint_svm = train_joint_model(node2vec, train_data)
        (p,r) = evaluate(text_svm, joint_svm, node2vec, test_data, labels)
        avg_prec = sum(p)/float(len(p))
        total_precs.append(avg_prec)
        avg_rec = sum(r)/float(len(r))
        total_recs.append(avg_rec)
    print total_precs
    print total_recs
    total_avg_prec = sum(total_precs)/float(len(total_precs))
    total_avg_rec = sum(total_recs)/float(len(total_recs))
    print "the average precision and recall are {} and {}".format(total_avg_prec, total_avg_rec)
