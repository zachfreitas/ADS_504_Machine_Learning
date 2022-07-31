current_folder = os.getcwd()


# data = sklearn.datasets.load_svmlight_files(['labeledBow.feat'])

def load_aclImdb_file(filename):
    ''' filename is id_rating

        returns a dict object, {id, rating, text}
    '''
    id,rating,txt = re.split('[_.]',filename)
    with open(filename,'r',encoding="utf8") as f:
        text_of_file = f.read()

    return {'id':id,'rating':rating,'text':text_of_file}


data_dicts=[]
for cat in ['neg','pos']:

    os.chdir(os.path.join('C:\\Users\\vicen\\Dropbox\\ML TEACHING\\usd-ml-course\\Untitled Folder 1\\aclImdb\\train\\',cat))

    for file in os.listdir('.'):
        dd = load_aclImdb_file(file)
        dd.update({'cat':cat})
        data_dicts.append(dd)

acl_imdb_data = pd.DataFrame(data_dicts)

# first take on this...
corpus = acl_imdb_data['text']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# MemoryError: Unable to allocate 12.5 GiB for an array with shape (76065, 22110) and data type int64

features = vectorizer.get_feature_names()
word_count_df=pd.DataFrame.sparse.from_spmatrix(X,columns=features,index=corpus.index)

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(acl_imdb_data['cat'])
labels = pd.Series(lb.transform(acl_imdb_data['cat']).reshape(-1))

labels.name = 'LABELS_FOR_CLASSIFICATION'
