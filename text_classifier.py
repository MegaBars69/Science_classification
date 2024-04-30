from Utils.train_model import *
from name_to_titles import *

def classify(text):
    clf_filename = 'weights/' + 'Areas.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))

    vec_filename = 'weights/' + 'Areas_vectorizer.pkl'
    vectorizer = pickle.load(open(vec_filename, 'rb'))

    area_pred = nb_clf.predict(vectorizer.transform([text]))

    cat_filename = 'weights/' + area_pred[0] + '.pkl'
    nb_cat_clf = pickle.load(open(cat_filename, 'rb'))

    cat_vec_filename = 'weights/' + area_pred[0] +'_vectorizer.pkl'
    cat_vectorizer = pickle.load(open(cat_vec_filename, 'rb'))

    category_pred = nb_cat_clf.predict(cat_vectorizer.transform([text]))

    return area_pred[0] + '(' + category_pred[0] + ')'

def categorization(path):
    dir_list = os.listdir(path)
    for name in dir_list:
        docs = setup_docs(path + '/' + name)
        name = name.replace('.csv', '')
        train_classifier(docs, name)  

def NameToCategories(name, Small = False):
    NameToFile(name,Small)
    file = open('titles.txt', 'r', encoding="utf-8")
    file = file.read()
    file = file.split('\n')
    with open('output.txt', 'w', encoding="utf-8") as f:
        for title in file:
            if(len(title) > 1):
                f.write(classify(title) + '\n')
                print(classify(title).replace('_',' ') + ' – ' + title + '\n')

def ListToCategory(list):
    for item in list:
        print(classify(item).replace('_',' ') + ' – ' + item + '\n')

if __name__ == '__main__':
    """"""
    #titles = ['Nijenhuis geometry II: Left-symmetric algebras and linearization problem for Nijenhuis operators', 'Linear induction machines for electrodynamic separation of non-ferrous metals', 'Linear induction machines with the opposite direction travelling magnetic fields for induction heating', 'Application of linear inductors with opposite direction travelling magnetic fields in eddy-current separators', 'Investigation of Double-Purpose Linear Induction Motors', 'Orthogonal separation of variables for spaces of constant curvature', 'Applications of Nijenhuis geometry II: maximal pencils of multi-Hamiltonian structures of hydrodynamic type', 'When a (1,1)-tensor generates separation of variables of a certain metric', 'Applications of Nijenhuis Geometry V: Geodesic Equivalence and Finite-Dimensional Reductions of Integrable Quasilinear Systems', 'Applications of Nijenhuis geometry IV: Multicomponent KdV and Camassa–Holm equations', 'Comparison of Electrodynamic Separators with a Traveling Magnetic Field with Different Designs of Inductors', 'Applications of Nijenhuis Geometry III: Frobenius Pencils and Compatible Non-homogeneous Poisson Structures', 'Applications of Nijenhuis geometry: non-degenerate singular points of Poisson–Nijenhuis structures', 'Remote detection and recognition of bio-aerosols by laser-induced fluorescense lidar: practical implementation and field tests', 'Sample title', 'Prevalence of causative agents of respiratory infections in cats and dogs in Russia', 'On the Linearization of Certain Singularities   of Nijenhuis Operators']
    #ListToCategory(titles)
    #NameToCategories('Konyaev',True) 
    #print(AuthorRank('Kudryavtseva+Elena',True))
    #print(AuthorRank('Andrey+Konyaev',True))
    