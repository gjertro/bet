# script for visualizing olympic bet

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import LabelEncoder

# set seaborn style
sns.set()

def get_bet_data():

    # set seaborn style
    sns.set()

    # import data to dataframe
    file = 'alle2.xlsx'

    columns =  list(np.arange(8,126,3))
    print('Loading data from: ',file)

    # read form excel
    df = pd.read_excel(file, sheet_name='Totaloversikt', usecols=columns, skiprows=1)

    # transpose to have rowvize datapoints - one person in each row
    df = df.transpose()

    # add column for points
    df['Points'] = 0

    # remove column that contains "svar," drop coluns with nan
    df = df.drop(labels = 2, axis = 1)
    df = df.dropna(axis = 1, how = 'all')

    # rename column headers
    i = 1
    for col in df.columns:
        df.rename(columns = {col: "sporsmal_" + str(i)}, inplace = True)
        i = i + 1
    df.rename(columns = {"sporsmal_186":"Points"}, inplace = True)

    df_points = pd.read_excel(file, sheet_name='Totaloversikt', usecols="E").transpose()
    df_fasit = pd.read_excel(file, sheet_name='Totaloversikt', usecols="F").transpose()
    df_fasit = df_fasit.drop(labels = 0, axis = 1)
    return df, df_fasit, df_points

def plot_histogram(df, spm_id):

    ans_counts = Counter(df.iloc[:,spm_id])
    _ = pd.DataFrame.from_dict(ans_counts, orient='index').plot(kind = 'bar')
    plt.show()

def encode_df(df, type = ''):

    if type == 'numeric':
        # encoder - get numeric categories
        lb_make = LabelEncoder()
        df_num = pd.DataFrame()
        for col in df.columns[:-1]:
            df_num[col] = lb_make.fit_transform(df[col].astype('str'))


    elif type == 'dummies':
        # encoder - get dummy variables
        list_of_dummies = []
        for col in df.columns[:73]:
            dummy = pd.get_dummies(df[col]).rename(columns=lambda x: 'Category_' + str(x))
            list_of_dummies.append(dummy)
        df_num = pd.concat(list_of_dummies, axis = 1)
    else:
        print('Error in encode_df: wrong choise of type')
        df_num = pd.DataFrame([1])

    df_num.index = df.index
    df_num = df_num.fillna(0)

    return df_num

def apply_tsne(df):
    # Import TSNE
    from sklearn.manifold import TSNE
    print('Creating t-SNE')
    # Create a TSNE instance: model
    model = TSNE(learning_rate=50)

    # Apply fit_transform to normalized_movements: tsne_features
    tsne_features = model.fit_transform(df.values)

    # Select the 0th feature: xs
    xs = tsne_features[:,0]

    # Select the 1th feature: ys
    ys = tsne_features[:,1]

    # Scatter plotx
    plt.scatter(xs,ys, alpha = 0.5)

    # Annotate the points
    for x, y, name in zip(xs, ys, df.index):
        plt.annotate(name, (x, y), fontsize=5, alpha=0.75)
    plt.show()

def apply_kmeans(df, n_clusters = 10):

    from sklearn.cluster import KMeans
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA
    import matplotlib.cm

    pca = PCA(n_components=2)
    pca_answers = pca.fit_transform(df.values)
    #pca_kernels = pca.transform(kmeans.cluster_centers_)

    # Create a KMeans model with n_clusters: kmeans
    kmeans = KMeans(n_clusters=n_clusters)

    # Fit kmeans
    # kmeans.fit(df)
    kmeans.fit(pca_answers)

    # Predict the cluster labels: labels
    labels = kmeans.predict(pca_answers)

    # Create a DataFrame aligning labels and participant: df
    df_result_kmeans = pd.DataFrame({'labels': labels, 'Participant': df.index})

    # Display df sorted by cluster label
    print(df_result_kmeans.sort_values('labels'))

    #print(kmeans.cluster_centers_)
    #print(kmeans.labels_)
    #print(kmeans.inertia_)
    #print(labels)

    plt.scatter(pca_answers[:, 0], pca_answers[:,1], alpha = 0.9, c=labels)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='D')

    # Annotate the points
    for x, y, name in zip(pca_answers[:,0], pca_answers[:,1], df.index):
        plt.annotate(name, (x, y), fontsize=5, alpha=0.75)
    plt.show()

def apply_heatmap(df,df_num,spm_start = 0, spm_end = 5 ):
    print('Creating heatmap..')
    _ = sns.heatmap(df_num.iloc[:, spm_start:spm_end], annot = df.iloc[:,spm_start:spm_end], cmap ="Paired", cbar = False, fmt = '', yticklabels=True)
    plt.show()

def apply_hierarchical_clustering(df):
    print('hierarchical clustering..')

    # Perform the necessary imports
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt

    # Calculate the linkage: mergings
    mergings = linkage(df, method='complete')

    # Plot the dendrogram, using varieties as labels
    dendrogram(mergings,
               labels=df.index,
               leaf_rotation=90,
               leaf_font_size=6,
               )
    plt.show()

def calculate_points(df, df_points, df_fasit, plot_leaderboard = True):
    # calculate points
    n_rows, n_cols = df.shape
    m_rows, m_cols = df_fasit.shape

    for row in range(n_rows):
        pnts = 0
        for col in range(m_cols):
            if df.iloc[row, col] == df_fasit.iloc[0, col]:
                pnts = pnts + df_points.iloc[0, col]
        df.iloc[row, 185] = pnts

    if plot_leaderboard:
        df = df.sort_values(by = 'Points', ascending=False)
        print("Plotting leaderboard")
        ax = sns.barplot(x=df['Points'], y=df.index)
        plt.show()

    return df

def calculate_position_leaderboard(df, df_points, df_fasit, plot = True, persons_to_plot = []):
    list_of_leaderbords = []
    # calculate leaderboard after each question
    for n_col in range(df_fasit.shape[1] - 1):
        df_i = calculate_points(df, df_points, df_fasit.iloc[:, :n_col + 1], plot_leaderboard = False)
        idx = 40 - np.argsort(df_i['Points'].values)
        list_of_leaderbords.append(idx)

    df_leaderboard = pd.DataFrame(list_of_leaderbords)
    df_leaderboard.columns= df.index

   #create plot
    if plot:
        for person in df_leaderboard.columns:
            if person in persons_to_plot:
                df_leaderboard.loc[:, person].plot(marker='o', label=person)
        plt.ylim(max(idx), 0)
        plt.legend()
        plt.show()

    #from bokeh.plotting import figure, show

    #p = figure()
    #p.multi_line(xs=[df_leaderboard.index.values]*len(df_leaderboard.columns),
    #             ys=[df_leaderboard[name].values for name in df_leaderboard],
    #             legend=[name for name in df_leaderboard.columns])
    #show(p)

### Main
# read data
df, df_fasit, df_points = get_bet_data()

# encode answers
df_num = encode_df(df, type = 'numeric')
df_dum = encode_df(df, type = 'dummies')

# heatmap
#apply_heatmap(df, df_num, spm_start = 100, spm_end = 105 )

# calculate points
df = calculate_points(df, df_points, df_fasit, plot_leaderboard = True)

# plot development of leaderboard
calculate_position_leaderboard(df, df_points, df_fasit, plot = True, persons_to_plot = ['Gjert', 'Ida', 'Jon Inge'])

# hierarcical clustering
#apply_hierarchical_clustering(df_dum)

# kmeans
#apply_kmeans(df_dum, n_clusters = 15)
#apply_kmeans(df_num, n_clusters = 10)


# t-SNE
#apply_tsne(df_dum)

print('bye')
