# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import click as ck
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
# %matplotlib inline
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.metrics import (accuracy_score,precision_score,recall_score,confusion_matrix,
                                roc_curve,roc_auc_score, mean_absolute_error)

#dl libraraies
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import dot

# specifically for deeplearning.
from keras.layers import (Dense, Dropout, Flatten,Activation,Input,Embedding,
                            Conv2D, MaxPooling2D, BatchNormalization)

@ck.command()
@ck.option(
    '--train-data-file', '-trdf', default='movielens100k/ratings.csv',
    help='File for rating')
@ck.option(
    '--batch-size', '-bs', default=128,
    help='Batch Size')
@ck.option(
    '--epochs', '-e', default=50,
    help='# of epochs')


def main(train_data_file, batch_size, epochs):
    df=pd.read_csv(train_data_file)
    df.userId = df.userId.astype('category').cat.codes.values
    df.movieId = df.movieId.astype('category').cat.codes.values

    # creating utility matrix.
    util_df=pd.pivot_table(data=df,values='rating',index='userId',columns='movieId')
    # Nan implies that user has not rated the corressponding movie.
    util_df.fillna(0)

    users = df.userId.unique()
    movies = df.movieId.unique()

    userid2idx = {o:i for i,o in enumerate(users)}
    movieid2idx = {o:i for i,o in enumerate(movies)}

    df['userId'] = df['userId'].apply(lambda x: userid2idx[x])
    df['movieId'] = df['movieId'].apply(lambda x: movieid2idx[x])
    split = np.random.rand(len(df)) < 0.8
    train = df[split]
    valid = df[~split]

    '''
            Model1 : Input-Embedding-Similarity
    '''

    n_latent_factors=64  # hyperparamter to deal with.
    n_movies=len(df['movieId'].unique())
    n_users=len(df['userId'].unique())
    final_output = pd.DataFrame(columns=["Model", "Architecture",
                                     "TRAIN_LOSS", "TRAIN_ACC"])
    model = Model1(n_users=n_users, n_movies=n_movies, n_latent_factors=n_latent_factors)
    model.compile(optimizer=Adam(lr=1e-4),loss='mse', metrics=['accuracy'])
    History = model.fit([train.userId,train.movieId],train.rating, batch_size=batch_size,
                              epochs =epochs, validation_data = ([valid.userId,valid.movieId],
                                                                valid.rating),verbose = 1)
    plot_graph(History)
    final_output = final_output.append({"Model": 1,
                                    "Architecture": 'Input-Embedding-Similarity', 
                                    "TRAIN_LOSS": '{:.5f}'.format(History.history["loss"][epochs-1]),
                                    "TRAIN_ACC": '{:.5f}'.format(History.history["acc"][epochs-1])},
                                    ignore_index = True)

    '''
            Model2 : Input-Embedding-Dense-Dense
    '''

    n_latent_factors=50
    n_movies=len(df['movieId'].unique())
    n_users=len(df['userId'].unique())
    nn_model = Model2(n_users=n_users, n_movies=n_movies, n_latent_factors=n_latent_factors)
    nn_model.compile(optimizer=Adam(lr=1e-3),loss='mse', metrics=['accuracy'])
    epochs=20
    History = nn_model.fit([train.userId,train.movieId],train.rating, batch_size=batch_size,
                              epochs =epochs, validation_data = ([valid.userId,valid.movieId],
                                                                 valid.rating),verbose = 1)
    final_output = final_output.append({"Model": 2,
                                    "Architecture": 'Input-Embedding-Dense-Dense', 
                                    "TRAIN_LOSS": '{:.5f}'.format(History.history["loss"][epochs-1]),
                                    "TRAIN_ACC": '{:.5f}'.format(History.history["acc"][epochs-1])},
                                    ignore_index = True)
    plot_graph(History)

def Model1(n_users, n_movies, n_latent_factors):
    user_input=Input(shape=(1,),name='user_input',dtype='int64')

    user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
    user_embedding.shape

    user_vec =Flatten(name='FlattenUsers')(user_embedding)
    user_vec.shape

    movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
    movie_embedding=Embedding(n_movies,n_latent_factors,name='movie_embedding')(movie_input)
    movie_vec=Flatten(name='FlattenMovies')(movie_embedding)
    movie_vec

    sim=dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)
    model =keras.models.Model([user_input, movie_input],sim)
    model.summary()

    return model

def Model2(n_users, n_movies, n_latent_factors):
    user_input=Input(shape=(1,),name='user_input',dtype='int64')
    user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
    user_vec=Flatten(name='FlattenUsers')(user_embedding)
    user_vec=Dropout(0.40)(user_vec)

    movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
    movie_embedding=Embedding(n_movies,n_latent_factors,name='movie_embedding')(movie_input)
    movie_vec=Flatten(name='FlattenMovies')(movie_embedding)
    movie_vec=Dropout(0.40)(movie_vec)

    sim=dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)

    nn_inp=Dense(96,activation='relu')(sim)
    nn_inp=Dropout(0.4)(nn_inp)
    nn_inp=BatchNormalization()(nn_inp)
    nn_inp=Dense(1,activation='relu')(nn_inp)
    nn_model =keras.models.Model([user_input, movie_input],nn_inp)
    nn_model.summary()

    return nn_model

def plot_graph(History):
    from pylab import rcParams
    rcParams['figure.figsize'] = 10, 5
    import matplotlib.pyplot as plt
    plt.plot(History.history['loss'] , 'g')
    plt.plot(History.history['val_loss'] , 'b')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()