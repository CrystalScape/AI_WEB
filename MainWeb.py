import keras.layers
import keras.preprocessing
import keras.regularizers
import keras.utils
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st 
import streamlit_option_menu as som
import PIL.Image as img
import numpy as np
import os

# Machine Learning

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error , accuracy_score , classification_report

# Deep Learning 

import tensorflow as tf 
import keras
from keras.layers import (
    Conv2D ,
    Flatten ,
    BatchNormalization,
    Dense,
    MaxPooling2D , 
    add,
    Dropout as Drp, 
    LeakyReLU, 
    Softmax as sfx , 
    ReLU)

# Deep Reinforcmen Learning

import torch 
from torch.nn import(
    Linear , 
    Softmax, 
    Embedding,
    Dropout,
    Module,
    ModuleList,
    GELU,
    LayerNorm)

style = """<style>
* {
   font-family: 'Times New Roman', Times, serif; 
}
h2 {
    text-align: center; 
    color: black;
    }
    </style>"""
st.markdown(style, unsafe_allow_html=True)
logo1 = img.open('Logo_web2.png')
logo1 = np.array(logo1)
st.image(logo1)
sb = st.sidebar
with sb : 
    col1 , col2 = st.columns((50,50))
    with col1 : 
        logo2 = img.open('logo UNP.jpg').resize((120,120))
        logo2 = np.array(logo2)
        st.image(logo2)
    with col2 : 
        st.title('Universitas Nusantara PGRI Kediri')
    opt = som.option_menu('Select Menu' , ['Main Menu' , 'About Creator'])
if opt == 'Main Menu' : 
    st.header('All Exsperiment')
    st.write('''Web Ini adalah web Laboratorium Online dimana Orang dapat melihat Hasil Exsperiment dari tim think laboratory ,
         jadi silahkan memilih Algoritma di bawah untuk melihat hasil exsperimen , Jika masih terdapat ketidak akuratan pada prediksi 
         itu adalah hal wajar mengingat ini adalah Model yang sedang Di kembangkan''')
    opt2 = som.option_menu(menu_title=None , options=['Machine Learning' , 'Deep Learning' , 'Deep Reinforcmen Learning'] ,
                           orientation='horizontal')
    
    if opt2 == 'Machine Learning' : 
        
        st.write('Powerd By Sklearn')
        st.header('Gaussian Naive Bayes for classification')
        
        # Pemrosesan data
        st.header('1.1 Data Preprocessing')
        df_nb = pd.read_csv('Obesity Classification.csv')
        st.write('Perview Data Raw')
        st.table(df_nb[:10])
        st.write(f'Nan Value : {df_nb.isna().sum()}')
        
        # Ploting data 
        cl1 , cl2 = st.columns((5 , 5))
        with cl1 : 
            fig , ax = plt.subplots(1,1)
            st.write('Jumlah Laki laki dan Perempuan Obey')
            male_ob = len(df_nb[(df_nb['Gender'] == 'Male') & (df_nb['Label'] == 'Overweight')])
            fimale_ob = len(df_nb[(df_nb['Gender'] == 'Female') & (df_nb['Label'] == 'Overweight')])
            ax.bar(['Male' , 'Female'] , [male_ob , fimale_ob])
            st.pyplot(fig)
        with cl2 : 
            fig , ax = plt.subplots(1,1)
            st.write('Jumlah Laki laki dan Perempuan Normal Weight')
            male_nw = len(df_nb[(df_nb['Gender'] == 'Male') & (df_nb['Label'] == 'Normal Weight')])
            fimale_nw = len(df_nb[(df_nb['Gender'] == 'Female') & (df_nb['Label'] == 'Normal Weight')])
            ax.bar(['Male' , 'Female'] , [male_nw , fimale_nw])
            st.pyplot(fig)
        plt.show()
        
        st.write('Processed data')
        Le_g = LabelEncoder().fit(df_nb['Gender'])
        Le_L = LabelEncoder().fit(df_nb['Label'])
        df_nb['Gender'] = Le_g.transform(df_nb['Gender'])
        df_nb['Label'] = Le_L.transform(df_nb['Label'])
        st.table(df_nb[:10])
        
        # Prediksi 
        st.header('1.2 Train Model')
        SS = StandardScaler().fit(df_nb.drop(columns=['Label'  , 'ID']))
        X = SS.transform(df_nb.drop(columns=['Label' , 'ID']))
        y = df_nb['Label']
        test_size = float(st.number_input('Test Size'))
        random_state = int(st.number_input('Random State'))
        button1 = st.button('Train!')
        x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=test_size , random_state=random_state , shuffle=False)
        Models = GaussianNB().fit(x_train , y_train)
        if button1 : 
            predic = Models.predict(x_test)
            st.write(f'MAE : {np.round(mean_absolute_error(y_test , predic) , decimals=2)} | Acc : {np.round(accuracy_score(y_test , predic) , decimals=2)}')
            dfport = pd.DataFrame(classification_report(y_test , predic , output_dict=True)).transpose()
            st.dataframe(dfport)
        
        # Input 
        st.header('1.3 Prediksi')
        age = int(st.number_input('Umur'))
        Genders = st.radio('Genders' , ['Male' , 'Female'])
        Genders = Le_g.transform([Genders])[0]
        Tinggi = float(st.number_input('Tinggi'))
        Lebar_pinggang = float(st.number_input('Berat Badan'))
        BMI = float(st.number_input('Body Mass Index'))
        inpts = np.array([age , Genders , Tinggi , Lebar_pinggang , BMI]).reshape(1 ,-1)
        inpts = SS.transform(inpts)
        prediks = Models.predict(inpts)
        button = st.button('Predict')
        if button : 
            prediks = Le_L.inverse_transform(prediks)
            st.write(f'Kamu Mengalami {prediks[0]}')
        
    if opt2 == 'Deep Learning' : 
        
        st.write('Powerd By Tensorflow & Keras')
        st.header('ResNet Combiner Model For Image Net (RCNet)')
        
        # Import Gambar
        st.header('1.1 Perview Dataset')
        
        # Perview Alimage 
        st.write('Dataset Yang saya Pakai adalah Dataset Tentang clasifikasi Cuaca Dan berikut Perviwe nya')
        cols1 , cols2= st.columns((5 ,5))
        with cols1 : 
            cld = img.open('Winter\Cloudy\cloudy3.jpg')
            cld = np.array(cld)
            st.write('Cloudy')
            st.image(cld)
            cld_1 = img.open('Winter\\Rain\\rain26.jpg')
            cld_1 = np.array(cld_1)
            st.write('Rain')
            st.image(cld_1)
        with cols2 : 
            cld_3 = img.open('Winter\Shine\shine3.jpg')
            cld_3 = np.array(cld_3)
            st.write('Shine')
            st.image(cld_3)
            cld_4 = img.open('Winter\Sunrise\sunrise14.jpg')
            cld_4 = np.array(cld_4)
            st.write('Sunrise')
            st.image(cld_4)
        
        # Bulid Models 
        st.header('1.2 Build Model') 
        
        class ResNetCombinerNetwork(keras.Model):

            class ResNetLayers(keras.layers.Layer): 
                def __init__(self, Filters, leaky_rate , drop_rate,chandim ,kernel,
                             regularizer = keras.regularizers.L2(0.0005),**kwargs):
                    super().__init__(**kwargs)
                    self.convs = Conv2D(filters=Filters , kernel_size=kernel , 
                                        kernel_regularizer=regularizer)
                    self.Leaky = LeakyReLU(leaky_rate)
                    self.dropout = Drp(drop_rate)
                    self.batchNorms = BatchNormalization(chandim)
                    
                def call(self, inputs):
                    x1 = self.convs(inputs)
                    x = self.batchNorms(x1)
                    x = self.Leaky(x)
                    x = self.dropout(x)
                    skips = add([x1 , x])
                    return skips

            def __init__(self, filters , num_label, 
                         regularizer = keras.regularizers.L2(0.0005),  *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.A = ResNetCombinerNetwork.ResNetLayers(filters , 0.1 , 0.2 , -1, 3)
                self.B = ResNetCombinerNetwork.ResNetLayers(filters , 0.3 , 0.3 , -1, 3)
                self.C = ResNetCombinerNetwork.ResNetLayers(filters , 0.5 , 0.5 , -1, 3)
                self.canger = Dense(filters + (3*num_label))
                self.max_poll = MaxPooling2D((3,3))
                self.Fc = Dense(1200 , kernel_regularizer=regularizer)
                self.Batch_norms = BatchNormalization()
                self.flat = Flatten()
                self.drops = Drp(0.2)
                self.Relus = ReLU()
                self.out = Dense(num_label , kernel_regularizer=regularizer)
                self.soft = sfx()
                self.built = True

            def call(self, inputs):
                BloctA = self.A(inputs)
                BloctB = self.B(inputs)
                BloctC = self.C(inputs)
                Combiner = tf.concat([BloctA , BloctB , BloctC] , axis=-1)
                canger = self.Relus(Combiner)
                canger = self.canger(canger)
                canger = self.max_poll(canger)
                canger = self.drops(canger)
                x = self.flat(canger)
                x = self.Fc(x)
                x = self.Relus(x)
                x = self.Batch_norms(x)
                x = self.out(x) 
                x = self.soft(x)
                return x
            
        buttones = st.button('show Models!')
        colns1 , colns2 = st.columns((5,5))
        if buttones :
            with colns1 : 
                codes = '''class ResNetCombinerNetwork(keras.Model):

            #Class di bawah ini adalah class Untuk Block Resnet nya 

                class ResNetLayers(keras.layers.Layer): 
                    def __init__(self, Filters, leaky_rate , drop_rate,chandim ,kernel,
                                 regularizer = keras.regularizers.L2(0.0005),**kwargs):
                        super().__init__(**kwargs)
                        self.convs = Conv2D(filters=Filters , kernel_size=kernel , 
                                            kernel_regularizer=regularizer)
                        self.Leaky = LeakyReLU(leaky_rate)
                        self.dropout = Drp(drop_rate)
                        self.batchNorms = BatchNormalization(chandim)

                    def call(self, inputs):
                        x1 = self.convs(inputs)
                        x = self.batchNorms(x1)
                        x = self.Leaky(x)
                        x = self.dropout(x)
                        skips = add([x1 , x])
                        return skips

            # Main Arsitektur

                def __init__(self, filters , num_label, 
                             regularizer = keras.regularizers.L2(0.0005),  *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.A = ResNetCombinerNetwork.ResNetLayers(filters , 0.1 , 0.2 , -1, 3)
                    self.B = ResNetCombinerNetwork.ResNetLayers(filters , 0.3 , 0.3 , -1, 3)
                    self.C = ResNetCombinerNetwork.ResNetLayers(filters , 0.5 , 0.5 , -1, 3)
                    self.canger = Dense(filters + (3*num_label))
                    self.max_poll = MaxPooling2D((3,3))
                    self.Fc = Dense(1200 , kernel_regularizer=regularizer)
                    self.Batch_norms = BatchNormalization()
                    self.flat = Flatten()
                    self.drops = Drp(0.2)
                    self.Relus = ReLU()
                    self.out = Dense(num_label , kernel_regularizer=regularizer)
                    self.soft = sfx()
                    self.built = True

                def call(self, inputs):
                    BloctA = self.A(inputs)
                    BloctB = self.B(inputs)
                    BloctC = self.C(inputs)
                    Combiner = tf.concat([BloctA , BloctB , BloctC] , axis=-1)
                    canger = self.Relus(Combiner)
                    canger = self.canger(canger)
                    canger = self.max_poll(canger)
                    canger = self.drops(canger)
                    x = self.flat(canger)
                    x = self.Fc(x)
                    x = self.Relus(x)
                    x = self.Batch_norms(x)
                    x = self.out(x) 
                    x = self.soft(x)
                    return x'''
                st.code(codes , line_numbers=True)
            with colns2 : 
                st.text('''
                                    Model: "res_net_combiner_model_20"
                _________________________________________________________________
                 Layer (type)                Output Shape              Param #   
                =================================================================
                 res_net_layers_60 (ResNetLa  multiple                 3840      
                 yers)                                                           

                 res_net_layers_61 (ResNetLa  multiple                 3840      
                 yers)                                                           

                 res_net_layers_62 (ResNetLa  multiple                 3840      
                 yers)                                                           

                 dense_58 (Dense)            multiple                  4804      

                 dense_59 (Dense)            multiple                  240927600 

                 dense_60 (Dense)            multiple                  47652     

                 softmax_20 (Softmax)        multiple                  0         

                 flatten_20 (Flatten)        multiple                  0         

                 max_pooling2d_20 (MaxPoolin  multiple                 0         
                 g2D)                                                            

                 batch_normalization_83 (Bat  multiple                 4800      
                 chNormalization)                                                

                 dropout_83 (Dropout)        multiple                  0         

                 re_lu_20 (ReLU)             multiple                  0         

                =================================================================
                Total params: 240,996,376
                Trainable params: 240,993,256
                Non-trainable params: 3,120
                _________________________________________________________________
                                    ''')
        selecop = st.selectbox('Pilih Image dari sini dan lihat bagaimana AI memprediksi' , 
                               ['Gambar 1' , 'Gambar 2' , 'Gambar 3' , 'Gambar 4'])
        labs = ['Berawan' , 'Hujan' , 'cerah' , 'pagi']
        if selecop == 'Gambar 1' : 
            imlist = os.listdir('Winter\Cloudy')
            openes = f'Winter\Cloudy\{imlist[np.random.randint(0,len(imlist))-1]}'
            gambars = img.open(openes)
            gambars = np.array(gambars)
            st.image(gambars)
            img_inp = tf.keras.preprocessing.image.load_img(openes , target_size=(120 , 120))
            img_inp = tf.keras.preprocessing.image.img_to_array(img_inp).astype('float32') / 255
            img_inp = tf.expand_dims(img_inp , axis=0)
            models_k = keras.models.load_model('Cloud.tf')
            predict = models_k.predict(img_inp)
            predict = tf.argmax(predict[0])
            st.write(f'hasil Prediksi AI : {labs[predict.numpy()]}')
        
        elif selecop == 'Gambar 2' : 
            imlist = os.listdir('Winter\\Rain')
            openes = f'Winter\\Rain\\{imlist[np.random.randint(0,len(imlist))-1]}'
            gambars = img.open(openes)
            gambars = np.array(gambars)
            st.image(gambars)
            img_inp = tf.keras.preprocessing.image.load_img(openes , target_size=(120 , 120))
            img_inp = tf.keras.preprocessing.image.img_to_array(img_inp).astype('float32') / 255
            img_inp = tf.expand_dims(img_inp , axis=0)
            models_k = keras.models.load_model('Cloud.tf')
            predict = models_k.predict(img_inp)
            predict = tf.argmax(predict[0])
            st.write(f'hasil Prediksi AI : {labs[predict.numpy()]}')
            
        elif selecop == 'Gambar 3' : 
            imlist = os.listdir('Winter\Shine')
            openes = f'Winter\Shine\{imlist[np.random.randint(0,len(imlist))-1]}'
            gambars = img.open(openes)
            gambars = np.array(gambars)
            st.image(gambars)
            img_inp = tf.keras.preprocessing.image.load_img(openes , target_size=(120 , 120))
            img_inp = tf.keras.preprocessing.image.img_to_array(img_inp).astype('float32') / 255
            img_inp = tf.expand_dims(img_inp , axis=0)
            models_k = keras.models.load_model('Cloud.tf')
            predict = models_k.predict(img_inp)
            predict = tf.argmax(predict[0])
            st.write(f'hasil Prediksi AI : {labs[predict.numpy()]}')
            
        elif selecop == 'Gambar 4' : 
            imlist = os.listdir('Winter\Sunrise')
            openes = f'Winter\Sunrise\{imlist[np.random.randint(0,len(imlist))-1]}'
            gambars = img.open(openes)
            gambars = np.array(gambars)
            st.image(gambars)
            img_inp = tf.keras.preprocessing.image.load_img(openes , target_size=(120 , 120))
            img_inp = tf.keras.preprocessing.image.img_to_array(img_inp).astype('float32') / 255
            img_inp = tf.expand_dims(img_inp , axis=0)
            models_k = keras.models.load_model('Cloud.tf')
            predict = models_k.predict(img_inp)
            predict = tf.argmax(predict[0])
            st.write(f'hasil Prediksi AI : {labs[predict.numpy()]}')
        
    if opt2 == 'Deep Reinforcmen Learning' :
        st.write('Powered By Pytorch') 
        st.header('BlindNet using MAB Encoder For solve Blind Clasif task')
        st.header('1.1 The Dataset')
        df_t = pd.read_csv('SPAM text message 20170820 - Data.csv')
        dfp = df_t.head()
        st.table(dfp)
        st.write('TO BE CONTINUE')
        
        # Build Models 
        class BlindNet(Module): 
            
            class Embens(Module) :
                def __init__(self, n_words , hiddens , droprate ,  *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.n_w = n_words
                    self.Pos_em = Embedding(n_words , hiddens)
                    self.id_em = Embedding(hiddens , hiddens)
                    self.drop = Dropout(droprate)
                    self.norms = LayerNorm(hiddens)
                def forward(self , inputs:torch.Tensor): 
                    ids = torch.clamp(torch.range(0 , inputs.shape(1) , dtype=torch.long()) , 0 , 
                                      self.id_em.num_embeddings-1)
                    tokens = torch.clamp(inputs.long() , 0 , self.Pos_em.num_embeddings-1)
                    emid = self.id_em(ids)
                    token = self.Pos_em(tokens)
                    combine = emid + token
                    combine = self.norms(combine)
                    combine = self.drop(combine)
                    return combine
                
            class Mul_Attentions(Module): 
                
                class Head(Module): 
                    
                    def Scale_Dot_product(self , q:torch.Tensor , k:torch.Tensor , v:torch.Tensor): 
                        dim_k = np.sqrt(k.shape(-1))
                        Attention = torch.bmm(q , k.transpose((1,2))) / dim_k
                        Attention = Softmax(-1)(Attention)
                        Attention = torch.bmm(Attention , v)
                        return Attention
                    
                    def __init__(self,n_model , d_head, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.LT1 , self.LT2 , self.LT3 = (Linear(n_model , d_head) for _ in range(3))
                    
                    def forward(self , inputs): 
                        q = self.LT1(inputs) 
                        k = self.LT2(inputs)
                        v = self.LT3(inputs)
                        Out = self.Scale_Dot_product(q , k , v)
                        return Out
                    
                class Attentions_mult(Module): 
                    
                    def __init__(self, n_models, n_head , *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        pass
                    
            def __init__(self,n_arms , hidden , n_model , n_heads, n_words ,
                         droprate, *args, **kwargs):
                super().__init__(*args, **kwargs) 
                
        
elif opt == 'About Creator' :
    st.header('LOGIN KE WEB UNTUK MEMASTIKAN ANDA PESERTA')
    nama_kel = ['Rafli Ardiansyah'.lower() , 'Yudo nidlom firmansyah'.lower() , 'Muhammad Masyari Hikmal Kiromy'.lower()]
    npm = ['2213020129' , '2213020102' , '2213020111']
    name = st.text_input('Masukan Nama').lower()
    npms = st.text_input('Masukan NPM').lower()
    butts = st.button('Login')
    if butts : 
        if (name in nama_kel) & (npms in npm) : 
            st.write('Anda Adalah salah satu dari kami')
            st.header('Nama Anggota')
            st.text('''
                    > Yudo Nidlom Firmansyah (2213020129)

                    > Rafli Ardiansyah (2213020102)

                    > Muhammad Masyari Hikmal Kiromy (2213020111)
                    ''')
        else : 
            st.header('Blud think he can get inside 😴😪💀💀🔥🔥💯‼️‼️')
            stiker = img.open('image.png').resize((120,120))
            stiker = np.array(stiker)
            st.image(stiker)
            
            