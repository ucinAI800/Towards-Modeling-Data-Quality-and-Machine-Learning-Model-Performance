# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:15:36 2023

@author: 15135
"""
import numpy as np
import os
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as NNF
import sklearn
import sklearn.metrics as M
import urllib.request as R
import shutil
import tarfile
import zipfile
from scipy.io import loadmat
import pandas as pd
import datetime as date
from dateutil.relativedelta import relativedelta
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage import io, transform
from torchvision import transforms, utils
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor, KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Ridge, MultiTaskLasso, MultiTaskElasticNet, Lars, OrthogonalMatchingPursuit, LogisticRegression, Perceptron
import torch.nn as nn
import torchvision
from sklearn.cross_decomposition import PLSRegression
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, AgglomerativeClustering,DBSCAN  
from sklearn.naive_bayes import GaussianNB
import tqdm
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Type:
    """A class representing a type.
    
    Attributes:
       string (str): The string of the type."""
    def __init__(self, string):
        """Initializes a Type object.
        
        Parameters:
            string (str): The string of the type."""
        self.string=string
        

def Task_Types():
    """Returns the array of task type objects"""
    task_type_strings=["Regression", "Clustering", "Classification"]
    
    task_types=[]
    
    for task_type_string in task_type_strings:
        
        task_type=Type(task_type_string)
        
        task_types.append(task_type)
    
    return task_types

def Dataset_Types():
    """Returns the array of dataset type objects"""
    dataset_type_strings=["Sequence-Based", "Image-Based"]
    
    dataset_types=[]
    
    for dataset_type_string in dataset_type_strings:
        
        dataset_type=Type(dataset_type_string)
        
        dataset_types.append(dataset_type)
    
    return dataset_types
    
def np_cube(array):
    """Takes an array as input and returns an array whose elements are the cubes of the input array"""
    cube_array=np.power(array, 3)
    
    return cube_array

rng=np.random.default_rng(seed=0)

def Make_Image_Datasets_Directory():
    """Make a subdirectory called "Image Datasets" in parent directory of this file, if it does not already exist."""
    current_directory=os.path.dirname(__file__)
    image_datasets_directory_name="Image Datasets"
    image_datasets_directory=os.path.join(current_directory, image_datasets_directory_name)
    if glob.glob(pathname=image_datasets_directory)==[]:
        os.mkdir(image_datasets_directory)
    
    return

def Image_Datasets_Directory():
    """Return the Image Datasets Directory path"""
    current_directory=os.path.dirname(__file__)
    image_datasets_directory_list=glob.glob("Image Datasets", root_dir=current_directory)
    image_datasets_directory=image_datasets_directory_list[0]
    image_datasets_directory=os.path.join(current_directory, image_datasets_directory)
    
    return image_datasets_directory

def Make_MNIST_Datasets_Directory():
    """Make a subdirectory called "MNIST Datasets" in the directory "Image Datasets", if it does not already exist."""
    image_datasets_directory=Image_Datasets_Directory()
    mnist_datasets_directory_name="MNIST Datasets"
    mnist_datasets_directory=os.path.join(image_datasets_directory, mnist_datasets_directory_name)
    if glob.glob(pathname=mnist_datasets_directory)==[]:
        os.mkdir(mnist_datasets_directory)
        
    return

def MNIST_Datasets_Directory():
    """Return the MNIST Datasets Directory path"""
    image_datasets_directory=Image_Datasets_Directory()
    mnist_datasets_directory_list=glob.glob("MNIST Datasets", root_dir=image_datasets_directory)
    mnist_datasets_directory=mnist_datasets_directory_list[0]
    mnist_datasets_directory=os.path.join(image_datasets_directory, mnist_datasets_directory)
    
    return mnist_datasets_directory
    
def Add_MNIST_Datasets_To_MNIST_Datasets_Directory():
    """Add the MNIST datasets to the image datasets directory. Return the datasets."""
    mnist_datasets_directory=MNIST_Datasets_Directory()
    mnist_train_dataset=datasets.MNIST(root=mnist_datasets_directory, train=True, download=True, transform=T.ToTensor())
    mnist_test_dataset=datasets.MNIST(root=mnist_datasets_directory, train=False, download=True, transform=T.ToTensor())
    mnist_train_dataset=["MNIST Train", mnist_train_dataset]
    mnist_test_dataset=["MNIST Test", mnist_test_dataset]
    
    mnist_datasets=[mnist_train_dataset, mnist_test_dataset]
    
    return mnist_datasets

def Make_IMDB_WIKI_Datasets_Directory():
    """Make a subdirectory called "IMDB-WIKI Datasets" in the directory "Image Datasets", if it does not already exist."""
    image_datasets_directory=Image_Datasets_Directory()
    imdb_wiki_datasets_directory_name="IMDB-WIKI Datasets"
    imdb_wiki_datasets_directory=os.path.join(image_datasets_directory, imdb_wiki_datasets_directory_name)
    if glob.glob(pathname=imdb_wiki_datasets_directory)==[]:
        os.mkdir(imdb_wiki_datasets_directory)
        
    return

def IMDB_WIKI_Datasets_Directory():
    """Return the IMDB-WIKI Datasets Directory path"""
    image_datasets_directory=Image_Datasets_Directory()
    imdb_wiki_datasets_directory_list=glob.glob("IMDB-WIKI Datasets", root_dir=image_datasets_directory)
    imdb_wiki_datasets_directory=imdb_wiki_datasets_directory_list[0]
    imdb_wiki_datasets_directory=os.path.join(image_datasets_directory, imdb_wiki_datasets_directory)
    
    return imdb_wiki_datasets_directory

def Mat_Py():
    """Preprocesses the IMDB-WIKI dataset. Store metadata in csv file 'Image Datasets/IMDB-WIKI Datasets/meta.csv'"""

    cols = ['age', 'gender', 'path', 'face_score1', 'face_score2']

    parent_directory=os.path.dirname(__file__)
    image_datasets_directory=os.path.join(parent_directory, "Image Datasets")
    imdb_wiki_datasets_directory=os.path.join(image_datasets_directory, "IMDB-WIKI Datasets")
    os.chdir(imdb_wiki_datasets_directory)


    imdb_mat = 'imdb_crop/imdb.mat'
    wiki_mat = 'wiki_crop/wiki.mat'

    imdb_data = loadmat(imdb_mat)
    wiki_data = loadmat(wiki_mat)

    del imdb_mat, wiki_mat

    imdb = imdb_data['imdb']
    wiki = wiki_data['wiki']

    imdb_photo_taken = imdb[0][0][1][0]
    imdb_full_path = imdb[0][0][2][0]
    imdb_gender = imdb[0][0][3][0]
    imdb_face_score1 = imdb[0][0][6][0]
    imdb_face_score2 = imdb[0][0][7][0]

    wiki_photo_taken = wiki[0][0][1][0]
    wiki_full_path = wiki[0][0][2][0]
    wiki_gender = wiki[0][0][3][0]
    wiki_face_score1 = wiki[0][0][6][0]
    wiki_face_score2 = wiki[0][0][7][0]

    imdb_path = []
    wiki_path = []

    for path in imdb_full_path:
        imdb_path.append('imdb_crop/' + path[0])

    for path in wiki_full_path:
        wiki_path.append('wiki_crop/' + path[0])

    imdb_genders = []
    wiki_genders = []

    for n in range(len(imdb_gender)):
        if imdb_gender[n] == 1:
            imdb_genders.append('male')
        else:
            imdb_genders.append('female')

    for n in range(len(wiki_gender)):
        if wiki_gender[n] == 1:
            wiki_genders.append('male')
        else:
            wiki_genders.append('female')

    imdb_dob = []
    wiki_dob = []

    for file in imdb_path:
        temp = file.split('_')[3]
        temp = temp.split('-')
        if len(temp[1]) == 1:
            temp[1] = '0' + temp[1]
        if len(temp[2]) == 1:
            temp[2] = '0' + temp[2]

        if temp[1] == '00':
            temp[1] = '01'
        if temp[2] == '00':
            temp[2] = '01'
        
        imdb_dob.append('-'.join(temp))

    for file in wiki_path:
        wiki_dob.append(file.split('_')[2])


    imdb_age = []
    wiki_age = []

    for i in range(len(imdb_dob)):
        try:
            d1 = date.datetime.strptime(imdb_dob[i][0:10], '%Y-%m-%d')
            d2 = date.datetime.strptime(str(imdb_photo_taken[i]), '%Y')
            rdelta = relativedelta(d2, d1)
            diff = rdelta.years
        except Exception as ex:
            print(ex)
            diff = -1
        imdb_age.append(diff)

    for i in range(len(wiki_dob)):
        try:
            d1 = date.datetime.strptime(wiki_dob[i][0:10], '%Y-%m-%d')
            d2 = date.datetime.strptime(str(wiki_photo_taken[i]), '%Y')
            rdelta = relativedelta(d2, d1)
            diff = rdelta.years
        except Exception as ex:
            print(ex)
            diff = -1
        wiki_age.append(diff)

    final_imdb = np.vstack((imdb_age, imdb_genders, imdb_path, imdb_face_score1, imdb_face_score2)).T
    final_wiki = np.vstack((wiki_age, wiki_genders, wiki_path, wiki_face_score1, wiki_face_score2)).T

    final_imdb_df = pd.DataFrame(final_imdb)
    final_wiki_df = pd.DataFrame(final_wiki)

    final_imdb_df.columns = cols
    final_wiki_df.columns = cols

    meta = pd.concat((final_imdb_df, final_wiki_df))

    meta = meta[meta['face_score1'] != '-inf']
    meta = meta[meta['face_score2'] == 'nan']

    meta = meta.drop(['face_score1', 'face_score2'], axis=1)

    meta = meta.sample(frac=1)

    meta.to_csv('meta.csv', index=False)

    os.chdir(parent_directory)

    return

def Age_Py():
    """Preprocesses the data in IMDB-WIKI datasets. Creates folder structure and saves results."""
    # Importing dependencies
   

    parent_directory=os.path.dirname(__file__)
    image_datasets_directory=os.path.join(parent_directory, "Image Datasets")
    imdb_wiki_datasets_directory=os.path.join(image_datasets_directory, "IMDB-WIKI Datasets")
    os.chdir(imdb_wiki_datasets_directory)

    # Loading dataset
    meta = pd.read_csv('meta.csv')

    # Dropping gender column
    meta = meta.drop(['gender'], axis=1)

    # Filtaring dataset
    meta = meta[meta['age'] >= 0]
    meta = meta[meta['age'] <= 101]

    # Converting into numpy array
    meta = meta.values

    # Spliting dataset into training and testing set
    D_train, D_test = train_test_split(meta, test_size=0.2, random_state=42)

    # Making the directory structure
    for i in range(102):
        output_dir_train_male = 'dataset/age/train/' + str(i)
        output_dir_train_female = 'dataset/age/train/' + str(i)

        if not os.path.exists(output_dir_train_male):
            os.makedirs(output_dir_train_male)

        if not os.path.exists(output_dir_train_female):
            os.makedirs(output_dir_train_female)

        output_dir_test_male = 'dataset/age/test/' + str(i)
        output_dir_test_female = 'dataset/age/test/' + str(i)

        if not os.path.exists(output_dir_test_male):
            os.makedirs(output_dir_test_male)

        if not os.path.exists(output_dir_test_female):
            os.makedirs(output_dir_test_female)

    # Finally making the training and testing set
    counter = 0

    for image in D_train:
        img = cv2.imread(image[1], 1)
        img = cv2.resize(img, (128,128))
        cv2.imwrite('dataset/age/train/' + str(image[0]) + '/' + str(counter) + '.jpg', img)
        print('--('+str(counter)+')Processing--')
        counter += 1

    counter = 0

    for image in D_test:
        img = cv2.imread(image[1], 1)
        img = cv2.resize(img, (128,128))
        cv2.imwrite('dataset/age/test/' + str(image[0]) +  '/' + str(counter) + '.jpg', img)
        print('--('+str(counter)+')Processing--')
        counter += 1

    os.chdir(parent_directory)
    
    return

def Gender_Py():
    """Preprocesses the IMDB-WIKI datasets. Creates folder structure and save results"""
    parent_directory=os.path.dirname(__file__)
    image_datasets_directory=os.path.join(parent_directory, "Image Datasets")
    imdb_wiki_datasets_directory=os.path.join(image_datasets_directory, "IMDB-WIKI Datasets")
    os.chdir(imdb_wiki_datasets_directory)

    # Loading the data
    meta = pd.read_csv('meta.csv')

    # Deleting the age column as we dont need it
    meta = meta.drop(['age'], axis=1)

    # Spliting the dataset into train and test set
    D_train, D_test = train_test_split(meta, test_size=0.1, random_state=42)

    # The dataset contains more male faces that female faces. This can couse some problems.
    # One feature can start dominating on other feature. To solve this I am selecting equal number of male and female faces in the training set
    D_train_male = D_train[D_train['gender'] == 'male']
    D_train_female = D_train[D_train['gender'] == 'female']

    no_male = len(D_train_male)
    no_female = len(D_train_female)

    if no_male > no_female:
        extra = D_train_male[no_female:]
        D_train_male = D_train_male[0:no_female]

        D_test = pd.concat((D_test, extra))
    else:
        extra = D_train_male[no_male:]
        D_train_male = D_train_male[0:no_male]

        D_test = pd.concat((D_test, extra))

    D_train = pd.concat((D_train_male, D_train_female))

    # Shuffling the dataset
    D_train = D_train.sample(frac=1)
    D_test = D_test.sample(frac=1)

    # Generating folder struture for the data
    output_dir_train_male = 'dataset/gender/train/male'
    output_dir_train_female = 'dataset/gender/train/female'

    if not os.path.exists(output_dir_train_male):
        os.makedirs(output_dir_train_male)

    if not os.path.exists(output_dir_train_female):
        os.makedirs(output_dir_train_female)

    output_dir_test_male = 'dataset/gender/test/male'
    output_dir_test_female = 'dataset/gender/test/female'

    if not os.path.exists(output_dir_test_male):
        os.makedirs(output_dir_test_male)

    if not os.path.exists(output_dir_test_female):
        os.makedirs(output_dir_test_female)

    # Finally processing the image training and testting set
    counter = 0

    for image in D_train.values:
        img = cv2.imread(image[1], 1)
        img = cv2.resize(img, (128,128))
        if image[0] == 'male':
            cv2.imwrite('dataset/gender/train/male/' + str(counter) + '.jpg', img)
        else:
            cv2.imwrite('dataset/gender/train/female/' + str(counter) + '.jpg', img)
        print('--('+str(counter)+')Processing--')
        counter += 1

    counter = 0

    for image in D_test.values:
        img = cv2.imread(image[1], 1)
        img = cv2.resize(img, (128,128))
        if image[0] == 'male':
            cv2.imwrite('dataset/gender/test/male/' + str(counter) + '.jpg', img)
        else:
            cv2.imwrite('dataset/gender/test/female/' + str(counter) + '.jpg', img)
        print('--('+str(counter)+')Processing--')
        counter += 1

    os.chdir(parent_directory)

    return

class IMDB_WIKI_Dataset(Dataset):
    """IMDB-WIKI Dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        imdb_wiki_frame=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
        samples=[]
        self.labels=np.zeros(10)
        for idx in tqdm.tqdm(range(imdb_wiki_frame.size)[:10], desc="Importing IMDB-WIKI Dataset"):
            img_name=os.path.join(self.root_dir, imdb_wiki_frame.iloc[idx, 3])
            image=io.imread(img_name)
            if len(image.shape)==3:
                age=imdb_wiki_frame.iloc[idx, 1]
                self.labels[idx]=age
                sample={'age': age, 'image' : image}
                if self.transform:
                    sample=self.transform(sample)
                image=sample["image"]
                age=sample["age"]
                sample=(image, age)
                samples.append(sample)
        self.data=samples
        
        
    def __len__(self):
        return len(self.imdb_wiki_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
            
        sample=self.data[idx]
                    
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, age = sample['image'], sample['age']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'age': age}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, age = sample['image'], sample['age']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'age': torch.tensor(age)}

def IMDB_WIKI_Datasets():
    """Return the IMDB-WIKI datasets."""
        
    imdb_wiki_datasets_directory=IMDB_WIKI_Datasets_Directory()
    
    meta=os.path.join(imdb_wiki_datasets_directory, "meta.csv")
    
    meta_df=pd.read_csv(meta)
    
    meta_train_df, meta_test_df=train_test_split(meta_df, test_size=0.2)
    
    meta_train=os.path.join(imdb_wiki_datasets_directory, "meta train.csv")

    meta_test=os.path.join(imdb_wiki_datasets_directory, "meta test.csv")
    
    meta_train_df.to_csv(meta_train)
    
    meta_test_df.to_csv(meta_test)
    
    transform=T.Compose([Rescale((256,256)), ToTensor()])
    
    imdb_wiki_train_dataset=IMDB_WIKI_Dataset(csv_file=meta_train, root_dir=imdb_wiki_datasets_directory, transform=transform)
    
    imdb_wiki_train_dataset=["IMDB-WIKI Train", imdb_wiki_train_dataset]
    
    imdb_wiki_test_dataset=IMDB_WIKI_Dataset(csv_file=meta_test, root_dir=imdb_wiki_datasets_directory, transform=transform)

    imdb_wiki_test_dataset=["IMDB-WIKI Test", imdb_wiki_test_dataset]

    imdb_wiki_datasets=[imdb_wiki_train_dataset, imdb_wiki_test_dataset]
        
    return imdb_wiki_datasets

def Add_IMDB_WIKI_Datasets_To_IMDB_WIKI_Datasets_Directory():
    """Add the IMDB-WIKI datasets to the image datasets directory. Return the datasets."""
    imdb_wiki_datasets_directory=IMDB_WIKI_Datasets_Directory()
    
    imdb_crop_url="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
    imdb_crop_tar=os.path.join(imdb_wiki_datasets_directory, "imdb_crop.tar")
    if glob.glob(imdb_crop_tar)==[]:
        with R.urlopen(imdb_crop_url) as in_stream, open(imdb_crop_tar, "wb") as out_file:
            shutil.copyfileobj(in_stream, out_file)
        imdb_crop=tarfile.open(name=imdb_crop_tar)
    
        imdb_crop.extractall(path=imdb_wiki_datasets_directory)
    
    wiki_crop_url="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar"    
    wiki_crop_tar=os.path.join(imdb_wiki_datasets_directory, "wiki_crop.tar")
    if glob.glob(wiki_crop_tar)==[]:
        with R.urlopen(wiki_crop_url) as in_stream, open(wiki_crop_tar, "wb") as out_file:
            shutil.copyfileobj(in_stream, out_file)
        
        wiki_crop=tarfile.open(name=wiki_crop_tar)
    
        wiki_crop.extractall(path=imdb_wiki_datasets_directory)
    
    processed_imdb_wiki_url="https://github.com/imdeepmind/processed-imdb-wiki-dataset/archive/refs/heads/master.zip"
    processed_imdb_wiki_zip=os.path.join(imdb_wiki_datasets_directory, "processed_imdb_wiki.zip")
    if glob.glob(processed_imdb_wiki_zip)==[]:
        with R.urlopen(processed_imdb_wiki_url) as in_stream, open(processed_imdb_wiki_zip, "wb") as out_file:
            shutil.copyfileobj(in_stream, out_file)
            
        processed_imdb_wiki=zipfile.ZipFile(processed_imdb_wiki_zip)
        processed_imdb_wiki.extractall(imdb_wiki_datasets_directory)
    
        Mat_Py()
        Age_Py()
        Gender_Py()
    
    imdb_wiki_datasets=IMDB_WIKI_Datasets()
    
    return imdb_wiki_datasets

def Add_Image_Datasets_To_Image_Datasets_Directory():
    """Add image datasets to their respective directories. Return the image datasets"""
    Make_Image_Datasets_Directory()
    Make_MNIST_Datasets_Directory()
    mnist_datasets=Add_MNIST_Datasets_To_MNIST_Datasets_Directory()
    Make_IMDB_WIKI_Datasets_Directory()
    imdb_wiki_datasets=Add_IMDB_WIKI_Datasets_To_IMDB_WIKI_Datasets_Directory()
    image_datasets=[mnist_datasets, imdb_wiki_datasets]

    return image_datasets
    
class Function_Type(Type):
    """A Derived Class of Type representing a function type. 
    
    Attributes:
        string (str): The string of the function type.
        
        function (Callable): The function of the function type.
        
        domain (array): The domain of the function type."""
        
        
    def __init__(self, string, function, domain):
        """Extends the initialization of a Type object to a Function_Type object.
        
        Parameters:
            string (str): The string of the function type.
            
            function (Callable): The function of the function type
            
            domain (array): The domain of the function."""
        Type.__init__(self, string)
        self.function=function
        self.domain=domain

def Deterministic_Types():
    """Returns a 2-D list of arrays of deterministic types."""   
    
    sequence_based_regression_deterministic_type_strings=[["sin(t)"], ["log(t)"], ["t^2"]]
    
    sequence_based_clustering_and_classification_deterministic_type_strings=[["sin(t)", "cos(t)"], ["log_2(t)", "log_10(t)"], ["t^2", "t^3"]]

    sequence_based_regression_deterministic_type_functions=[[np.sin], [np.log], [np.square]]
    
    sequence_based_clustering_and_classification_deterministic_type_functions=[[np.sin, np.cos], [np.log2, np.log10], [np.square, np_cube]]
    
    sequence_based_regression_deterministic_type_features=[[rng.uniform(low=0, high=2*np.pi, size=10)], [rng.uniform(low=np.e**(-1), high=np.e, size=10)], [rng.uniform(low=-2**(1/2), high=2**(1/2), size=10)]]
    
    sequence_based_clustering_and_classification_deterministic_type_domains=[[rng.uniform(low=0, high=2*np.pi, size=(10,10)),rng.uniform(low=0, high=2*np.pi, size=(10,10))], [rng.uniform(low=2**(-1), high=2, size=(10,10)), rng.uniform(low=10**(-1), high=10, size=(10,10))], [rng.uniform(low=-2**(1/2), high=2**(1/2), size=(10,10)), rng.uniform(low=-2**(1/3), high=2**(1/3), size=(10,10))]]
    
    image_datasets=Add_Image_Datasets_To_Image_Datasets_Directory()

    image_based_regression_deterministic_type_dataset=image_datasets[1]
    
    image_based_regression_deterministic_type_train_dataset=image_based_regression_deterministic_type_dataset[0]
    
    image_based_regression_deterministic_type_test_dataset=image_based_regression_deterministic_type_dataset[1]
    
    image_based_regression_deterministic_type_train_string=image_based_regression_deterministic_type_train_dataset[0]
    
    image_based_regression_deterministic_type_test_string=image_based_regression_deterministic_type_test_dataset[0]
        
    image_based_clustering_and_classification_deterministic_type_dataset=image_datasets[0]

    image_based_clustering_and_classification_deterministic_type_train_dataset=image_based_clustering_and_classification_deterministic_type_dataset[0]
    
    image_based_clustering_and_classification_deterministic_type_test_dataset=image_based_clustering_and_classification_deterministic_type_dataset[1]


    image_based_clustering_and_classification_deterministic_type_train_string=image_based_clustering_and_classification_deterministic_type_train_dataset[0]
    
    image_based_clustering_and_classification_deterministic_type_test_string=image_based_clustering_and_classification_deterministic_type_test_dataset[0]


    image_based_regression_deterministic_type_strings=[image_based_regression_deterministic_type_train_string, image_based_regression_deterministic_type_test_string]

    image_based_clustering_and_classification_deterministic_type_strings=[image_based_clustering_and_classification_deterministic_type_train_string, image_based_clustering_and_classification_deterministic_type_test_string]

    image_based_regression_deterministic_type_train_dataset=image_based_regression_deterministic_type_train_dataset[1]

    image_based_regression_deterministic_type_test_dataset=image_based_regression_deterministic_type_test_dataset[1]

    image_based_clustering_and_classification_deterministic_type_train_dataset=image_based_clustering_and_classification_deterministic_type_train_dataset[1]
    
    image_based_clustering_and_classification_deterministic_type_test_dataset=image_based_clustering_and_classification_deterministic_type_test_dataset[1]
    
    image_based_regression_deterministic_type_functions=[[lambda i: image_based_regression_deterministic_type_train_dataset[i]], [lambda i: image_based_regression_deterministic_type_test_dataset[i]]]

    image_based_clustering_and_classification_deterministic_type_functions=[[lambda i: image_based_clustering_and_classification_deterministic_type_train_dataset[i]], [lambda i: image_based_clustering_and_classification_deterministic_type_test_dataset[i]]]
    
    image_based_regression_deterministic_type_domains=[[rng.integers(low=0, high=len(image_based_regression_deterministic_type_train_dataset.data), size=16)], [rng.integers(low=0, high=len(image_based_regression_deterministic_type_test_dataset.data), size=4)]]

    image_based_clustering_and_classification_deterministic_type_domains=[[rng.integers(low=0, high=image_based_clustering_and_classification_deterministic_type_train_dataset.data.shape[0], size=16)], [rng.integers(low=0, high=image_based_clustering_and_classification_deterministic_type_test_dataset.data.shape[0], size=4)]]
        
    regression_deterministic_type_strings=[sequence_based_regression_deterministic_type_strings, image_based_regression_deterministic_type_strings]
    
    regression_deterministic_type_functions=[sequence_based_regression_deterministic_type_functions, image_based_regression_deterministic_type_functions]
    
    regression_deterministic_type_domains=[sequence_based_regression_deterministic_type_features, image_based_regression_deterministic_type_domains]
        
    clustering_and_classification_deterministic_type_strings=[sequence_based_clustering_and_classification_deterministic_type_strings, image_based_clustering_and_classification_deterministic_type_strings]
    
    clustering_and_classification_deterministic_type_functions=[sequence_based_clustering_and_classification_deterministic_type_functions, image_based_clustering_and_classification_deterministic_type_functions]
    
    clustering_and_classification_deterministic_type_domains=[sequence_based_clustering_and_classification_deterministic_type_domains, image_based_clustering_and_classification_deterministic_type_domains]
    
    deterministic_type_strings=[regression_deterministic_type_strings, clustering_and_classification_deterministic_type_strings] 
    
    deterministic_type_functions=[regression_deterministic_type_functions, clustering_and_classification_deterministic_type_functions]
    
    deterministic_type_domains=[regression_deterministic_type_domains, clustering_and_classification_deterministic_type_domains]   
    
    deterministic_types=[]
    
    for i_0, task_type_deterministic_type_strings in enumerate(deterministic_type_strings):
    
        task_type_deterministic_type_functions=deterministic_type_functions[i_0]
        
        task_type_deterministic_type_domains=deterministic_type_domains[i_0]
    
        task_type_deterministic_types=[]
    
        for i_1, dataset_type_based_task_type_deterministic_type_strings in enumerate(task_type_deterministic_type_strings):
                
            dataset_type_based_task_type_deterministic_type_functions=task_type_deterministic_type_functions[i_1]
            
            dataset_type_based_task_type_deterministic_type_domains=task_type_deterministic_type_domains[i_1]
            
            dataset_type_based_task_type_deterministic_types=[]
            
            for i_2, dataset_type_based_task_type_deterministic_type_string in enumerate(dataset_type_based_task_type_deterministic_type_strings):
                
                dataset_type_based_task_type_deterministic_type_function=dataset_type_based_task_type_deterministic_type_functions[i_2]
                
                dataset_type_based_task_type_deterministic_type_domain=dataset_type_based_task_type_deterministic_type_domains[i_2]

                dataset_type_based_task_type_deterministic_type=Function_Type(dataset_type_based_task_type_deterministic_type_string, dataset_type_based_task_type_deterministic_type_function, dataset_type_based_task_type_deterministic_type_domain)
                            
                dataset_type_based_task_type_deterministic_types.append(dataset_type_based_task_type_deterministic_type)

                

            task_type_deterministic_types.append(dataset_type_based_task_type_deterministic_types)
        
        deterministic_types.append(task_type_deterministic_types)
    
    return deterministic_types

def Nondeterministic_Types():
    """Returns the nondeterministic types."""   
    sequence_based_nondeterministic_type_strings=["Gaussian", "Uniform"]
    
    sequence_based_nondeterministic_type_functions=[rng.normal, rng.uniform]
    
    image_based_nondeterministic_type_strings=["Gaussian"]
    
    image_based_nondeterministic_type_functions=[TF.gaussian_blur]
    
    nondeterministic_type_strings=[sequence_based_nondeterministic_type_strings, image_based_nondeterministic_type_strings]
    nondeterministic_type_functions=[sequence_based_nondeterministic_type_functions, image_based_nondeterministic_type_functions]
    
    nondeterministic_types=[]
    
    for i_0, dataset_type_nondeterministic_type_strings in enumerate(nondeterministic_type_strings):
        
        dataset_type_nondeterministic_type_functions=nondeterministic_type_functions[i_0]
        
        dataset_type_nondeterministic_types=[]
        
        for i_1, dataset_type_nondeterministic_type_string in enumerate(dataset_type_nondeterministic_type_strings):
            
            dataset_type_nondeterministic_type_function=dataset_type_nondeterministic_type_functions[i_1]
            
            dataset_type_nondeterministic_type=Function_Type(dataset_type_nondeterministic_type_string, dataset_type_nondeterministic_type_function, None)
            
            dataset_type_nondeterministic_types.append(dataset_type_nondeterministic_type)
            
        nondeterministic_types.append(dataset_type_nondeterministic_types)
        
    return nondeterministic_types

def Evaluation_Metric_Types():
    """Returns a list of arrays of evaluation metrics."""
    regression_evaluation_metric_type_strings=["MAE-Based Accuracy", "MSE-Based Accuracy"]
    
    regression_evaluation_metric_type_functions=[NNF.l1_loss, NNF.mse_loss]
    
    clustering_evaluation_metric_type_strings=["Rand Index", "Fowlkes-Mallows Index", "V-Measure"]
    
    clustering_evaluation_metric_type_functions=[M.rand_score, M.fowlkes_mallows_score, M.v_measure_score]
    
    classification_evaluation_metric_type_strings=["ROC AUC Score", "F1 Score"]
    
    classification_evaluation_metric_type_functions=[M.roc_auc_score, M.f1_score]
    
    evaluation_metric_type_strings=[regression_evaluation_metric_type_strings, clustering_evaluation_metric_type_strings, classification_evaluation_metric_type_strings]
    evaluation_metric_type_functions=[regression_evaluation_metric_type_functions, clustering_evaluation_metric_type_functions, classification_evaluation_metric_type_functions]

    evaluation_metric_types=[]

    for i_0, task_type_evaluation_metric_type_strings in enumerate(evaluation_metric_type_strings):
        
        task_type_evaluation_metric_type_functions=evaluation_metric_type_functions[i_0]
        
        task_type_evaluation_metric_types=[]
        
        for i_1, task_type_evaluation_metric_type_string in enumerate(task_type_evaluation_metric_type_strings):
            
            task_type_evaluation_metric_type_function=task_type_evaluation_metric_type_functions[i_1]
            
            task_type_evaluation_metric_type=Function_Type(task_type_evaluation_metric_type_string, task_type_evaluation_metric_type_function, domain=None)

            task_type_evaluation_metric_types.append(task_type_evaluation_metric_type)

        evaluation_metric_types.append(task_type_evaluation_metric_types)

    return evaluation_metric_types
    


class MLP_Regressor(nn.Module):
    """Class defining MLP model."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP_Regressor, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3=nn.Linear(self.hidden_dim, self.out_dim)
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x=self.linear3(x)
        return x

class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)

def gaussian(alpha):
    """Gaussian RBF"""
    phi = torch.exp(-1*alpha.pow(2))
    return phi

class RBF_Network(nn.Module):
    """Class defining RBF Network model"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RBF_Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            RBF(in_features=hidden_dim, out_features=hidden_dim, basis_func=gaussian),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class CNN(nn.Module):
    """Class defining CNN model. Uses pretrained CNN VGG16 as base."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(CNN, self).__init__()
        self.conv = torchvision.models.vgg16(pretrained=True).features
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

"""
Much of the code is modified from:
- https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
"""
 

class SOM(nn.Module):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
    def __init__(self, m, n, dim, niter, alpha=None, sigma=None):
        super(SOM, self).__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.niter = niter
        if alpha is None:
            self.alpha = 0.3
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)

        self.weights = torch.randn(m*n, dim)
        self.locations = torch.LongTensor(np.array(list(self.neuron_locations())))
        self.pdist = nn.PairwiseDistance(p=2)

    def get_weights(self):
        return self.weights

    def get_locations(self):
        return self.locations

    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])

    def map_vects(self, input_vects):
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self.weights))],
                            key=lambda x: np.linalg.norm(vect-self.weights[x]))
            to_return.append(self.locations[min_index])

        return to_return

    def forward(self, x, it):
        dists = self.pdist(torch.stack([x for i in range(self.m*self.n)]), self.weights)
        _, bmu_index = torch.min(dists, 0)
        bmu_loc = self.locations[bmu_index,:]
        bmu_loc = bmu_loc.squeeze()
        
        learning_rate_op = 1.0 - it/self.niter
        alpha_op = self.alpha * learning_rate_op
        sigma_op = self.sigma * learning_rate_op

        bmu_distance_squares = torch.sum(torch.pow(self.locations.float() - torch.stack([bmu_loc for i in range(self.m*self.n)]).float(), 2), 1)
        
        neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, sigma_op**2)))
        
        learning_rate_op = alpha_op * neighbourhood_func

        learning_rate_multiplier = torch.stack([learning_rate_op[i:i+1].repeat(self.dim) for i in range(self.m*self.n)])
        delta = torch.mul(learning_rate_multiplier, (torch.stack([x for i in range(self.m*self.n)]) - self.weights))                                         
        new_weights = torch.add(self.weights, delta)
        self.weights = new_weights

class MLP_Classifier(nn.Module):
    """Class defining MLP Classifier model."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP_Classifier, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.hidden_dim)
        self.relu=nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3=nn.Linear(self.hidden_dim, self.out_dim)
    def forward(self, x):
        x = self.linear1(x)
        x=self.relu(x)
        x = self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        return x



def Model_Types():
    """Returns the initialized model types"""
    non_linear_sequence_based_regression_model_type_strings=["KRR", "Epsilon-Support Vector Regression", "k-Nearest Neighbors Regression", "Fixed Radius-Contained Neighbors Regression", "GPR", "Decision Tree Regressor", "GBDT Regressor", "Random Forest Regressor", "AdaBoost Regressor"] 
    linear_sequence_based_regression_model_type_strings=["Ordinary Least Squares", "Ridge Regression", "Quadratic Regression", "Cubic Regression"]
    nn_sequence_based_regression_model_type_strings=["MLP Regressor", "RBF Network Regressor"]
    
    sequence_based_regression_model_type_strings=[non_linear_sequence_based_regression_model_type_strings, linear_sequence_based_regression_model_type_strings, nn_sequence_based_regression_model_type_strings]
    
    linear_image_based_regression_model_type_strings=["Ordinary Least Squares", "Ridge Regression", "Multi-Task Lasso", "Multi-Task Elastic-Net", "LARS", "OMP", "Quadratic Regression", "Cubic Regression"]
    non_linear_image_based_regression_model_type_strings=["KRR", "Epsilon-Support Vector Regression", "k-Nearest Neighbors Regression", "Fixed Radius-Contained Neighbors Regression",  "PLS Regression", "Decision Tree Regressor", "GBDT Regressor", "Random Forest Regressor", "AdaBoost Regressor"]
    nn_image_based_regression_model_type_strings=["MLP Regressor", "CNN Regressor", "RBF Network Regressor"]
    
    image_based_regression_model_type_strings=[non_linear_image_based_regression_model_type_strings, linear_image_based_regression_model_type_strings, nn_image_based_regression_model_type_strings]
    
    regression_model_type_strings=[sequence_based_regression_model_type_strings, image_based_regression_model_type_strings]

    non_linear_sequence_based_regression_model_type_functions=[KernelRidge(kernel="rbf"), SVR(), KNeighborsRegressor(), RadiusNeighborsRegressor(), GaussianProcessRegressor(), DecisionTreeRegressor(), GradientBoostingRegressor(), RandomForestRegressor(), AdaBoostRegressor()]
    linear_sequence_based_regression_model_type_functions=[LinearRegression(), Ridge(), LinearRegression(), LinearRegression()]
    nn_sequence_based_regression_model_type_functions=[MLP_Regressor(in_dim=1, hidden_dim=200, out_dim=1), RBF_Network(in_dim=1, hidden_dim=200, out_dim=1)]
    
    sequence_based_regression_model_type_functions=[non_linear_sequence_based_regression_model_type_functions, linear_sequence_based_regression_model_type_functions, nn_sequence_based_regression_model_type_functions]
    
    linear_image_based_regression_model_type_functions=[LinearRegression(), Ridge(), MultiTaskLasso(), MultiTaskElasticNet(), Lars(), OrthogonalMatchingPursuit(), LinearRegression(), LinearRegression()]
    non_linear_image_based_regression_model_type_functions=[KernelRidge(kernel="rbf"), SVR(), KNeighborsRegressor(), RadiusNeighborsRegressor(),  PLSRegression(n_components=1), DecisionTreeRegressor(), GradientBoostingRegressor(), RandomForestRegressor(), AdaBoostRegressor()]
    nn_image_based_regression_model_type_functions=[MLP_Regressor(in_dim=256*256*3, hidden_dim=200, out_dim=1), CNN(in_dim=256*256*3, hidden_dim=4096, out_dim=1), RBF_Network(in_dim=256*256, hidden_dim=4096, out_dim=1) ]
    
    image_based_regression_model_type_functions=[non_linear_image_based_regression_model_type_functions, linear_image_based_regression_model_type_functions, nn_image_based_regression_model_type_functions]
    
    regression_model_type_functions=[sequence_based_regression_model_type_functions, image_based_regression_model_type_functions]    
    
    sequence_based_clustering_model_type_strings=["K-Means Clustering", "Affinity Propagation Clustering", "Mean Shift Clustering", "Agglomerative Clustering", "DBSCAN", "Self-Organizing Maps"] 
        
    image_based_clustering_model_type_strings=["K-Means Clustering", "Affinity Propagation Clustering", "Mean Shift Clustering", "Agglomerative Clustering", "DBSCAN", "OMP"]
        
    clustering_model_type_strings=[sequence_based_clustering_model_type_strings, image_based_clustering_model_type_strings]

    sequence_based_clustering_model_type_functions=[KMeans(n_clusters=2), AffinityPropagation(), MeanShift(), AgglomerativeClustering(n_clusters=2), DBSCAN(), SOM(m=20, n=30, dim=1, niter=400)] 
        
    image_based_clustering_model_type_functions=[KMeans(n_clusters=10), AffinityPropagation(), MeanShift(), AgglomerativeClustering(n_clusters=10), DBSCAN(), SOM(m=20, n=30, dim=3, niter=400)] 
        
    clustering_model_type_functions=[sequence_based_clustering_model_type_functions, image_based_clustering_model_type_functions]    

    non_nn_sequence_based_classification_model_type_strings=["Logisitic Regression Classifier", "Linear Perceptron Classifier", "C-Support Vector Classifier", "K-Nearest Neighbors Classifier", "Fixed Radius Neighbors Classifier", "Gaussian Process Classifier", "Gaussian NB", "Decision Tree Classifier", "Gradient Boosting Classifier", "Random Forest Classifier", "AdaBoost Classifier"] 
    
    nn_sequence_based_classification_model_type_strings=["MLP Classifier", "RBF Network Classifier"]
    
    sequence_based_classification_model_type_strings=[non_nn_sequence_based_classification_model_type_strings, nn_sequence_based_classification_model_type_strings]
    
    non_nn_image_based_classification_model_type_strings=["Logisitic Regression Classifier", "Linear Perceptron Classifier", "C-Support Vector Classifier", "K-Nearest Neighbors Classifier", "Fixed Radius Neighbors Classifier", "Gaussian NB", "Decision Tree Classifier", "Gradient Boosting Classifier", "Random Forest Classifier", "AdaBoost Classifier"] 
    
    nn_image_based_classification_model_type_strings=["MLP Classifier", "CNN Classifier", "RBF Network Classifier"]
    
    image_based_classification_model_type_strings=[non_nn_image_based_classification_model_type_strings, nn_image_based_classification_model_type_strings]    
    
    classification_model_type_strings=[sequence_based_classification_model_type_strings, image_based_classification_model_type_strings]

    non_nn_sequence_based_classification_model_type_functions=[LogisticRegression(), Perceptron(), SVC(), KNeighborsClassifier(), RadiusNeighborsClassifier(), GaussianProcessClassifier(), GaussianNB(), DecisionTreeClassifier(), GradientBoostingClassifier(), RandomForestClassifier(), AdaBoostClassifier()] 
    
    nn_sequence_based_classification_model_type_functions=[MLP_Classifier(in_dim=1000, hidden_dim=200, out_dim=1), RBF_Network(in_dim=1000, hidden_dim=200, out_dim=1)]
    
    sequence_based_classification_model_type_functions=[non_nn_sequence_based_classification_model_type_functions, nn_sequence_based_classification_model_type_functions]
    
    non_nn_image_based_classification_model_type_functions=[LogisticRegression(), Perceptron(), SVC(), KNeighborsClassifier(), RadiusNeighborsClassifier(), GaussianNB(), DecisionTreeClassifier(), GradientBoostingClassifier(), RandomForestClassifier(), AdaBoostClassifier()] 
    
    nn_image_based_classification_model_type_functions=[MLP_Classifier(in_dim=256*256*3, hidden_dim=200, out_dim=1), CNN(in_dim=3, hidden_dim=4096, out_dim=1), RBF_Network(in_dim=256*256*3, hidden_dim=200, out_dim=1)]
    
    image_based_classification_model_type_functions=[non_nn_image_based_classification_model_type_functions, nn_image_based_classification_model_type_functions]    
    
    classification_model_type_functions=[sequence_based_classification_model_type_functions, image_based_classification_model_type_functions]
    
    model_type_strings=[regression_model_type_strings, clustering_model_type_strings, classification_model_type_strings]
    
    model_type_functions=[regression_model_type_functions, clustering_model_type_functions, classification_model_type_functions]
    
    model_types=[]
    
    for i_0, task_type_model_type_strings in enumerate(model_type_strings):
        
        task_type_model_type_functions=model_type_functions[i_0]

        task_type_model_types=[]
        
        for i_1, dataset_type_based_task_type_model_type_strings in enumerate(task_type_model_type_strings):
            
            dataset_type_based_task_type_model_type_functions=task_type_model_type_functions[i_1]
            
            dataset_type_based_task_type_model_types=[]
            
            if i_0==0:
                
                
                
                linear_dataset_type_based_task_type_model_type_strings=dataset_type_based_task_type_model_type_strings[0]
                
                linear_dataset_type_based_task_type_model_type_functions=dataset_type_based_task_type_model_type_functions[0]
                
                linear_dataset_type_based_task_type_model_types=[]
                
                for i_2, linear_dataset_type_based_task_type_model_type_string in enumerate(linear_dataset_type_based_task_type_model_type_strings):
                    
                    linear_dataset_type_based_task_type_model_type_function=linear_dataset_type_based_task_type_model_type_functions[i_2]

                    linear_dataset_type_based_task_type_model_type=Function_Type(string=linear_dataset_type_based_task_type_model_type_string, function=linear_dataset_type_based_task_type_model_type_function, domain=None)
                    
                    linear_dataset_type_based_task_type_model_types.append(linear_dataset_type_based_task_type_model_type)
                    
                dataset_type_based_task_type_model_types.append(linear_dataset_type_based_task_type_model_types)
                
                nonlinear_dataset_type_based_task_type_model_type_strings=dataset_type_based_task_type_model_type_strings[1]
                
                nonlinear_dataset_type_based_task_type_model_type_functions=dataset_type_based_task_type_model_type_functions[1]
                
                nonlinear_dataset_type_based_task_type_model_types=[]
                
                for i_2, nonlinear_dataset_type_based_task_type_model_type_string in enumerate(nonlinear_dataset_type_based_task_type_model_type_strings):
                    
                    nonlinear_dataset_type_based_task_type_model_type_function=nonlinear_dataset_type_based_task_type_model_type_functions[i_2]

                    nonlinear_dataset_type_based_task_type_model_type=Function_Type(string=nonlinear_dataset_type_based_task_type_model_type_string, function=nonlinear_dataset_type_based_task_type_model_type_function, domain=None)
                    
                    nonlinear_dataset_type_based_task_type_model_types.append(nonlinear_dataset_type_based_task_type_model_type)
                    
                dataset_type_based_task_type_model_types.append(nonlinear_dataset_type_based_task_type_model_types)
            
                nn_dataset_type_based_task_type_model_type_strings=dataset_type_based_task_type_model_type_strings[2]
                
                nn_dataset_type_based_task_type_model_type_functions=dataset_type_based_task_type_model_type_functions[2]
                
                nn_dataset_type_based_task_type_model_types=[]
                
                for i_2, nn_dataset_type_based_task_type_model_type_string in enumerate(nn_dataset_type_based_task_type_model_type_strings):
                    
                    nn_dataset_type_based_task_type_model_type_function=nn_dataset_type_based_task_type_model_type_functions[i_2]

                    nn_dataset_type_based_task_type_model_type=Function_Type(string=nn_dataset_type_based_task_type_model_type_string, function=nn_dataset_type_based_task_type_model_type_function, domain=None)
                    
                    nn_dataset_type_based_task_type_model_types.append(nn_dataset_type_based_task_type_model_type)
                    
                dataset_type_based_task_type_model_types.append(nn_dataset_type_based_task_type_model_types)
            
            elif i_0==1:
                
                
                
                for i_2, dataset_type_based_task_type_model_type_string in enumerate(dataset_type_based_task_type_model_type_strings):
                    
                    dataset_type_based_task_type_model_type_function=dataset_type_based_task_type_model_type_functions[i_2]

                    dataset_type_based_task_type_model_type=Function_Type(string=dataset_type_based_task_type_model_type_string, function=dataset_type_based_task_type_model_type_function, domain=None)
                    
                    dataset_type_based_task_type_model_types.append(dataset_type_based_task_type_model_type)
                    
            
            
            else:
                
                non_nn_dataset_type_based_task_type_model_type_strings=dataset_type_based_task_type_model_type_strings[0]
                
                non_nn_dataset_type_based_task_type_model_type_functions=dataset_type_based_task_type_model_type_functions[0]
                
                non_nn_dataset_type_based_task_type_model_types=[]
                
                for i_2, non_nn_dataset_type_based_task_type_model_type_string in enumerate(non_nn_dataset_type_based_task_type_model_type_strings):
                    
                    non_nn_dataset_type_based_task_type_model_type_function=non_nn_dataset_type_based_task_type_model_type_functions[i_2]

                    non_nn_dataset_type_based_task_type_model_type=Function_Type(string=non_nn_dataset_type_based_task_type_model_type_string, function=non_nn_dataset_type_based_task_type_model_type_function, domain=None)
                    
                    non_nn_dataset_type_based_task_type_model_types.append(non_nn_dataset_type_based_task_type_model_type)
                    
                dataset_type_based_task_type_model_types.append(non_nn_dataset_type_based_task_type_model_types)
                
                nn_dataset_type_based_task_type_model_type_strings=dataset_type_based_task_type_model_type_strings[1]
                
                nn_dataset_type_based_task_type_model_type_functions=dataset_type_based_task_type_model_type_functions[1]
                
                nn_dataset_type_based_task_type_model_types=[]
                
                for i_2, nn_dataset_type_based_task_type_model_type_string in enumerate(nn_dataset_type_based_task_type_model_type_strings):
                    
                    nn_dataset_type_based_task_type_model_type_function=nn_dataset_type_based_task_type_model_type_functions[i_2]

                    nn_dataset_type_based_task_type_model_type=Function_Type(string=nn_dataset_type_based_task_type_model_type_string, function=nn_dataset_type_based_task_type_model_type_function, domain=None)
                    
                    nn_dataset_type_based_task_type_model_types.append(nn_dataset_type_based_task_type_model_type)
                    
                dataset_type_based_task_type_model_types.append(nn_dataset_type_based_task_type_model_types)
            
            task_type_model_types.append(dataset_type_based_task_type_model_types)
            
        model_types.append(task_type_model_types)
        
    return model_types

"""
A class to hold polytopes in H-representation.

Francesc Font-Clos
Oct 2018
"""


class Polytope(object):
    """A polytope in H-representation."""

    def __init__(self, A=None, b=None):
        """
        Create a polytope in H-representation.

        The polytope is defined as the set of
        points x in Rn such that

        A x <= b

        """
        # dimensionality verifications
        assert A is not None and b is not None
        assert len(b.shape) == 1
        assert len(A.shape) == 2
        assert A.shape[0] == len(b)
        # store data
        self.A = A
        self.b = b
        self.dim = A.shape[1]
        self.nplanes = A.shape[0]
        self._find_auxiliar_points_in_planes()

    def check_inside(self, point):
        """Check if a point is inside the polytope."""
        checks = self.A@point <= self.b
        check = np.all(checks)
        return check

    def _find_auxiliar_points_in_planes(self):
        """Find an auxiliar point for each plane."""
        aux_points = [self._find_auxiliar_point(self.A[i],
                                                self.b[i])
                      for i in range(self.nplanes)]
        self.auxiliar_points = aux_points

    def _find_auxiliar_point(self, Ai, bi):
        """Find an auxiliar point for one plane."""
        p = np.zeros(self.dim)
        j = np.argmax(Ai != 0)
        p[j] = bi / Ai[j]
        return p


"""
MinOver algorithm to find a point inside a polytope.

Francesc Font-Clos
Oct 2018
"""


class MinOver(object):
    """MinOver solver."""

    def __init__(self, polytope, ):
        """
        Create a MinOver solver.

        Parameters
        ----------
        polytope: hitandrun.polytope
            Polytope in H-representation

        """
        self.polytope = polytope

    def run(self, speed=1, starting_point=None, max_iters=100, verbose=False):
        """
        Run the MinOver algorithm.

        Parameters
        ----------
        speed: float
            Distance moved at each learning step
        max_iters: int
            Maximum number of iterations (per hyperplan).
        starting_poin: np.array
            Initial condition.

        Returns
        -------
        current: np.array
            The final point.
        convergence: bool
            True if the algorithm converged, False, otherwise.

        """
        self.max_iters = max_iters * self.polytope.nplanes
        self.speed = speed
        if starting_point is None:
            self.current = np.zeros(self.polytope.dim)
        else:
            self.current = starting_point
        # compute step 0 worst planes
        # this is a trick to handle first steps
        self.worst_indexes = [-1, -2]
        self.worst_distances = [-1, -2]
        self._set_worst_constraint()
        for i in range(self.max_iters):
            convergence = self._step()
            self.iter = i
            self._check_speed()
            if verbose:
                self._print_worst()
            if convergence:
                break
        return self.current, convergence

    def _step(self):
        self._move_towards_worst_plane()
        self._set_worst_constraint()
        return np.all(self.distances < 0)

    def _check_speed(self):
        i0, i1, i2 = self.worst_indexes[::-1][:3]
        d0, d1, d2 = self.worst_distances[::-1][:3]
        if i0 != i1 and i0 == i2 and d0 >= d2:
            self.speed *= 0.9

    def _set_worst_constraint(self):
        self.distances = self.polytope.A @ self.current - self.polytope.b
        self.worst = np.argmax(self.distances)
        self.worst_indexes.append(self.worst)
        self.worst_distances.append(self.distances[self.worst])

    def _move_towards_worst_plane(self):
        self.current = self.current - self.speed * self.polytope.A[self.worst]

    def _print_worst(self):
        worst_distance = self.distances[self.worst]
        print("iter", self.iter,
              "index:", self.worst,
              "distance:", worst_distance,
              "speed:", self.speed)


"""
Hit-and-run sampler.

Francesc Font-Clos
Oct 2018
"""
from scipy.spatial.distance import norm


class HitAndRun(object):
    """Hit-and-run sampler."""

    def __init__(self, polytope=None, starting_point=None,
                 n_samples=100, thin=1):
        """
        Create a hit-and-run sampler.

        Parameters
        ----------
        polytope: hitandrun.polytope
            The convex polytope to be sampled.
        starting_point: np.array
            Initial condition. Must be inside the polytope.
        n_samples: int
            Number of desired samples.
        thin : int
            Thinning factor, increase to get independent samples.

        """
        # make sure we got a point inside the polytope
        assert starting_point is not None
        assert len(starting_point) == polytope.dim
        assert polytope.check_inside(starting_point)

        self.polytope = polytope
        self.starting_point = starting_point
        self.n_samples = n_samples
        self.thin = thin
        # place starting point as current point
        self.current = starting_point
        # set a starting random direction
        self._set_random_direction()
        # create empty list of samples
        self.samples = []

    def get_samples(self, n_samples=None, thin=None):
        """Get the requested samples."""
        self.samples = []
        if n_samples is not None:
            self.n_samples = n_samples
        if thin is not None:
            self.thin = thin

        # keep only one every thin
        for i in tqdm.tqdm(
            range(self.n_samples),
            desc="hit-and-run steps:"
        ):
            for _ in range(self.thin):
                self._step()
            self._add_current_to_samples()
        return np.array(self.samples)

    # private functions
    def _step(self):
        """Make one step."""
        # set random direction
        self._set_random_direction()
        # find lambdas
        self._find_lambdas()
        # find smallest positive and negative lambdas
        try:
            lam_plus = np.min(self.lambdas[self.lambdas > 0])
            lam_minus = np.max(self.lambdas[self.lambdas < 0])
        except(Exception):
            raise RuntimeError("The current direction does not intersect"
                               "any of the hyperplanes.")
        # throw random point between lambdas
        lam = np.random.uniform(low=lam_minus, high=lam_plus)
        # compute new point and add it
        new_point = self.current + lam * self.direction
        self.current = new_point

    def _find_lambdas(self):
        """
        Find the lambda value for each hyperplane.

        The lambda value is the distance we have to travel
        in the current direction, from the current point, to
        reach a given hyperplane.
        """
        A = self.polytope.A
        p = self.polytope.auxiliar_points

        lambdas = []
        for i in range(self.polytope.nplanes):
            if np.isclose(self.direction @ A[i], 0):
                lambdas.append(np.nan)
            else:
                lam = ((p[i] - self.current) @ A[i]) / (self.direction @ A[i])
                lambdas.append(lam)
        self.lambdas = np.array(lambdas)

    def _set_random_direction(self):
        """Set a unitary random direction in which to travel."""
        direction = np.random.randn(self.polytope.dim)
        self.direction = direction / norm(direction)

    def _add_current_to_samples(self):
        self.samples.append(list(self.current))

def A_Function(ddr, num_ddr_set_elements):
    
    A=np.zeros((2*num_ddr_set_elements, num_ddr_set_elements-1))
    
    for i_0 in range(num_ddr_set_elements-1):
        
        A_row_type_1=np.zeros(num_ddr_set_elements-1)
        
        A_row_type_2=np.zeros(num_ddr_set_elements-1)
        
        A_row_type_1[i_0]=-1

        A_row_type_2[i_0]=1
        
        A[i_0]=A_row_type_1

        A[num_ddr_set_elements-1+i_0]=A_row_type_2

    A[2*num_ddr_set_elements-2]=-np.ones(num_ddr_set_elements-1)

    A[2*num_ddr_set_elements-1]=np.ones(num_ddr_set_elements-1)        
            
    return A

def b_Function(ddr, num_ddr_set_elements):
    
    b=np.zeros(2*num_ddr_set_elements)
    
    for i in range(num_ddr_set_elements-1):
        
        b[i]=0
        
        b[num_ddr_set_elements-1+i]=1
        
    b[2*num_ddr_set_elements-2]=1-num_ddr_set_elements*ddr
    
    b[2*num_ddr_set_elements-1]=num_ddr_set_elements*ddr
    
    return b

def DDRs():
    
    ddrs=np.linspace(start=1/10, stop=1, num=10)
    
    ddr_sets_sets=[]
    
    for i_0,ddr in enumerate(ddrs):
        
        if ddr!=1:
            
            A=A_Function(ddr, 10)
            
            b=b_Function(ddr, 10)
            
            polytope=Polytope(A=A, b=b)
            
            min_over=MinOver(polytope=polytope)
            
            (starting_point, condition)=min_over.run(speed=1, starting_point=np.zeros(10-1))
            
            hit_and_run=HitAndRun(polytope=polytope, starting_point=starting_point, n_samples=9, thin=1)
            
            ddr_sets=hit_and_run.get_samples(n_samples=10, thin=1).tolist()
            
            last_ddr_set_elements=[10*ddr-sum(ddr_set) for ddr_set in ddr_sets]
            
            new_ddr_sets=[] 
            
            for i, ddr_set in enumerate(ddr_sets):
                
                last_ddr_set_element=last_ddr_set_elements[i]
                
                ddr_set.append(last_ddr_set_element)
                
                shuffle(ddr_set)
            
                new_ddr_sets.append(ddr_set)
                
            shuffle(new_ddr_sets)
            
            ddr_sets=np.array(new_ddr_sets)
            
        else:
            
            ddr_sets=np.array([np.ones(10)])
            
        ddr_sets_sets.append(ddr_sets)
    
    ddrs=[ddrs, ddr_sets_sets]
    
    return ddrs
            
def Variables():
    """Returns the initialized variables."""
    print("Initialize Task Types")
    task_types=Task_Types()
    print("Initialize datasets")
    dataset_types=Dataset_Types()
    print("Initialize Deterministic Types")
    deterministic_types=Deterministic_Types()
    print("Initialize Nondeterministic Types")
    nondeterministic_types=Nondeterministic_Types()
    print("Initialize evaluation metric types")
    evaluation_metric_types=Evaluation_Metric_Types()
    print("Initialize Model Types")
    model_types=Model_Types()
    print("Initialize DDRs")
    ddrs=DDRs()
    
    variables=[task_types, dataset_types, deterministic_types, nondeterministic_types, evaluation_metric_types, model_types, ddrs]
    
    return variables
    

def Deterministic_Types_Data(variables):
    """Generate Data using Deterministic Types"""
    deterministic_types=variables[0]
    
    deterministic_types_data=[]
   
    targets=[]
    for i_0, task_type_deterministic_types in enumerate(deterministic_types):
        task_type_deterministic_types_data=[]
        
        task_type_targets=[]
        for i_1,dataset_type_based_task_type_deterministic_types in enumerate(task_type_deterministic_types):
            dataset_type_based_task_type_deterministic_types_data=[]
    
            dataset_type_based_task_type_targets=[]
            for dataset_type_based_task_type_deterministic_type in dataset_type_based_task_type_deterministic_types:
                #dataset_type_based_task_type_deterministic_type_component_types_data=[]
                #for dataset_type_based_task_type_deterministic_type_component_type in dataset_type_based_task_type_deterministic_type_component_types:    
                function=dataset_type_based_task_type_deterministic_type.function
                domain=dataset_type_based_task_type_deterministic_type.domain
                dataset_type_based_task_type_deterministic_type_data=[]
                
                function_dataset_type_based_task_type_targets=[]
                for i_2, func in enumerate(function):
                    if i_0==0 and i_1==0:
                        dataset_type_based_task_type_deterministic_type_data.append(domain[i_2])
                        
                        function_dataset_type_based_task_type_targets.append(func(domain[i_2]))
                    elif i_0==0:
                        dataset_type_based_task_type_deterministic_type_data.append(torch.stack([func(x)[0] for x in domain[i_2]]))
                        
                        function_dataset_type_based_task_type_targets.append(torch.stack([func(x)[1] for x in domain[i_2]]))
                    elif i_0==1 and i_1==0:
                        dataset_type_based_task_type_deterministic_type_data.append(func(domain[i_2]))

                        function_dataset_type_based_task_type_targets.append(np.array(10*[i_2]))
                    else:
                        dataset_type_based_task_type_deterministic_type_data.append(torch.stack([func(x)[0] for x in domain[i_2]]))
                        
                        function_dataset_type_based_task_type_targets.append(torch.stack([torch.tensor(func(x)[1]) for x in domain[i_2]]))
                        
                dataset_type_based_task_type_deterministic_types_data.append(dataset_type_based_task_type_deterministic_type_data)
                dataset_type_based_task_type_targets.append(function_dataset_type_based_task_type_targets)
                    
            task_type_deterministic_types_data.append(dataset_type_based_task_type_deterministic_types_data)
        
            task_type_targets.append(dataset_type_based_task_type_targets)
        deterministic_types_data.append(task_type_deterministic_types_data)
    
        targets.append(task_type_targets)
    
    return [deterministic_types_data, targets]
   
def Train_Test_Split(data):
    """Split Data into Train Data and Test Data"""
    features=data[0]
    targets=data[1]
    features_train=[]
    features_test=[]
    targets_train=[]
    targets_test=[]
    for i_0,task_type_features in enumerate(features):
        task_type_targets=targets[i_0]
        
        task_type_features_train=[]
        task_type_features_test=[]
        task_type_targets_train=[]
        task_type_targets_test=[]
        for i_1, dataset_type_based_task_type_features in enumerate(task_type_features):
            dataset_type_based_task_type_targets=task_type_targets[i_1]
            
            dataset_type_based_task_type_features_train=[]
            dataset_type_based_task_type_features_test=[]
            dataset_type_based_task_type_targets_train=[]
            dataset_type_based_task_type_targets_test=[]
            for i_2, dataset_type_based_task_type_deterministic_type_features in enumerate(dataset_type_based_task_type_features):
                dataset_type_based_task_type_deterministic_type_targets=dataset_type_based_task_type_targets[i_2]
                
                dataset_type_based_task_type_deterministic_type_features_train=[]
                dataset_type_based_task_type_deterministic_type_features_test=[]
                dataset_type_based_task_type_deterministic_type_targets_train=[]
                dataset_type_based_task_type_deterministic_type_targets_test=[]
                for i_3, dataset_type_based_task_type_deterministic_type_function_features in enumerate(dataset_type_based_task_type_deterministic_type_features):
                    dataset_type_based_task_type_deterministic_type_function_targets=dataset_type_based_task_type_deterministic_type_targets[i_3]
                    dataset_type_based_task_type_deterministic_type_function_features_train, dataset_type_based_task_type_deterministic_type_function_features_test, dataset_type_based_task_type_deterministic_type_function_targets_train, dataset_type_based_task_type_deterministic_type_function_targets_test=train_test_split(dataset_type_based_task_type_deterministic_type_function_features, dataset_type_based_task_type_deterministic_type_function_targets, test_size=0.2, random_state=0)
                    
                    dataset_type_based_task_type_deterministic_type_features_train.append(dataset_type_based_task_type_deterministic_type_function_features_train)
                    dataset_type_based_task_type_deterministic_type_features_test.append(dataset_type_based_task_type_deterministic_type_function_features_test)
                    dataset_type_based_task_type_deterministic_type_targets_train.append(dataset_type_based_task_type_deterministic_type_function_targets_train)
                    dataset_type_based_task_type_deterministic_type_targets_test.append(dataset_type_based_task_type_deterministic_type_function_targets_test)
                dataset_type_based_task_type_features_train.append(dataset_type_based_task_type_deterministic_type_features_train)
                dataset_type_based_task_type_features_test.append(dataset_type_based_task_type_deterministic_type_features_test)
                dataset_type_based_task_type_targets_train.append(dataset_type_based_task_type_deterministic_type_targets_train)
                dataset_type_based_task_type_targets_test.append(dataset_type_based_task_type_deterministic_type_targets_test)
            task_type_features_train.append(dataset_type_based_task_type_features_train)
            task_type_features_test.append(dataset_type_based_task_type_features_test)
            task_type_targets_train.append(dataset_type_based_task_type_targets_train)
            task_type_targets_test.append(dataset_type_based_task_type_targets_test)
        features_train.append(task_type_features_train)
        features_test.append(task_type_features_test)
        targets_train.append(task_type_targets_train)
        targets_test.append(task_type_targets_test)
    features=[features_train, features_test]
    targets=[targets_train, targets_test]
    data=[features, targets]
    
    return data

def Processed_Data(data):
    """Process the data."""
    data=Train_Test_Split(data)
    data=DDR_Invariant_Target_Standardization
    
    features=data[0]
    targets=data[1]
    for i_0, features_type in enumerate(features):
        targets_type=targets[i_0]
        for i_1, task_type_features_type in enumerate(features_type):
            task_type_targets_type=targets_type[i_1]
            for i_2, dataset_type_based_task_type_features_type in enumerate(task_type_features_type):
                dataset_type_based_task_type_targets_type=task_type_targets_type[i_2]
                for i_3, dataset_type_based_task_type_deterministic_type_features_type in enumerate(dataset_type_based_task_type_features_type):
                    dataset_type_based_task_type_deterministic_type_targets_type=dataset_type_based_task_type_targets_type[i_3]
                    
    

#Start Experiment
print("Start Experiment")

#Initialize Variables
print("Initialize Variables")
variables=Variables()

print("\nGenerate Deterministic Types Data")
data=Deterministic_Types_Data(variables=[variables[i] for i in [2,3,6]])

print("Process Data")
processed_data=Processed_Data(data)
