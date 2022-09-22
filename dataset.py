"""
Code provided with the paper:
"Attribute Prediction as Multiple Instance Learning"
Dataset classes
22-09-2022
"""

import os
import numpy as np
from PIL import Image
from skimage import io
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
from torch import nn
from skimage.transform import resize
from skimage.color import rgba2rgb
import pdb
import scipy.io
import random
import math


class CUB():
    """CUB dataset."""

    def __init__(self, root_path, do_val=False):
        # Find combinations of noun+adjective to build the attributes
        with open(root_path + 'CUB_200_2011/attributes/attributes.txt', 'r') as text:
            content = text.read()
            attr_names = [b.split(' ')[1].split('::') for b in content.split('\n')[0:-1]]
            nouns_orig, adjs_orig = zip(*attr_names)
            adjs_orig = list(adjs_orig)
            nouns_orig = list(nouns_orig)
            adjs = list(set(adjs_orig))
            adjs.sort()
            nouns = list(set(nouns_orig))
            nouns.sort()
            elems = nouns + adjs

            combine = np.zeros([len(attr_names), len(elems)])
            for j in range(len(adjs_orig)):
                line = np.array([adjs_orig[j] == elems[i] for i in range(len(elems))])
                line += np.array([nouns_orig[j] == elems[i] for i in range(len(elems))])
                combine[j, :] = line

        with open(root_path + 'CUB_200_2011/train_test_split.txt', 'r') as text:
            content = text.read()
            traintest = [b.split(' ')[1] for b in content.split('\n')[:-1]]
        with open(root_path + 'CUB_200_2011/images.txt', 'r') as text:
            content = text.read()
            images = [b.split(' ') for b in content.split('\n')]
        with open(root_path + 'CUB_200_2011/attributes/class_attribute_labels_continuous.txt', 'r') as text:
            content = text.read()
            class_attr = [b.split(' ') for b in content.split('\n')]
            class_attr = class_attr[0:-1]
            class_attr = np.array([[float(b) for b in line] for line in class_attr])
        with open(root_path + 'CUB_200_2011/attributes/image_attribute_labels.txt', 'r') as text:
            content = text.read()
            img_attr = [b.split(' ') for b in content.split('\n')]
            img_attr = img_attr[0:-1]
            new_list = []
            for line in img_attr:
                if len(line) == 7 or len(line) == 6:
                    new_line = line[0:4]
                    new_line.append(line[-1])
                    line = new_line
                new_list.append([float(b) for b in line])
            img_attr = np.array(new_list)

        data_train = []
        data_test = []
        labels_train = []
        labels_test = []
        attr_train = []
        attr_test = []
        names_train = []
        names_test = []

        pbar = tqdm(total=len(images), position=0, leave=True)
        zsl_test_classes = [43, 111, 23, 98, 55, 130, 139, 123, 156, 124, 200, 72, 173, 28, 119, 165, 103, 180, 77, 12,
                            45, 190, 191, 138, 157, 52, 33, 164, 31, 143, 94, 70, 97, 91, 104, 127, 161, 49, 169, 148,
                            113, 87, 163, 136, 188, 84, 26, 4, 132, 168]
        zsl_test_classes = [cl - 1 for cl in zsl_test_classes]
        if do_val:
            zsl_val_classes = [76, 117, 150, 182, 140, 69, 48, 114, 109, 5, 51, 144, 177, 153, 189, 151, 162, 89, 155, 59, 66, 184, 
                            198, 63, 194, 195, 83, 80, 36, 18, 56, 192, 128, 73, 121, 34, 64, 174, 118, 61, 116, 29, 9, 158, 
                            86, 179, 74, 7, 100, 141]
            zsl_val_classes = [cl - 1 for cl in zsl_val_classes]
            all_classes_set = set(range(200))
            zsl_train_classes = list(all_classes_set - set(zsl_val_classes) - set(zsl_test_classes))
            # We are doing val, so the val split is used for the test set and the test set is not used
            zsl_test_classes = zsl_val_classes
        else:
            all_classes_set = set(range(200))
            zsl_train_classes = list(all_classes_set - set(zsl_test_classes))

        # Reassign ZSL unseen classes and split test set into test and validation set
        traintest = list(map(int, traintest))

        for i in range(len(images)-1):
            bird_class = np.int(images[i][1][0:3]) - 1
            traintest[i] *= bird_class in zsl_train_classes


        for i in range(len(images) - 1):
            bird_class = np.int(images[i][1][0:3]) - 1
            bird_attr = img_attr[img_attr[:, 0] == i + 1, :]
            bird_attr = bird_attr[:, 2] * bird_attr[:, 3] / 4


            im = root_path + 'CUB_200_2011/images/' + images[i][1]

            is_train = -1
            if traintest[i] == 1:
                is_train = 1
            if traintest[i] == 0:
                is_train = 0    
            if bird_class in zsl_test_classes:
                is_train = 0
            if bird_class not in zsl_train_classes and bird_class not in zsl_test_classes:
                is_train = -1


            if is_train == 1:
                data_train.append(im)
                labels_train.append(bird_class)
                attr_train.append(bird_attr)
                names_train.append(images[i][1].split('/')[1])
            elif is_train == 0:
                data_test.append(im)
                labels_test.append(bird_class)
                attr_test.append(bird_attr)
                names_test.append(images[i][1].split('/')[1])

            pbar.update()

        train = list(zip(data_train, labels_train, attr_train, names_train))
        test = list(zip(data_test, labels_test, attr_test, names_test))

        attr_names = self.renameAttributes(attr_names)
        rel_matrix, attr_names = self.createRelMatrix(attr_names)

        self.train = train
        self.test = test
        self.class_attr = class_attr
        self.zsl_train_classes = zsl_train_classes
        self.zsl_test_classes = zsl_test_classes
        self.attr_names = attr_names
        self.rel_matrix = rel_matrix

    def renameAttributes(self, attr_names):
        # Rename attributes
        parts = ['belly', 'wing', 'bill', 'head', 'tail', 'throat', 'breast', 'back', 'leg']
        for i in range(len(attr_names)):
            for part in parts:
                if part in attr_names[i][0]:
                    attr_names[i][0] = part
        return attr_names

    def createRelMatrix(self, attr_names):
        # Build relational matrix
        rel_matrix = torch.zeros((len(attr_names), len(attr_names)))
        for i in range(len(attr_names)):
            for j in range(len(attr_names)):
                for idx in range(len(attr_names[i])):
                    word = attr_names[i][idx]
                    if word in attr_names[j]:
                        if idx == 0:
                            rel_matrix[i, j] = 1
                        if idx == 1:
                            rel_matrix[i, j] = 1
        return rel_matrix, attr_names

    def getData(self):
        return self.train, self.test, self.class_attr, self.zsl_train_classes, self.zsl_test_classes, self.attr_names, self.rel_matrix


class SUN():
    """SUN dataset."""

    def __init__(self, root_path, do_val=False):
        data_path = root_path + 'SUNAttributes/'
        ## Generate lists of paths, class names, class IDs and attributes
        # Create empty lists for storing values

        im_paths = []
        class_names = []
        im_class_ids = []
        attr_names = []

        # Load attr list from .mat file
        im_attrs = scipy.io.loadmat(data_path + 'attributeLabels_continuous.mat')['labels_cv']

        # Load class names and paths from .mat file
        paths_array = scipy.io.loadmat(data_path + 'images.mat')['images']
        for i in range(len(paths_array)):
            # Get path including folder structure
            path = paths_array[i][0].item()

            # Get class name from path
            name_list = path.split(sep='/')[1:-1]
            class_name = "_".join(name_list)

            # Build list of class names
            if class_name not in class_names:
                class_names.append(class_name)

            # Turn path into full path
            path = data_path + 'images/' + path

            # Append both to respective lists
            im_paths.append(path)

        # Sort class names in alphabetical order
        class_names.sort()

        # Create class id list
        for i in range(len(paths_array)):
            # Get class name from path
            path = paths_array[i][0].item()
            name_list = path.split(sep='/')[1:-1]
            class_name = "_".join(name_list)

            # Save class id
            class_id = class_names.index(class_name)
            im_class_ids.append(class_id)

        # Create attr name list
        attr_name_array = scipy.io.loadmat(data_path + 'attributes.mat')['attributes']
        for i in range(len(attr_name_array)):
            # Get path including folder structure
            attr = attr_name_array[i][0].item()
            attr_names.append(attr)

        attr_names.sort()

        # Build lists of images, labels, attributes and names for train, val, and test
        data_train = []
        data_test = []
        labels_train = []
        labels_test = []
        attr_train = []
        attr_test = []
        names_train = []
        names_test = []


        pbar = tqdm(total=len(im_paths), position=0, leave=True)
        zsl_test_class_names = ["airlock", "alley", "archive", "arena_basketball", "artists_loft", "auditorium",
                "ballroom", "bank_vault", "batting_cage_outdoor", "bazaar_indoor", "bazaar_outdoor",
                "betting_shop", "bog", "bow_window_indoor", "brewery_indoor", "bus_depot_outdoor",
                "canal_natural", "car_interior_frontseat", "casino_outdoor", "cemetery",
                "chemistry_lab", "church_indoor", "corral", "cubicle_office", "dirt_track",
                "doorway_indoor", "elevator_interior", "excavation", "exhibition_hall",
                "field_cultivated", "firing_range_indoor", "fishpond", "galley", "geodesic_dome_indoor",
                "hangar_indoor", "hoodoo", "hotel_room", "ice_shelf", "jacuzzi_indoor",
                "japanese_garden", "landing_deck", "lawn", "monastery_outdoor", "mosque_indoor",
                "motel", "nightclub", "observatory_outdoor", "parking_lot", "piano_store",
                "playground", "promenade_deck", "pub_indoor", "racecourse", "rectory", "sandbox",
                "savanna", "ski_resort", "temple_south_asia", "theater_indoor_seats", "ticket_booth",
                "trading_floor", "train_station_platform", "tundra", "tunnel_road_outdoor", "vestry",
                "vineyard", "volleyball_court_outdoor", "watering_hole", "workshop",
                "wrestling_ring_indoor", "yard", "ziggurat"]
        if do_val:
            zsl_val_class_names = ["pilothouse_indoor","factory_indoor","lift_bridge","cottage","lab_classroom","stilt_house_water",
                "dry_dock","dam","shoe_shop","kennel_indoor","organ_loft_exterior","balcony_exterior","schoolhouse",
                "amusement_arcade","cafeteria","mobile_home","trestle_bridge","hot_tub_outdoor","sky","tunnel_rail_outdoor",
                "inn_indoor","lock_chamber","bow_window_outdoor","underwater_wreck","great_hall","gatehouse","plaza",
                "synagogue_outdoor","staircase","wine_cellar_barrel_storage","waiting_room","rock_arch","delicatessen",
                "cloister_indoor","city","auto_factory","darkroom","computer_room","industrial_park","packaging_plant",
                "garbage_dump","baptistry_indoor","garage_indoor","viaduct","factory_outdoor","weighbridge","science_museum",
                "market_indoor","moor","chapel","subway_station_platform","assembly_line","banquet_hall","valley",
                "discotheque","waterfall_cascade","podium_outdoor","farm","ski_lodge","theater_outdoor","quonset_hut_outdoor",
                "parade_ground","basketball_court_indoor","art_school","checkout_counter"]
            # Define lists with class ids for zsl train, validation and test classes according to the train/val split
            all_classes_set = set(range(len(class_names)))
            zsl_test_classes = [class_names.index(cl) for cl in zsl_test_class_names]
            zsl_val_classes = [class_names.index(cl) for cl in zsl_val_class_names]
            zsl_train_classes = list(all_classes_set - set(zsl_val_classes) - set(zsl_test_classes))
            assert(len(zsl_test_classes) + len(zsl_val_classes) + len(zsl_train_classes) == len(class_names))
            # We are doing val, so the val split is used for the test set and the test set is not used
            zsl_test_classes = zsl_val_classes
        else:
            # Define lists with class ids for zsl train and test classes
            all_classes_set = set(range(len(class_names)))
            zsl_test_classes = [class_names.index(cl) for cl in zsl_test_class_names]
            zsl_train_classes = list(all_classes_set - set(zsl_test_classes))
            assert(len(zsl_test_classes) + len(zsl_train_classes) == len(class_names))


        #Split the data according to the defined data split
        for i in range(len(im_paths) - 1):

            im_class = im_class_ids[i]
            im_attr = im_attrs[i]
            im_path = im_paths[i]
            im_name = im_path.split('/')[-1]

            is_train = -1
            if im_class in zsl_test_classes:
                is_train = 0
            if im_class in zsl_train_classes:
                is_train = np.int(np.random.rand() > 0.2)


            if is_train == 1:
                data_train.append(im_path)
                labels_train.append(im_class)
                attr_train.append(im_attr)
                names_train.append(im_name)
            elif is_train == 0:
                data_test.append(im_path)
                labels_test.append(im_class)
                attr_test.append(im_attr)
                names_test.append(im_name)

            pbar.update()

        # Split dataset
        train = list(zip(data_train, labels_train, attr_train, names_train))
        test = list(zip(data_test, labels_test, attr_test, names_test))

        class_attr = np.zeros((len(class_names), im_attrs.shape[1]))
        for class_id, attrlist in zip(im_class_ids, im_attrs):
            class_attr[class_id] += attrlist

        # Rescale 0-100, with 100 being class max
        max_per_class = class_attr.max(axis=1)
        class_attr = 100*class_attr / max_per_class[:, np.newaxis]

        rel_matrix = self.createRelMatrix(attr_names)

        print(len(train), len(test))
        print(len(train)/len(train), len(test)/len(train))
        self.train = train
        self.test = test

        self.zsl_train_classes = zsl_train_classes
        self.zsl_test_classes = zsl_test_classes

        self.class_attr = class_attr
        self.attr_names = attr_names
        self.rel_matrix = rel_matrix

    def createRelMatrix(self, attr_names):
        # Build relational matrix, identity matrix for SUN
        rel_matrix = torch.eye(len(attr_names))
        return rel_matrix

    def getData(self):
        return self.train, self.test, self.class_attr, self.zsl_train_classes, self.zsl_test_classes, self.attr_names, self.rel_matrix


class AWA2():
    def __init__(self, root_path, do_val=False, samples_per_class = 200):

        data_path = root_path + 'Animals_with_Attributes2/'

        im_paths = []
        im_class_ids = []
        im_class_names = []

        class_attr =  np.loadtxt(data_path + 'predicate-matrix-continuous.txt')  #> np.mean(np.loadtxt(data_path + 'predicate-matrix-continuous.txt'))

        # Load class names
        with open(data_path + 'classes.txt', 'r') as text:
            content = text.read()
            class_names = [b.strip().split('\t')[1] for b in content.split('\n')[:-1]]

        # Load attr names
        with open(data_path + 'predicates.txt', 'r') as text:
            content = text.read()
            attr_names = [b.strip().split('\t')[1] for b in content.split('\n')[:-1]]


        for folder in os.listdir(data_path + 'JPEGImages'):
            count = 0
            for file in os.listdir(data_path + 'JPEGImages/' + folder):
                if (file[-4:] == '.jpg'):  # check if the files are jpg files
                    to_add = count < samples_per_class
                    count += 1
                    if to_add:
                        path = os.path.join(data_path, 'JPEGImages/', folder, file)
                        im_paths.append(os.path.join(path))
                        im_class_names.append(folder)
                        im_class_ids.append(class_names.index(folder))


        data_train = []
        data_test = []
        labels_train = []
        labels_test = []
        attr_train = []
        attr_test = []
        names_train = []
        names_test = []

        pbar = tqdm(total=len(im_paths), position=0, leave=True)
        zsl_test_class_names = ['sheep', 'dolphin', 'bat', 'seal', 'blue+whale','rat', 'horse', 'walrus', 'giraffe', 'bobcat']
        if do_val:
            zsl_val_class_names = ["mole","beaver","deer","gorilla","chimpanzee","dalmatian","ox","giant+panda","leopard","hamster","moose","rabbit","raccoon"]
            # Define lists with class ids for zsl train, validation and test classes
            all_classes_set = set(range(len(class_names)))
            zsl_test_classes = [class_names.index(cl) for cl in zsl_test_class_names]
            zsl_val_classes = [class_names.index(cl) for cl in zsl_val_class_names]
            zsl_train_classes = list(all_classes_set - set(zsl_val_classes) - set(zsl_test_classes))
            assert (len(zsl_test_classes) + len(zsl_val_classes) + len(zsl_train_classes) == len(class_names))
            # We are doing val, so the val split is used for the test set and the test set is not used
            zsl_test_classes = zsl_val_classes
        else:
            # Define lists with class ids for zsl train and test classes
            all_classes_set = set(range(len(class_names)))
            zsl_test_classes = [class_names.index(cl) for cl in zsl_test_class_names]
            zsl_train_classes = list(all_classes_set - set(zsl_test_classes))
            assert(len(zsl_test_classes) + len(zsl_train_classes) == len(class_names))

         # Split the data according to the defined data split
        for i in range(len(im_paths) - 1):

            im_class = im_class_ids[i]
            im_attr = class_attr[im_class]
            im_path = im_paths[i]
            im_name = im_path.split('/')[-1]
            is_train = -1

            if im_class in zsl_test_classes:
                is_train = 0
            if im_class in zsl_train_classes:
                is_train = np.int(np.random.rand() > 0.2)

            if is_train == 1:
                data_train.append(im_path)
                labels_train.append(im_class)
                attr_train.append(im_attr)
                names_train.append(im_name)
            elif is_train == 0:
                data_test.append(im_path)
                labels_test.append(im_class)
                attr_test.append(im_attr)
                names_test.append(im_name)

            pbar.update()

        # Split dataset ------------------------------------------------------------------------------------------------
        train = list(zip(data_train, labels_train, attr_train, names_train))
        test = list(zip(data_test, labels_test, attr_test, names_test))

        rel_matrix = self.createRelMatrix(attr_names)

        self.train = train
        self.test = test

        self.zsl_train_classes = zsl_train_classes
        self.zsl_test_classes = zsl_test_classes

        self.class_attr = class_attr
        self.attr_names = attr_names
        self.rel_matrix = rel_matrix

    def createRelMatrix(self, attr_names):
        # Build relational matrix, identity matrix for SUN
        rel_matrix = torch.eye(len(attr_names))
        return rel_matrix

    def getData(self):
        return self.train, self.test, self.class_attr, self.zsl_train_classes, self.zsl_test_classes, self.attr_names, self.rel_matrix


class MILDataset(torch.utils.data.Dataset):
    """Dataset class that gets created for each training, validation and test set."""

    def __init__(self, zipped_data):
        """
        Args:
            zipped_data (list of lists): zipped_data[i] contains [image, class, attributes]
        """
        self.zipped_data = zipped_data
        self.transform = transforms.Compose([transforms.Resize([300, 300])])

    def __len__(self):
        return len(self.zipped_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.zipped_data[idx]

        im = io.imread(sample[0])

        # If grayscale image, change to rgb format
        if len(im.shape) == 2:
            im = np.stack((im,) * 3, axis=-1)

        # If RGBA, make RGB
        if im.shape[2] == 4:
            im = im[:, :, :3]

        im = resize(im, (300, 300), order=1)
        im = np.float32(np.transpose(im / 255, axes=(2, 0, 1)))

        im_class = sample[1]
        attr = np.float32(sample[2])

        return (im, im_class, attr, idx)
