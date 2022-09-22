This is the code to evaluate the body part localization accuracy for attribute attention maps.

1. Preparation:

Data: copy CUB_200_2011/images/ direction into ./data/images

2. Example: run script to calculate the IoU:
CUDA_VISIBLE_DEVICES=0 python ./main.py --cuda

run script to save attention maps and bounding boxes:
CUDA_VISIBLE_DEVICES=0 python ./main.py --cuda --save_img_num 3 --save_att './visualize_IoU/1/'
Examples are shown in ./visualize_IoU/1/

3. How to transfer you model into this code:

What you need is a model that can output the attribute attention map and predicted attribute value with following shape:
    pre_attri.shape : (batch_size * 312)
    maps.shape : (batch_size, 312, 7, 7) (batch_size * attribute_numbers * feature_map_size)

and a dataloader that can output the input (image), target (label), impath (the path for the image)
    The input image take the following transform:
    transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize,])

Then modify the code between line 181 to line 197 in ./model_proto_IoU_C24.py
