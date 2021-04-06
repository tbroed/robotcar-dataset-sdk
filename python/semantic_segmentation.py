import torch
import encoding
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Resize, ToPILImage

# https://hangzhang.org/PyTorch-Encoding/model_zoo/segmentation.html
# https://github.com/zhanghang1989/PyTorch-Encoding

# TODO: remove cuda parts
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

def get_model():
    # Get the model
    # TODO: try out DeepLab_ResNeSt101_ADE - EncNet_ResNet101s_ADE(previous)
    model = encoding.models.get_model('DeepLab_ResNeSt101_ADE', pretrained=True)  # .cuda()
    model.eval()
    return model

def predict_image(model, img):
    transform = ToTensor()
    trans_resize = Resize((333, 500))

    img = Image.fromarray(img)
    # img.show()

    # im_rgb = Image.open(filename).convert("RGB")

    orig_size = img.size

    im_resized = trans_resize(img)

    input = transform(im_resized).unsqueeze(0)  # TODO: im_resized wieder einsetzen

    # Make prediction
    output = model.evaluate(input)
    predict = torch.max(output, 1)[1].numpy() + 1  # .cpu()

    # Get color pallete for visualization
    mask = encoding.utils.get_mask_pallete(predict, 'pascal_voc')
    mask_resized = np.array(Resize((orig_size[1], orig_size[0]))(mask))
    # mask.save(os.path.join("output_stereo_ResNeSt_full", str(radar_timestamp) + '.png'))

    return mask_resized


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device: ", device)
#
# # Get the model
# # TODO: try out DeepLab_ResNeSt101_ADE - EncNet_ResNet101s_ADE(previous)
# model = encoding.models.get_model('DeepLab_ResNeSt101_ADE', pretrained=True) #.cuda()
# model.eval()
#
# # Prepare the image
# url = 'https://github.com/zhanghang1989/image-data/blob/master/' + \
#       'encoding/segmentation/pcontext/2010_001829_org.jpg?raw=true'
# filename = 'example.jpg'
# img = encoding.utils.load_image(
#     encoding.utils.download(url, filename)).unsqueeze(0) #.cuda()
#
#
#
# transform = ToTensor()
# trans_resize = Resize((333, 500))
#
# import os
#
# radar_timestamps = np.loadtxt('stereo.timestamps', delimiter=' ', usecols=[0], dtype=np.int64)
# mypath = "color_left_stereo"
#
# from os import walk
#
# _, _, filenames = next(walk(mypath))
#
#
# for image_filename in filenames:
#     radar_timestamp = image_filename.split('.')[0]
#     # radar_timestamp = 1547131077566419
#     filename = os.path.join("color_left_stereo", str(radar_timestamp) + '.png')
#     im_rgb = Image.open(filename).convert("RGB")
#     im_resized = trans_resize(im_rgb) # TODO: comment out and try if it sill works
#
#     input = transform(im_resized).unsqueeze(0) # TODO: im_resized wieder einsetzen
#     # im_tens = torch.from_numpy(im).unsqueeze(0)
#
#     # Make prediction
#     output = model.evaluate(input)
#     predict = torch.max(output, 1)[1].numpy() + 1 #.cpu()
#
#     # Get color pallete for visualization
#     mask = encoding.utils.get_mask_pallete(predict, 'pascal_voc')
#     mask.save(os.path.join("output_stereo_ResNeSt_full", str(radar_timestamp) + '.png'))
#
#     print("created " + str(radar_timestamp))
#
#
#     # output output_stereo (EncNet_ResNet101s_ADE):
#     # example of sidewalk: 1547131049128184, bike and human
#     # for a car: 1547131055565871
#     # bus and car 1547131057128391
#     # turning truck: 1547131069378701
#     # 4 cars: 1547131077566419
#     # car right in fron: 1547131105754642