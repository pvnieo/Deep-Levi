import sys
import fnmatch
import os
from tqdm import tqdm

if __name__ == '__main__':

   data_dir = sys.argv[1]
   pattern = "*.tif"
   image_list = list()
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))

   print ('Working on images...')
   for image in tqdm(image_list):
      image_dir = os.path.dirname(image)
      resized_image = image_dir+'/resized/'+image.split('/')[-1].split('.')[0]+'_resized.png'
     
      # the 'true' image to be used in tensorflow (label)
      os.system('convert "' + image + '" -resize 224x224\! "' + resized_image +'"')

