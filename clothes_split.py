import cv2 as cv
import os


img_path = "./clothes/"
tex_result_path = "./texture/"

window_width = 48
window_height = 48

window_number = 1

def img_read(path, img_name):
	img_fullpath = os.path.join(path, img_name)
	img = cv.imread(img_fullpath)
	return img

def sliding_window(img, x_step_size,y_step_size, window_size):
	for y in range(0, img.shape[0], y_step_size):
		for x in range(0, img.shape[1], x_step_size):
			yield (x, y, img[y:y+window_size[1], x:x+window_size[0]])

def img_save(filename, img):
	cv.imwrite(filename, img)

#for i in os.listdir(img_path):
clothes_filename = "FIND Women's Contrast Stripe T-Shirt, Blue, 16 (Manufacturer size X-Large).jpg"
tex_dir = "stripe_3/"
clothes_img = img_read(img_path, clothes_filename)
os.makedirs(tex_result_path + tex_dir)
for (x, y, window) in sliding_window(clothes_img, 3, 3, (window_width, window_height)):
	if window.shape[0] != window_width or window.shape[1] != window_height:
		continue
	clone_img = clothes_img.copy()
	cv.rectangle(clone_img, (x, y), (x+window_width, y+window_height), (0, 255, 0),2)
	sliding_window_img = clothes_img[y:y+window_width, x:x+window_height]
	#result_img_name = str(x) + "," + str(y) +  "," +str(x+window_width) + ","+ str(y+window_height)+'.jpg'
	result_img_name = "stripe_3" + str(window_number)+".jpg"
	result_img_fullpath = os.path.join(tex_result_path + tex_dir, result_img_name)
	img_save(result_img_fullpath, sliding_window_img)
	window_number += 1
window_number = 0