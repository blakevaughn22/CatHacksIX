import cv2

def upscale_image(image):
    img = cv2.imread(image)
    resized = cv2.resize(img,dsize=None,fx=4,fy=4)
    filename = "./images/upscaled.jpg"
    cv2.imwrite(filename, resized)
    
def resize_image(image):
    img = cv2.imread(image)
    resized = cv2.resize(img, (256,144), interpolation= cv2.INTER_LINEAR)
    filename = "./images/resized.jpg"
    cv2.imwrite(filename, resized)
