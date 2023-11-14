# TableDataExtraction
Extract data from table and store it in csv

required libraries 
openCV          pip install opencv-python
matplotlib      pip install matplotlib
numpy           pip install numpy
pytesseract     pip install pytesseract
pandas          pip install pandas
regex           pip install regex




***thresh1,binary_image = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)***
cv2.THRESH_BINARY is a binary thresholding where pixel values above the threshold are set to the maximum value (255), and values below are set to 0. 
cv2.THRESH_OTSU is an adaptive thresholding method that automatically calculates the optimal threshold value based on the image histogram. The | operator is a bitwise OR operation, combining both types of thresholding.
one of the both will used according to the situation

***kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))***
for different images changes the tuple (10,10) accordingly which is matrix size
if you want extract from diffrent shape then change cv2.MORPH_RECT


