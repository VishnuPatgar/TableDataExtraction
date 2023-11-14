import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import pandas as pd
import re



image_path = 'demoTable.png.png'
img = cv2.imread(image_path,0)
_, bitnot = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)



thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
img_bin = 255-img_bin
plotting = plt.imshow(img_bin,cmap='gray')
plt.title("Inverted Image with global thresh holding")
plt.show()

thresh1, img_bin1_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
plotting = plt.imshow(img_bin1_otsu, cmap='gray')
plt.title("Image with OTSU Threshold")
plt.show()


img_bin2 = 255-img
thresh1,binary_image = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plotting = plt.imshow(binary_image,cmap='gray')
plt.title("Inverted Image with otsu thresh holding")
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
plotting = plt.imshow(kernel,cmap='gray')
plt.title("Inverted Image with otsu thresh holding")
plt.show()



vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//150))
eroded_image = cv2.erode(binary_image, vertical_kernel, iterations=5)

vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=5)
plotting = plt.imshow(vertical_lines,cmap='gray')
plt.title("Inverted Image with otsu thresh holding")
plt.show()

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1] // 150, 1))
eroded_image = cv2.erode(binary_image, horizontal_kernel, iterations=5)
horizontal_lines = cv2.dilate(eroded_image, horizontal_kernel, iterations=5)
plotting = plt.imshow(horizontal_lines,cmap='gray')
plt.title("Inverted Image with otsu thresh holding")
plt.show()


vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
b_image = cv2.bitwise_not(cv2.bitwise_xor(img,vertical_horizontal_lines))
plotting = plt.imshow(b_image,cmap='gray')
plt.show()

contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


boundingBoxes = [cv2.boundingRect(c) for c in contours]
(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
key=lambda x:x[1][1]))



boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if (w<1000 and h<500):
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        boxes.append([x,y,w,h])
plotting = plt.imshow(image,cmap='gray')
plt.title("Identified contours")
plt.show()




rows=[]
columns=[]
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
mean = np.mean(heights)
print("mean",mean)
columns.append(boxes[0])
previous=boxes[0]
for i in range(1,len(boxes)):
    if(boxes[i][1]<=previous[1]+mean/2):
        columns.append(boxes[i])
        previous=boxes[i]
        if(i==len(boxes)-1):
            rows.append(columns)
    else:
        rows.append(columns)
        columns=[]
        previous = boxes[i]
        columns.append(boxes[i])
print("Rows")
for row in rows:
    1+1




total_cells=0
for i in range(len(row)):
    if (len(row[i])+1) > total_cells:
        total_cells = (len(row[i])+1)
print("total_cells")
print(total_cells)


center = [int(rows[i][j][0]+rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
# print("center",center)
center=np.array(center)
center.sort()
# print("center",center)





boxes_list = []

for i in range(len(rows)):
    l = []
    for k in range(total_cells):
        l.append([])

    for j in range(len(rows[i])):
        diff = abs(center - (rows[i][j][0] + rows[i][j][2] / 4))
        minimum = min(diff)
        indexing = list(diff).index(minimum)

        # Check if indexing is within the valid range
        if indexing < len(l):
            l[indexing].append(rows[i][j])
        else:
            print("Indexing out of range:", indexing)
    
    boxes_list.append(l)

for box in boxes_list:
    print("box length",len(box))
    print("box", box)









# boxes_list = []
# for i in range(len(rows)):
#     l=[]
#     for k in range(total_cells):
#         l.append([])
#     for j in range(len(rows[i])):
#         diff = abs(center-(rows[i][j][0]+rows[i][j][2]/4))
#         minimum = min(diff)
#         indexing = list(diff).index(minimum)
#         l[indexing].append(rows[i][j])
#     boxes_list.append(l)
# for box in boxes_list:
#     print("box",box)

# for subrow in rows:
#     for itm in subrow:
#         print("length of subrow",subrow)
#         x = itm[0]
#         y = itm[1]
#         k = itm[2]
#         h = itm[3]
#         print(x,y,k,h)
#         cropped_region = img[y:y+h, x:x+k]
#         extracted_text = pytesseract.image_to_string(cropped_region)
#         print(extracted_text)
#         plotting = plt.imshow(cropped_region, cmap='gray')
#         plt.title("Cropped Region with Extracted Text")
#         plt.show()
#     else:
#         print("Invalid subrow:", subrow)




# for box in boxes_list:
#     for itm in box:
#         for i in itm:
#             print("length of subrow",box)
#             x = i[0]
#             y = i[1]
#             k = i[2]
#             h = i[3]
#             print(x,y,k,h)
#             cropped_region = img[y:y+h, x:x+k]
#             extracted_text = pytesseract.image_to_string(cropped_region)
#             print(extracted_text)
#             plotting = plt.imshow(cropped_region, cmap='gray')
#             plt.title("Cropped Region with Extracted Text")
#             plt.show()




dataframe_final=[]
for i in range(len(boxes_list)):
    for j in range(len(boxes_list[i])):
        s=''
        if(len(boxes_list[i][j])==0):
            dataframe_final.append(' ')
        else:
            for k in range(len(boxes_list[i][j])):
                y,x,w,h = boxes_list[i][j][k][0],boxes_list[i][j][k][1], boxes_list[i][j][k][2],boxes_list[i][j][k][3]
                roi = bitnot[x:x+h, y:y+w]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                border = cv2.copyMakeBorder(roi,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                dilation = cv2.dilate(resizing, kernel,iterations=1)
                erosion = cv2.erode(dilation, kernel,iterations=2)
                out = pytesseract.image_to_string(erosion)
                if(len(out)==0):
                    out = pytesseract.image_to_string(erosion)
                s = s +" "+ out
            dataframe_final.append(s)


arr = np.array(dataframe_final)

dataframe = pd.DataFrame(arr.reshape(len(rows), total_cells))
data = dataframe.style.set_properties(align="left")

# below commented code used to change the column head and creating json object of extracted data with key as we want
# for col in dataframe.columns:
#     dataframe[col] = dataframe[col].str.replace('\n', '').str.strip()

# new_column_names = dataframe.iloc[0]
# dataframe.columns = new_column_names
# dataframe = dataframe.iloc[1:]

#print(data)
# print(dataframe)
# d=[]


# dataframe.set_index('Subject Code', inplace=True)
# json_obj = dataframe.to_dict(orient="index")
# print(json_obj)


#below commented code is used for pattern matching in json object

# def search_json_for_pattern(json_obj, pattern):
#     matches = []

#     def check_for_match(obj, path=""):
#         if isinstance(obj, dict):
#             for key, value in obj.items():
#                 new_path = f"{path}.{key}" if path else key
#                 if key == "Subject Name" and re.match(pattern, str(value)):
#                     match_details = obj.copy()
#                     match_details["Subject Code"] = path
#                     matches.append(match_details)
#                 if isinstance(value, (dict, list)):
#                     check_for_match(value, path=new_path)
#         elif isinstance(obj, list):
#             for i, item in enumerate(obj):
#                 new_path = f"{path}[{i}]"
#                 if isinstance(item, dict) or isinstance(item, list):
#                     check_for_match(item, path=new_path)

#     check_for_match(json_obj)
#     return matches

# Example usage
# pattern = "ResearchMethodology"  # Replace with your desired pattern
# matching_items = search_json_for_pattern(json_obj, pattern)
# for item in matching_items:
#     print(item)



for i in range(0,len(rows)-1):
    for j in range(0,(total_cells-1)):
        print(dataframe.iloc[i][j],end=" \n")
print()


dataframe.to_csv("extracted.csv")


