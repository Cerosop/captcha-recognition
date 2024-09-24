import shutil
import numpy as np
import cv2
import os
from paddleocr import PaddleOCR
import easyocr


# is_resize = False
# is_opened = True # 太瘦或字毀損不能用 
# is_closed = False # 字的空隙太小不能用 ex e的裡面太小
# is_reverse = False # 字白色才需要
# big_or_low = 1 # 0數字 1小寫 2大寫 3全部
# length = 4

# answer_map = {}

# folder_path = 'captcha'
# image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# # Paddle OCR
# paddleocr = PaddleOCR()

# # Easy OCR
# reader = easyocr.Reader(
#     ['en'], 
#     gpu=True, 
#     # recog_network='english_g2', # 複雜英文文本
#     # detector=False
# )

# def predict(image):
#     # Paddle OCR   1 3 6   2 5 9 10 11
#     prediction_result = paddleocr.ocr(image, cls=True)[0]
#     if prediction_result:
#         prediction_result = list(prediction_result[0][1])
#         prediction_result[0] = prediction_result[0].replace(' ', '')
#         prediction_result[0] = prediction_result[0].lower()
#         if len(prediction_result[0]) == length:
#             if prediction_result[0] in answer_map:
#                 answer_map[prediction_result[0]] += prediction_result[1]
#             else:
#                 answer_map[prediction_result[0]] = prediction_result[1]
#     # print()
#     # print("paddle:", prediction_result)
    
#     # easy ocr 3 6 7 11 12   1 9 10 13
#     if big_or_low == 0:
#         prediction_result = reader.readtext(image, 
#                                             decoder='greedy', # 一個字一個字檢測
#                                             beamWidth=10, # 解码过程中保留的候选序列的数量
#                                             slope_ths=0.2, # 文本頃斜斜率阈值
#                                             low_text=0.2, # 低對比度區域的文本檢測閥值
#                                             allowlist='0123456789'
#                                             )
#     elif big_or_low == 1:
#         prediction_result = reader.readtext(image, 
#                                             decoder='greedy', # 一個字一個字檢測
#                                             beamWidth=10, # 解码过程中保留的候选序列的数量
#                                             slope_ths=0.2, # 文本頃斜斜率阈值
#                                             low_text=0.2, # 低對比度區域的文本檢測閥值
#                                             allowlist='0123456789abcdefghijklmnopqrstuvwxyz'
#                                             )
#     elif big_or_low == 2:
#         prediction_result = reader.readtext(image, 
#                                             decoder='greedy', # 一個字一個字檢測
#                                             beamWidth=10, # 解码过程中保留的候选序列的数量
#                                             slope_ths=0.2, # 文本頃斜斜率阈值
#                                             low_text=0.2, # 低對比度區域的文本檢測閥值
#                                             allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#                                             )
#     elif big_or_low == 3:
#         prediction_result = reader.readtext(image, 
#                                             decoder='greedy', # 一個字一個字檢測
#                                             beamWidth=10, # 解码过程中保留的候选序列的数量
#                                             slope_ths=0.2, # 文本頃斜斜率阈值
#                                             low_text=0.2, # 低對比度區域的文本檢測閥值
#                                             allowlist='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#                                             )
#     if prediction_result:
#         prediction_result = list(prediction_result[0][1:3])
#         prediction_result[0] = prediction_result[0].replace(' ', '')
#         prediction_result[0] = prediction_result[0].lower()
#         if len(prediction_result[0]) == length:
#             if prediction_result[0] in answer_map:
#                 answer_map[prediction_result[0]] += prediction_result[1]
#             else:
#                 answer_map[prediction_result[0]] = prediction_result[1]
#     # print("easy: ", prediction_result)
    
#     return prediction_result


# for image_file in image_files:
def code_recognizer(image_file, is_resize=False, is_opened = True, is_closed = False, is_reverse = False, big_or_low = 1, length = 4):
    filename = image_file.split('/')[-1].split('.')[0]

    answer_map = {}
    
    # Paddle OCR
    paddleocr = PaddleOCR()

    # Easy OCR
    reader = easyocr.Reader(
        ['en'], 
        gpu=True, 
        # recog_network='english_g2', # 複雜英文文本
        # detector=False
    )
    
    def predict(image):
        # Paddle OCR   1 3 6   2 5 9 10 11
        prediction_result = paddleocr.ocr(image, cls=True)[0]
        if prediction_result:
            prediction_result = list(prediction_result[0][1])
            prediction_result[0] = prediction_result[0].replace(' ', '')
            prediction_result[0] = prediction_result[0].lower()
            if len(prediction_result[0]) == length:
                if prediction_result[0] in answer_map:
                    answer_map[prediction_result[0]] += prediction_result[1]
                else:
                    answer_map[prediction_result[0]] = prediction_result[1]
        # print()
        # print("paddle:", prediction_result)
        
        # easy ocr 3 6 7 11 12   1 9 10 13
        if big_or_low == 0:
            prediction_result = reader.readtext(image, 
                                                decoder='greedy', # 一個字一個字檢測
                                                beamWidth=10, # 解码过程中保留的候选序列的数量
                                                slope_ths=0.2, # 文本頃斜斜率阈值
                                                low_text=0.2, # 低對比度區域的文本檢測閥值
                                                allowlist='0123456789'
                                                )
        elif big_or_low == 1:
            prediction_result = reader.readtext(image, 
                                                decoder='greedy', # 一個字一個字檢測
                                                beamWidth=10, # 解码过程中保留的候选序列的数量
                                                slope_ths=0.2, # 文本頃斜斜率阈值
                                                low_text=0.2, # 低對比度區域的文本檢測閥值
                                                allowlist='0123456789abcdefghijklmnopqrstuvwxyz'
                                                )
        elif big_or_low == 2:
            prediction_result = reader.readtext(image, 
                                                decoder='greedy', # 一個字一個字檢測
                                                beamWidth=10, # 解码过程中保留的候选序列的数量
                                                slope_ths=0.2, # 文本頃斜斜率阈值
                                                low_text=0.2, # 低對比度區域的文本檢測閥值
                                                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                                )
        elif big_or_low == 3:
            prediction_result = reader.readtext(image, 
                                                decoder='greedy', # 一個字一個字檢測
                                                beamWidth=10, # 解码过程中保留的候选序列的数量
                                                slope_ths=0.2, # 文本頃斜斜率阈值
                                                low_text=0.2, # 低對比度區域的文本檢測閥值
                                                allowlist='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                                                )
        if prediction_result:
            prediction_result = list(prediction_result[0][1:3])
            prediction_result[0] = prediction_result[0].replace(' ', '')
            prediction_result[0] = prediction_result[0].lower()
            if len(prediction_result[0]) == length:
                if prediction_result[0] in answer_map:
                    answer_map[prediction_result[0]] += prediction_result[1]
                else:
                    answer_map[prediction_result[0]] = prediction_result[1]
        # print("easy: ", prediction_result)
        
        return prediction_result

    # print(filename)
    # length = len(filename)
    
    # nyn = ['4D7YS', 'e5hb', '4682', 'XDHYN', 'xmqki']
    # nyy = ['HRAI']
    # nny = ['FH2DE']
    # yny = ['jw62k']
    # if filename in nyn:
    #     is_opened = False  
    #     is_closed = True 
    #     is_reverse = False
    # elif filename in nyy:
    #     is_opened = False 
    #     is_closed = True 
    #     is_reverse = True
    # elif filename in nny:
    #     is_opened = False 
    #     is_closed = False 
    #     is_reverse = True
    # elif filename in yny:
    #     is_opened = True 
    #     is_closed = False 
    #     is_reverse = True
    # else:
    #     is_opened = True 
    #     is_closed = False 
    #     is_reverse = False
        
    # if filename.isnumeric():
    #     big_or_low = 0
    # elif filename.lower() == filename:
    #     big_or_low = 1
    # elif filename.upper() == filename:
    #     big_or_low = 2
    # else:
    #     big_or_low = 3
        
    if os.path.exists(os.path.join("output", filename)):
        shutil.rmtree(os.path.join("output", filename))
    os.makedirs(os.path.join("output", filename))
    
    # image_path = os.path.join(folder_path, image_file)
    image_path = image_file
    image = cv2.imread(image_path)
    shape = 1
    if isinstance(image[0][0], list):
        shape = 3
    paddleocr = PaddleOCR(
        use_angle_cls=True, # 旋轉文本
        lang='en', 
        rec_image_shape=[shape, len(image), len(image[0])], # 圖片大小(黑白的由1開頭)
        max_text_length=len(filename), 
        det_db_unclip_ratio=1.5, # 邊框放大
        filter_ths=0.3 # 偵測閥值
    )
    cv2.imwrite(os.path.join(os.path.join("output", filename), 'image.png'), image)
    predict_result = predict(image)
    # print("image:", predict_result)
    
    # 反轉顏色
    if is_reverse:
        reverse = cv2.bitwise_not(image)
        cv2.imwrite(os.path.join(os.path.join("output", filename), 'reverse1.png'), reverse)
        image = reverse
        predict_result = predict(reverse)
        # print("reverse1:", predict_result)
    
    # 調整對比度 alpha為對比 beta為亮度
    adjust = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
    cv2.imwrite(os.path.join(os.path.join("output", filename), 'adjust.png'), adjust)
    predict_result = predict(adjust)
    # print("adjust:", predict_result)
    
    # 邊緣保留濾波  去噪
    dst = cv2.pyrMeanShiftFiltering(adjust, sp=1, sr=55)
    cv2.imwrite(os.path.join(os.path.join("output", filename), 'dst.png'), dst)
    predict_result = predict(dst)
    # print("dst:", predict_result)
        
    # 轉為灰度
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(os.path.join("output", filename), 'gray.png'), gray)
    predict_result = predict(gray)
    # print("gray:", predict_result)

    # 二值化
    _, binary = cv2.threshold(gray, np.mean(gray) - (np.max(gray) - np.min(gray)) * 0.2, 255, cv2.THRESH_BINARY_INV)
    paddleocr = PaddleOCR(
        use_angle_cls=True, # 旋轉文本
        lang='en', 
        rec_image_shape=[1, len(image), len(image[0])], # 圖片大小(黑白的由1開頭)
        max_text_length=len(filename), 
        det_db_unclip_ratio=1.5, # 邊框放大
        filter_ths=0.3 # 偵測閥值
    )
    cv2.imwrite(os.path.join(os.path.join("output", filename), 'binary.png'), binary)
    predict_result = predict(binary)
    # print("binary:", predict_result)
    
    # 定義结構元素
    kernel = np.ones((3, 3), np.uint8)
    
    # 開運算(先腐蝕後膨脹)
    if is_opened:
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cv2.imwrite(os.path.join(os.path.join("output", filename), 'opened.png'), opened)
        predict_result = predict(opened)
        # print("opened:", predict_result)
        binary = opened
    
    # 閉運算(先膨脹後腐蝕)
    if is_closed:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        cv2.imwrite(os.path.join(os.path.join("output", filename), 'closed.png'), closed)
        predict_result = predict(closed)
        # print("closed:", predict_result)
        binary = closed
    
    # 改變圖片大小
    if is_resize:
        origin_h, origin_w = binary.shape
        target_height = 100
        new_image = cv2.resize(binary, target_height, int((target_height / origin_h) * origin_w))
        cv2.imwrite('new_image', new_image)
        binary = new_image
        
    # 反轉顏色
    reverse = cv2.bitwise_not(binary)
    cv2.imwrite(os.path.join(os.path.join("output", filename), 'reverse2.png'), reverse)
    
    captcha_text = predict(reverse)
    # print(f"辨識的驗證碼是: {captcha_text}")
    
    answer = 'no found'
    if answer_map:
        answer = max(answer_map, key=answer_map.get)
    print()
    print(f"辨識{filename}的驗證碼是: {answer}")
    answer_map = {}
    
    return answer
    
    # print()    


# # pytesseract
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = 'D:/Tesseract-OCR/tesseract.exe'
# prediction_result = pytesseract.image_to_string(image)


# # Manga OCR
# from PIL import Image
# import manga_ocr
# mangaocr = manga_ocr.MangaOcr()
# # MangaOCR  3 12     1 7 9 13
# image = Image.open(filename)
# prediction_result = mangaocr(image)


# # Transformer-based OCR 9 8 7 2 10    1 13
# from transformers import VisionEncoderDecoderModel
# import torch
# try:
#     pixel_values = processor(images=image, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values)
#     prediction_result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# except:
#     prediction_result = ''


# TrOCRProcessor
# from transformers import TrOCRProcessor
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


# if is_seperate:
    # projection = np.sum(reverse, axis=0)
    # # 分割点：投影值>=全體8成亮度的地方
    # split_points = np.where(projection >= np.min(projection) + (np.max(projection) - np.min(projection)) * 0.8)[0]
    
    # print("辨識的驗證碼是: ")
    # for i in range(1, len(split_points)):
    #     if split_points[i] - split_points[i-1] > 10:
    #         end = split_points[i]
    #         start = split_points[i - 1]
    #         char = reverse[:, max(start-5, 0):min(end + 5, binary.shape[1])]
    #         captcha_text = predict(char)
    #         if captcha_text:
    #             print(captcha_text, end='')
    # print()