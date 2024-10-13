from flask import Flask, request, jsonify
import cv2
import numpy as np
import imutils
from imutils import contours

app = Flask(__name__)

@app.route('/process_omr', methods=['POST'])
def process_omr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    print('IAMGE CAPTURE')
    # Read the image from the request
    file = request.files['image']
    npimg = np.fromfile(file, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert image to grayscale and apply Gaussian blur and edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    # cv2.imshow("edged", edged)
    # cv2.waitKey(0)

    # Find the contours in the edged image
    cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    # Use the gray image for further processing
    warped = gray
    # cv2.imshow("warped", warped)
    # cv2.waitKey(0)

    # Apply threshold to get the binary image
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find external contours on the threshold image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Filter contours that likely correspond to bubbles
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.8 and ar <= 1.1:
            questionCnts.append(c)

    total_questions = len(questionCnts) // 13
    print(total_questions)
    bubble_threshold = 500
    results = []

    # Process each question group (13 bubbles per question)
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 13)):
        cnts = contours.sort_contours(questionCnts[i:i + 13], method="left-to-right")[0]
        bubbled = None
        selected_circles = []

        # cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
        # cv2.imshow("draw", image)
        # cv2.waitKey(0)

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            # print(total)

            if total > bubble_threshold:
                selected_circles.append(j)

        if len(selected_circles) > 1:
            bubbled = selected_circles[0]  # If multiple bubbles, choose the first
        elif len(selected_circles) == 0:
            bubbled = 0  # If no bubbles are filled, default to the first option
        else:
            bubbled = selected_circles[0]  # If one bubble is filled, select it

        if bubbled is not None:
            question_number = total_questions - q
            answer_index = bubbled
            if question_number > 0:
                results.append({
                    "question": question_number,
                    "answer": answer_index
                })

    # print(results)
    return jsonify(results)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)



