1.Giới thiệu đề tài

1.1.Bài toán

Trong những năm gần đây, việc ứng dụng trí tuệ nhân tạo trong lĩnh vực khí tượng – thời tiết ngày càng được quan tâm. Bài toán đặt ra là làm thế nào để tự động nhận dạng và phân loại trạng thái thời tiết dựa trên hình ảnh bầu trời, thay vì chỉ dựa vào dữ liệu số liệu đo đạc truyền thống. Cụ thể, bài toán yêu cầu xây dựng một mô hình học máy có khả năng phân loại ảnh thời tiết thành các nhóm như nắng, mưa, nhiều mây, sương mù,… dựa trên đặc trưng hình ảnh đầu vào.

1.2.Mục tiêu

Mục tiêu của đề tài là xây dựng và huấn luyện mô hình Convolutional Neural Network (CNN) nhằm phân loại ảnh thời tiết với độ chính xác ở mức chấp nhận được. Thông qua đề tài, sinh viên làm quen với quy trình xử lý dữ liệu hình ảnh, thiết kế kiến trúc CNN, huấn luyện và đánh giá mô hình bằng các chỉ số như Accuracy, Precision, Recall và F1-score. Ngoài ra, đề tài còn hướng đến việc đánh giá khả năng ứng dụng thực tế của mô hình trong các hệ thống hỗ trợ dự báo và giám sát thời tiết.

2.Dataset

2.1.Nguồn data

Weather Dataset – Jehan Bhathena (Kaggle)

2.2.Link tải

https://www.kaggle.com/datasets/jehanbhathena/weather-dataset/data

3.Pipeline

3.1.Tiền xử lý

Trước khi đưa dữ liệu vào huấn luyện mô hình CNN, tập dữ liệu hình ảnh thời tiết cần được tiền xử lý nhằm đảm bảo tính đồng nhất và nâng cao hiệu quả học của mô hình.

Đầu tiên, ảnh được resize về cùng một kích thước cố định (ví dụ 224×224 pixel). Việc này giúp tất cả ảnh có cùng hình dạng đầu vào, phù hợp với kiến trúc CNN và giảm độ phức tạp trong quá trình huấn luyện.

Tiếp theo, giá trị pixel của ảnh được chuẩn hóa về khoảng [0,1] bằng cách chia cho 255. Quá trình chuẩn hóa giúp mô hình học nhanh hơn, ổn định hơn và tránh hiện tượng gradient quá lớn trong quá trình lan truyền ngược.

Sau đó, nhãn của ảnh được mã hóa dưới dạng số nguyên, mỗi số tương ứng với một lớp thời tiết. Dữ liệu được chia thành tập huấn luyện (training set) và tập kiểm tra (test set) nhằm đánh giá khả năng tổng quát hóa của mô hình trên dữ liệu chưa từng thấy.

Cuối cùng, dữ liệu được chuyển về đúng định dạng tensor để làm đầu vào cho mô hình CNN.

3.2.Train

Sau khi hoàn tất bước tiền xử lý dữ liệu, mô hình Convolutional Neural Network (CNN) được tiến hành huấn luyện trên tập dữ liệu huấn luyện. Trong quá trình huấn luyện, mô hình học cách trích xuất các đặc trưng quan trọng từ hình ảnh thời tiết thông qua các tầng tích chập và pooling, sau đó thực hiện phân loại bằng các tầng fully connected.

Mô hình được huấn luyện với hàm mất mát phù hợp cho bài toán phân loại đa lớp, kết hợp với thuật toán tối ưu Adam nhằm cập nhật trọng số một cách hiệu quả. Quá trình huấn luyện được thực hiện trong nhiều epoch để mô hình dần hội tụ và cải thiện độ chính xác.

Trong quá trình train, mô hình được đánh giá trên tập validation để theo dõi sự thay đổi của loss và accuracy, từ đó phát hiện sớm hiện tượng overfitting hoặc underfitting. Kết quả huấn luyện cho thấy mô hình có khả năng học tốt từ dữ liệu ảnh và đạt được hiệu suất phân loại ở mức chấp nhận được.

3.3.Evaluate

Sau khi hoàn tất quá trình huấn luyện, mô hình CNN được đánh giá trên tập dữ liệu kiểm tra (test set) nhằm kiểm tra khả năng tổng quát hóa đối với dữ liệu chưa từng xuất hiện trong quá trình huấn luyện. Việc đánh giá giúp xác định mức độ hiệu quả và độ tin cậy của mô hình khi áp dụng vào thực tế.

Mô hình được đánh giá thông qua các chỉ số phổ biến trong bài toán phân loại đa lớp, bao gồm Accuracy, Precision, Recall và F1-score. Trong đó, Accuracy phản ánh hiệu quả tổng thể, còn Precision, Recall và F1-score cho phép đánh giá chi tiết hơn trên từng lớp thời tiết.

Bên cạnh đó, Confusion Matrix được sử dụng để phân tích trực quan kết quả dự đoán, giúp xác định các lớp mà mô hình dự đoán tốt cũng như các lớp dễ bị nhầm lẫn. Từ kết quả đánh giá, có thể rút ra nhận xét về điểm mạnh, hạn chế của mô hình và đề xuất hướng cải thiện trong tương lai.

3.4.Inference

Sau khi mô hình CNN được huấn luyện và đánh giá, bước Inference được thực hiện nhằm dự đoán nhãn cho ảnh thời tiết mới chưa từng xuất hiện trong tập dữ liệu huấn luyện. Ở bước này, ảnh đầu vào được tiền xử lý theo đúng quy trình đã áp dụng trong giai đoạn train, bao gồm resize ảnh về kích thước cố định và chuẩn hóa giá trị pixel.

Ảnh sau khi tiền xử lý được đưa vào mô hình CNN đã huấn luyện để suy luận. Mô hình xuất ra xác suất thuộc về từng lớp thông qua tầng Softmax, từ đó lớp có xác suất cao nhất được chọn làm kết quả dự đoán. Kết quả inference cho thấy mô hình có khả năng nhận dạng trạng thái thời tiết từ ảnh bầu trời một cách tự động.

4.Mô hình sử dụng

Trong project của tôi sử dụng thuật toán Convolutional Neutral Netwwork(CNN)

Lý do chọn

Thứ nhất, CNN là mô hình học sâu phù hợp nhất cho bài toán xử lý và phân loại hình ảnh. Khác với các mô hình học máy truyền thống, CNN có khả năng tự động trích xuất đặc trưng từ ảnh thông qua các tầng tích chập, giúp mô hình học được các đặc điểm quan trọng như cạnh, màu sắc và kết cấu của bầu trời. Điều này đặc biệt phù hợp với bài toán phân loại thời tiết, nơi đặc trưng hình ảnh đóng vai trò quyết định.

Thứ hai, CNN giúp giảm độ phức tạp của mô hình và nâng cao hiệu quả học nhờ cơ chế chia sẻ trọng số và pooling. So với việc thiết kế đặc trưng thủ công, CNN cho phép xây dựng mô hình end-to-end, dễ mở rộng và cải thiện. Việc sử dụng CNN cũng giúp sinh viên hiểu rõ quy trình xây dựng, huấn luyện và đánh giá một mô hình học sâu trong thực tế.

5.Kết quả

Confusion Matrix

Confusion Matrix được sử dụng để đánh giá chi tiết kết quả phân loại của mô hình CNN trên tập dữ liệu kiểm tra. Ma trận này thể hiện số lượng mẫu được dự đoán đúng và sai giữa các lớp thời tiết, từ đó cho phép phân tích cụ thể hiệu suất của mô hình trên từng lớp.

Kết quả cho thấy mô hình dự đoán chính xác cao đối với các lớp có đặc trưng hình ảnh rõ ràng, thể hiện qua các giá trị lớn trên đường chéo chính của confusion matrix. Ngược lại, một số lớp có đặc điểm hình ảnh tương đồng vẫn xảy ra nhầm lẫn, thể hiện ở các ô ngoài đường chéo.

Thông qua confusion matrix, có thể nhận thấy rằng mô hình CNN đã học được các đặc trưng cơ bản của ảnh thời tiết, tuy nhiên vẫn còn hạn chế trong việc phân biệt các trường hợp có sự giao thoa về đặc trưng hình ảnh. Đây là cơ sở để đề xuất các hướng cải thiện mô hình trong tương lai.

6.Hướng dẫn chạy

Bước 1: Chuẩn bị môi trường

Sử dụng Google Colab

Chọn Runtime → Change runtime type → GPU để tăng tốc huấn luyện

Cài đặt các thư viện cần thiết: TensorFlow, NumPy, Matplotlib, Scikit-learn

Bước 2: Tải và chuẩn bị dataset

Tải dataset từ Kaggle (Weather Dataset)

Upload file .zip lên Colab

Giải nén dataset và kiểm tra cấu trúc thư mục
(mỗi thư mục con tương ứng với một lớp thời tiết)

Bước 3: Load và tiền xử lý dữ liệu

Đọc ảnh từ các thư mục

Resize ảnh về cùng kích thước (ví dụ 224×224)

Chuẩn hóa giá trị pixel về khoảng [0,1]

Gán nhãn cho từng ảnh

Chia dữ liệu thành tập train và test

Bước 4: Xây dựng mô hình CNN

Khai báo kiến trúc CNN gồm:

Convolution layers

Pooling layers

Fully connected layers

Thiết lập hàm mất mát và optimizer

Compile mô hình

Bước 5: Huấn luyện mô hình (Train)

Huấn luyện mô hình trên tập train

Theo dõi loss và accuracy qua các epoch

Lưu lại mô hình sau khi huấn luyện xong

Bước 6: Đánh giá mô hình (Evaluate)

Dự đoán trên tập test

Tính các chỉ số:

Accuracy

Precision

Recall

F1-score

Vẽ Confusion Matrix để phân tích chi tiết kết quả

Bước 7: Dự đoán ảnh mới (Inference)

Upload ảnh thời tiết mới (jpg, png, jfif,…)

Tiền xử lý ảnh giống hệt lúc train

Đưa ảnh vào mô hình để dự đoán

Hiển thị kết quả dự đoán

7.Cài môi trường

Cài đặt môi trường

Sử dụng Google Colab để triển khai mô hình, không cần cài đặt cục bộ

Kích hoạt GPU nhằm tăng tốc quá trình huấn luyện CNN

Cài đặt các thư viện cần thiết cho xử lý ảnh và học sâu

import os

import numpy as np

import pandas as pd

import cv2

from PIL import Image

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (

    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

Môi trường chạy sử dụng Python 3 và TensorFlow (Keras)

8.Chạy train

Sau khi hoàn tất tiền xử lý dữ liệu và xây dựng kiến trúc CNN, mô hình được tiến hành huấn luyện trên tập dữ liệu huấn luyện. Quá trình huấn luyện nhằm tối ưu các trọng số của mạng thông qua việc giảm hàm mất mát và nâng cao độ chính xác phân loại.

Mô hình được huấn luyện trong nhiều epoch với kích thước batch phù hợp. Trong quá trình train, kết quả được đánh giá trên tập validation để theo dõi sự thay đổi của loss và accuracy, từ đó kiểm soát hiện tượng overfitting.

# Compile model 

model.compile(

    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Chạy train

history = model.fit(

    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

Kết quả:

Epoch 1/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 299s 2s/step - accuracy: 0.3302 - loss: 4.3958 - val_accuracy: 0.1828 - val_loss: 3.5007

Epoch 2/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 305s 2s/step - accuracy: 0.3659 - loss: 1.9633 - val_accuracy: 0.3052 - val_loss: 2.6484

Epoch 3/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 328s 2s/step - accuracy: 0.3957 - loss: 1.7286 - val_accuracy: 0.4115 - val_loss: 1.7700

Epoch 4/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 283s 2s/step - accuracy: 0.4191 - loss: 1.7093 - val_accuracy: 0.5484 - val_loss: 1.3780

Epoch 5/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 314s 2s/step - accuracy: 0.4432 - loss: 1.5598 - val_accuracy: 0.5229 - val_loss: 1.4494

Epoch 6/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 328s 2s/step - accuracy: 0.4494 - loss: 1.5253 - val_accuracy: 0.4399 - val_loss: 2.2059

Epoch 7/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 315s 2s/step - accuracy: 0.4645 - loss: 1.4868 - val_accuracy: 0.3365 - val_loss: 3.4288

Epoch 8/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 286s 2s/step - accuracy: 0.5011 - loss: 1.3907 - val_accuracy: 0.5018 - val_loss: 2.2782

Epoch 9/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 274s 2s/step - accuracy: 0.4912 - loss: 1.3834 - val_accuracy: 0.4800 - val_loss: 2.0111

Epoch 10/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 274s 2s/step - accuracy: 0.5320 - loss: 1.2721 - val_accuracy: 0.5761 - val_loss: 1.3392

Epoch 11/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 282s 2s/step - accuracy: 0.5426 - loss: 1.2271 - val_accuracy: 0.6249 - val_loss: 1.2115

Epoch 12/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 326s 2s/step - accuracy: 0.5495 - loss: 1.1867 - val_accuracy: 0.5091 - val_loss: 1.5468

Epoch 13/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 309s 2s/step - accuracy: 0.5659 - loss: 1.1879 - val_accuracy: 0.6672 - val_loss: 1.0567

Epoch 14/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 330s 2s/step - accuracy: 0.5865 - loss: 1.0835 - val_accuracy: 0.5149 - val_loss: 1.7579

Epoch 15/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 280s 2s/step - accuracy: 0.6117 - loss: 1.0362 - val_accuracy: 0.6227 - val_loss: 1.3936

Epoch 16/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 281s 2s/step - accuracy: 0.6017 - loss: 1.0660 - val_accuracy: 0.5637 - val_loss: 1.4288

Epoch 17/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 281s 2s/step - accuracy: 0.6250 - loss: 0.9865 - val_accuracy: 0.7028 - val_loss: 1.0039

Epoch 18/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 273s 2s/step - accuracy: 0.6451 - loss: 0.9845 - val_accuracy: 0.6795 - val_loss: 1.1942

Epoch 19/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 329s 2s/step - accuracy: 0.6444 - loss: 0.9647 - val_accuracy: 0.7101 - val_loss: 0.9781

Epoch 20/20

172/172 ━━━━━━━━━━━━━━━━━━━━ 275s 2s/step - accuracy: 0.6454 - loss: 0.9167 - val_accuracy: 0.6446 - val_loss: 1.2617

9.Chạy demo

Sau khi mô hình CNN đã được huấn luyện và đánh giá, mô hình được sử dụng để dự đoán ảnh thời tiết đầu vào mới. Ảnh đầu vào phải được tiền xử lý giống hệt dữ liệu lúc train (resize, chuẩn hóa pixel) trước khi đưa vào mô hình.

Bước 1: Upload ảnh mới

from google.colab import files

uploaded = files.upload()

Bước 2: Tiền xử lí ảnh

img_path = "image.xxxx"

img = image.load_img(img_path, target_size=(128, 128))

img_array = image.img_to_array(img)

# Chuẩn hóa nếu model huấn luyện có normalize

img_array = img_array / 255.0

# Thêm batch dimension (bắt buộc cho predict)

img_array = np.expand_dims(img_array, axis=0)

Bước 3: Dự đoán

# Dự đoán

prediction = model.predict(img_array)

# Xử lý kết quả

if prediction.shape[1] == 1:  # binary classification

    predicted_class = int(prediction[0][0] > 0.5)
    
else:  # multi-class

    predicted_class = np.argmax(prediction, axis=1)[0]

print("Dự đoán thời tiết:", classes[predicted_class])

Kết quả:

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 128ms/step

Dự đoán thời tiết: xxxxxxxxx

10.Cấu trúc thư mục dự án

Weather-CNN/

│

├── dataset/

│   ├── train/

│   │   ├── cloudy/

│   │   ├── rainy/

│   │   ├── sunny/

│   │   └── snow/

│   │

│   └── test/

│       ├── cloudy/

│       ├── rainy/

│       ├── sunny/

│       └── snow/

│

├── notebooks/

│   ├── 1_load_data.ipynb

│   ├── 2_preprocessing.ipynb

│   ├── 3_train_model.ipynb

│   ├── 4_evaluate.ipynb

│   └── 5_demo_inference.ipynb

│

├── models/

│   └── weather_cnn.h5

│

├── results/

│   ├── confusion_matrix.png

│   └── accuracy_loss.png

│

├── README.md

└── requirements.txt

11.Tác giả

Họ và tên:Nguyễn Đông Thăng

Mã sinh viên:10123304

Mã lớp:124231
