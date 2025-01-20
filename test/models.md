# **Machine Learning Model Documentation**

## **Key Terminologies**
To better understand the content, here are some key terminologies used in the documentation:

- **YOLOv11 (You Only Look Once version 11)**: A state-of-the-art, real-time object detection algorithm that can detect objects in images or videos. YOLOv11 is known for its speed and accuracy, making it a popular choice for object detection tasks in real-time applications.

- **mAP (mean Average Precision)**: A performance metric for object detection models. It calculates the average of precision values at different recall thresholds, providing a summary of the model's accuracy across all classes. A higher mAP indicates better overall performance.

- **Precision**: A metric that measures the percentage of correctly predicted positive instances (true positives) out of all predicted positive instances (true positives + false positives). Precision is important when the cost of false positives is high.

- **Recall**: A metric that measures the percentage of relevant instances that were successfully identified by the model (true positives out of true positives + false negatives). High recall is essential when it's critical to detect as many true positive instances as possible, even at the risk of some false positives.

- **Auto Orientation**: A preprocessing technique that automatically adjusts the orientation of images to ensure they are aligned correctly. This is important when working with real-world datasets where images may have varying orientations.

- **Resizing**: The process of changing the size of images to a standard resolution (in this case, 640x640 pixels). Resizing ensures that all images fed into the model have the same dimensions, making the training process more efficient.

- **Adaptive Equalization**: A technique used to enhance the contrast of images by adjusting pixel intensities based on the local distribution of pixel values. This helps improve the visibility of features, especially in images with varying lighting conditions.

- **Data Augmentation**: A technique used to artificially increase the size of the training dataset by applying random transformations to the images, such as changing brightness or applying blurring. This helps the model generalize better and avoid overfitting.

- **Batch Size**: The number of training samples that are processed together in one pass through the model before updating the model's weights. A larger batch size can speed up training but requires more memory.

- **Learning Rate**: A hyperparameter that controls how much the model's weights are adjusted with each iteration during training. A high learning rate may cause the model to converge too quickly, while a low learning rate may slow down the training process.

- **Epoch**: One complete pass through the entire training dataset during training. Training a model for multiple epochs helps it learn more effectively by continuously adjusting its weights based on the data.

- **Overfitting**: A scenario where a model learns the details and noise in the training data to the point that it negatively impacts the model's performance on new data. Regularization techniques, such as data augmentation or dropout, help mitigate overfitting.

- **Underfitting**: A scenario where the model is too simple and cannot learn the underlying patterns of the data effectively. This typically happens when the model is not complex enough to capture the data’s intricacies.

---

## **1. Dataset and Data Processing**

### **Source of Data**
The dataset was sourced from a variety of Universe projects that encompass a wide range of image types, contexts, annotation sizes, counts, and camera specifications. All datasets used in this project are licensed under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). The datasets included are:

- [First Pedestrian Test](https://universe.roboflow.com/safer-strides/first-pedestrian-test)  
- [person_camera_security1](https://universe.roboflow.com/chinh/person_camera_security1)  
- [Human Action Recognition 2000](https://universe.roboflow.com/skripsi-u18dy/human-action-recognition-2000)  
- [People Detection](https://universe.roboflow.com/chris-kydks/people-detection-2csbw)  
- [contador-de-gente teste 3](https://universe.roboflow.com/mackleaps/contador-de-gente-teste-3)  
- [OD3](https://universe.roboflow.com/object-detection-tuphv/od3-fq4yp)  
- [Person Detection](https://universe.roboflow.com/illimited/person-detection-gbuka)  
- [MOT17-03-DPM](https://universe.roboflow.com/bhu-ykklm/mot17-03-dpm-udorc)  
- [The Curve](https://universe.roboflow.com/people-8gcmt/the-curve-02)  
- [Pedestrian Safety](https://universe.roboflow.com/intel-9horw/pedestrian-safety-obyfo)  
- [People Detection](https://universe.roboflow.com/jmedel/people-detection-f0fgt)  
- [People, Rabish](https://universe.roboflow.com/cpk-wow-k5nlf/people-rabish)  
- [Pascal VOC 2012](https://universe.roboflow.com/jacob-solawetz/pascal-voc-2012)  
- [Person Detection (General)](https://universe.roboflow.com/mohamed-traore-2ekkp/people-detection-general)  
- [License Plate Recognition Project](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)  

### **Dataset Overview**
The dataset includes images and annotations related to the following domains:
- Human detection
- Crowd detection
- Vehicle detection (cars)
- License plate recognition
- Null images (non-relevant data)

This diverse dataset was selected to ensure comprehensive coverage of real-world scenarios and to support multiple object detection tasks, including human detection and license plate recognition.

### **Data Preprocessing**
The following preprocessing techniques were applied to standardize and optimize the dataset for training:
- **Auto Orientation:** Corrected the orientation of images where necessary to ensure consistency.
- **Resizing:** Images were resized to a uniform 640x640 resolution to maintain consistency across the dataset.
- **Contrast Adjustment:** Adaptive equalization was applied to adjust the contrast and enhance image features, improving the model's ability to extract meaningful information.

### **Data Augmentation**
To increase model robustness and generalization, the following augmentation techniques were applied:
- **Outputs per Training Example:** Three augmented versions were created for each training image to enhance diversity.
- **Brightness Adjustment:** A random brightness adjustment between -15% and +15% was applied to simulate different lighting conditions.
- **Gaussian Blur:** A blur of up to 2.5 pixels was applied to the images to simulate various environmental factors and improve robustness to blurriness.

### **Data Splitting**
The dataset was split into training, validation, and test sets for each model as follows:

- **Model 1: Human Detection**
  - Training Set: 87% (36,387 images)
  - Validation Set: 8% (3,506 images)
  - Test Set: 4% (1,729 images)

- **Model 2: License Plate Recognition**
  - Training Set: 87% (20,462 images)
  - Validation Set: 9% (2,006 images)
  - Test Set: 4% (1,006 images)

---

## **2. Model Selection**

### **Model(s) Used**
Both tasks utilized the **YOLOv11** (You Only Look Once version 11) architecture for object detection. YOLOv11 was selected for its proven performance in various computer vision tasks, including real-time object detection.

### **Reason for Selection**
YOLOv11 was chosen due to the following reasons:
- **Versatility:** YOLOv11 is highly flexible and can handle a wide variety of object detection tasks. It is suitable for both human detection and license plate recognition, capable of identifying multiple object classes in a single inference.
- **Efficiency:** YOLOv11 is known for its real-time detection capabilities, making it ideal for applications that require high-speed predictions, such as surveillance or security systems.
- **Accuracy:** Despite being optimized for speed, YOLOv11 maintains high accuracy, even in complex and cluttered environments, making it suitable for human detection in crowded scenes and license plate recognition under varying conditions.
- **Pre-trained Models:** YOLOv11 offers pre-trained models, reducing the need for extensive training and enabling rapid deployment with excellent performance.
- **Proven Success:** YOLO has been widely adopted in the machine learning community and has consistently shown strong performance in a variety of real-world applications.

### **Hyperparameter Tuning**
Key hyperparameters, such as **batch size**, **learning rate**, and **number of epochs**, were tuned to optimize performance. These hyperparameters were fine-tuned to ensure convergence and prevent both underfitting and overfitting during training.

### **Evaluation Metrics**
The models were evaluated using the following metrics:

#### **Model 1: Human Detection**
- **mAP (mean Average Precision):** 85.1%
- **Precision:** 87.8%
- **Recall:** 77.2%

#### **Model 2: License Plate Recognition**
- **mAP (mean Average Precision):** 97.7%
- **Precision:** 97.7%
- **Recall:** 96.1%

The performance indicates that **Model 2** (license plate recognition) performs exceptionally well, with high precision and recall. **Model 1** (human detection) performs well overall but could benefit from improvements in recall, as it misses some relevant detections in challenging environments.

---

## **3. Training and Testing**

### **Training Process**
The models were trained using the YOLOv11 architecture, leveraging high-performance GPUs for efficient processing. The following steps were taken:
- **Batch Size and Learning Rate:** Optimal batch sizes and learning rates were selected to improve convergence and maintain efficiency during training.
- **Epochs:** Models were trained for a sufficient number of epochs to ensure they were able to generalize well and avoid overfitting.

### **Testing Process**
After training, the models were evaluated on the unseen test set to assess generalization and performance. The metrics used to evaluate the models included mAP, precision, and recall, as reported in the previous section.

---

## **4. Results and Analysis**

### **Results Summary**
- **Model 1 (Human Detection)**:
  - mAP: 85.1%
  - Precision: 87.8%
  - Recall: 77.2%
  - The model showed solid performance in detecting humans, but its recall indicates that some human instances were missed, particularly in crowded or complex scenarios.

- **Model 2 (License Plate Recognition)**:
  - mAP: 97.7%
  - Precision: 97.7%
  - Recall: 96.1%
  - The model performed exceptionally well, with high accuracy, precision, and recall. This suggests that license plate recognition is a less variable task, making it easier to achieve high performance.

### **Error Analysis**
- **Model 1 (Human Detection):** The relatively lower recall suggests that the model sometimes fails to detect humans in crowded or obstructed environments. Additional data augmentation or more diverse training samples could improve recall.
- **Model 2 (License Plate Recognition):** The model performed nearly flawlessly, with only a few minor errors, likely due to extreme conditions such as poor image quality or heavily occluded license plates.

### **Areas for Improvement**
- **Model 1 (Human Detection):** To improve recall, it may be beneficial to enhance the model's ability to detect humans in more challenging scenarios, such as occlusions, varying poses, or smaller object sizes. Incorporating additional datasets and data augmentation techniques can help.
- **Model 2 (License Plate Recognition):** While already achieving impressive results, further enhancement could involve optimizing for more diverse license plate designs or improving the model's ability to handle extreme lighting conditions and weather effects.

---

This concludes the documentation for your project. The sections covered include **Dataset and Data Processing**, **Model Selection**, **Training and Testing**, and **Results and Analysis**. You may continue by expanding further into additional topics such as **Deployment**, **Future Work**, or **References** if necessary.

Let me know if you'd like any further adjustments or additional sections!
