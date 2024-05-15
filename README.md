# saikrishnauggi-Thyroid-Detection-using-Machine-Learning
 Detect thyroid disorders early using machine learning. Predict disease presence &amp; severity with decision trees &amp; Naïve Bayes algorithms. Improve patient outcomes through timely intervention.

Sure, I'll help you format this document for the README.md file in your GitHub repository without simplifying it. Here's the formatted version:

---

# DETECTION OF THYROID DISORDERS USING MACHINE LEARNING APPROACH

## Abstract:
Classification-based machine learning plays a major role in various medical services. In the medical field, the salient and demanding task is to diagnose patients' health conditions and provide proper care and treatment of the disease at the initial stage. Let us consider Thyroid disease as the example. The normal and traditional methods of thyroid diagnosis involve a thorough inspection and various blood tests. The main goal is to recognize the disease at the early stages with a very high correctness. Machine learning techniques play a major role in the medical field for making a correct decision, proper disease diagnosis, and also save cost and time for the patient. The purpose of this study is prediction of thyroid disease using classification Predictive Modelling followed by binary classification using Decision Tree ID3 and Naive Bayes Algorithms. The Thyroid Patient dataset with proper attributes are fetched and using the Decision Tree algorithm the presence of thyroid in the patient is tested. Further, if thyroid is present then Naïve Bayes algorithm is applied to check for the thyroid stage in the patient.

## Introduction:
Thyroid disease diagnosis is not a simple task. It involves many procedures. The normal traditional way includes a proper medical examination and many blood samples for blood tests. Therefore, there is a necessity for a model which detects the thyroid disease at a very early stage of development.

In the medical field, machine learning plays an important role for thyroid disease diagnosis as it has various classification models based on which we can train our model with proper train dataset of the thyroid patient and can predict and give the results in an accurate manner with a higher degree of correctness. Some recent studies from Mumbai have suggested that congenital hypothyroidism is common in India. The disease occurs in 1 part of 2640 newborn children, when compared to the worldwide average range of 1 in 3800 considered. Congenital hypothyroidism can lead to serious complications if not detected in early stages. Therefore, the proposed model serves the goal in early detection of thyroid disease.

Based on the obtained test values the healthcare staff can easily examine the condition of the patient and also skip further clinical examinations if not necessary. Hence, this approach proves to be very much beneficial to the healthcare field. A proper train dataset results in an accurate predicting model, therefore reducing the overall cost of the thyroid patient treatment and also saving time. Classification algorithms are most suitable in decision making and also solving the real-world problems.

### About Thyroid:
The Thyroid is a butterfly-shaped endocrine gland which is situated at the base of the human neck. The vital role of the thyroid gland is maintaining and balancing human metabolism and also the growth and development of the human body. The vital tasks performed by the thyroid gland are blood circulation, body temperature control, muscle strength, and brain functioning. Any damage or improper functioning of the gland may seriously affect normal human body functioning. Therefore, proper thyroid hormone secretion results in a healthy human body. If there is either low or high secretion of the hormone it will adversely affect human health.

#### Various Thyroid Harmones and their effects:
The Thyroid gland mainly produces tri-iodothyronine (T3), thyroxine (T4), and Thyroid stimulating hormone (TSH). The Thyroid stimulating hormone (TSH) is released by the pituitary gland which mainly stimulates the thyroid gland to produce T3 and T4 which further stimulate the metabolism of almost every tissue present in the human body. Therefore, the pituitary gland plays a vital role in controlling the production of the required amount of thyroid hormones. If the TSH production level is less then T3, T4 secretion will be more and vice versa. The Thyroid disorder is the most common endocrine disease across the world. In a survey carried out in India, around 42 million people are suffering from this disease. Thyroid disease is different from other types of endocrine diseases in terms of the mode of treatment relative attainability and the ease of predicting the disease. The high thyroid hormone secretion leads to Hyperthyroidism and low secretion leads to Hypothyroidism. Both conditions adversely affect human physiology and the symptoms shown for hyperthyroidism are dry skin, hair thinning, loss of weight, high blood pressure, neck enlargement, and short menstrual periods. The symptoms shown for hypothyroidism include thyroid gland inflammation, weight gain, low blood pressure, heavy menstrual periods, and loss of appetite. These symptoms may get even worse if they are not treated at an early stage. Hence, there is a need for a proper prediction model which helps in diagnosing the patient’s condition in an early stage of the disease.

## Literature Survey:

### Bibi Amina Begum et al.:
Have proposed different Thyroid prediction techniques using data mining approaches. They have considered different dataset attributes for prediction and have explained the classification techniques in data mining like Decision Tree, Backpropagation Neural Network, SVM, and density-based clustering. They have analyzed the correlation of T3, T4, and TSH with hyperthyroidism and hypothyroidism.

### Ankita Tyagi et al.:
Have studied various classification-based machine learning algorithms. They have considered train datasets from the UCI Machine Learning repository and compared and analyzed the performance metric of the decision tree, support vector machine, and K-nearest neighbor.

### Aswathi A K et al.:
Have proposed a training model consisting of 21 thyroid-causing attributes. They have proposed partial swarm optimization to optimize the support vector machine parameters.

### M. Deepika et al.:
Have performed a general empirical study on various disease diagnosis like Diabetes, Breast Cancer, Heart disease, Thyroid prediction and have compared the accuracy rate by applying SVM, Decision tree and Artificial Neural Networks.

### Sumathi A et al.:
Have considered Thyroid data preprocessing mainly by applying the decision tree algorithm. They have first calculated the mean values of T3, T4, and TSH and considered as the preprocessing stage. Later on, they have applied machine learning-based feature selection and feature construction. Further, they have applied classification-based J48 algorithm which is a continuation of the ID3 algorithm and calculated the results.

### I Md. Dendi Maysanjaya et al.:
Have analyzed a comparison of various classification methods used to diagnose thyroid disease. They have compared by using Artificial Neural Networks, Radial Based Function, Learning Vector Quantization, Back Propagation Algorithm and Artificial Immune recognition system and concluded the comparison results. Among them, they found out that Multilayer Perceptron has the

 highest accuracy of 96.74%.

### Ammulu K et al.:
Have proposed a Thyroid Prediction System based on data mining classification algorithm. They have used a random forest approach to predict the results using Weka open-source tool used for data mining. Using this tool they have applied random forest algorithm with 25 thyroid data attributes and predicted the results accordingly.

### Roshan Banu D et al.:
Have conducted a study on different data mining techniques to detect thyroid disease. They have done study on Linear Discriminant analysis, Kfold cross-validation, and Decision tree. They have analyzed various splitting rules for the attributes of Decision tree. They have also compared the obtained values.

### Dr.B.Srinivasan et al.:
Have conducted a study on the diagnosis of thyroid disease using different data mining approaches. They have explained the major cause of the thyroid disease and have also given a description of Decision Tree, Naïve Bayes classification, and SVM.

### Sunila Godara et al.:
Have performed a Prediction on Thyroid Disease using various machine learning techniques. They have considered Logistic Regression and Support Vector Machine as the main Thyroid detection models. They have concluded that these two proposed classifier methods are the best when the number of classes increases in the thyroid prediction model.

## Existing System:
In the data collection stage, small, memory-constrained, and low energy-consumption sensors with a short-range communications capability are employed to collect information about the physical environment. Ethernet, WiFi, ZigBee, and wire-based technologies are combined with Transmission Control Protocol/Internet Protocol to connect the objects and users across prolonged distances during data transmission. During the data processing and utilization stage, applications process the data to obtain useful information, and may initiate control commands to act on the physical environment after making decisions based on the collected information. The coordination of diverse technologies, the heterogeneity, and the distributed nature of communications technologies proposed for the IoT by different standards development organizations magnify the threat to end-to-end security in IoT applications.

### Disadvantages of Existing System:
1. Less accuracy
2. Low efficiency

## Proposed System:
The thyroid dataset is taken from Kaggle Machine Learning website. The Database mainly includes the thyroid patient records having all the necessary patient details in it. The patient record has important attributes as mentioned in the Table I. Along with this, the proposed model also takes all the records of the patient’s past clinical history showing in the Fig 1. These include whether the patient is allergic to any particular medicine, whether the patient has undergone any past thyroid surgery and also any recent thyroid test and genetic history of the patient. These also act as the major attributes since they ease the examination of the thyroid patient and reduce the thorough examination by the doctor. This saves time and eases the diagnosis process.

These attributes are stored in a dedicated cloud server which can be made private or hybrid based on the health organization’s need and interest. Among the considered attributes, a train dataset is prepared and is given as the input to the classification-based machine learning model. This is a supervised learning method and the designed model will generate the results based on the train dataset values. The proposed model has Decision tree and Naive Bayes algorithm to generate the results. A decision tree is a tree-based algorithm which follows a top-down approach build. ID3 algorithm is used to construct the decision tree. It mainly eliminates any redundant element if present and improves the accuracy of the classification.

This decision tree algorithm is applied to the thyroid patient’s records consisting of age, gender, T3, T4, TSH values. The decision tree algorithm calculates the inputted values present in the thyroid patient record. The calculation is done based on the train dataset. Therefore more the number of records in the train dataset higher the accuracy of the algorithm. For example, if 3000 train thyroid dataset records are considered and trained to the decision tree algorithm then the generated accuracy rate will be high. Our proposed model has considered more than 3000 train dataset attributes resulting in 95% accuracy rate in the prediction results. The algorithm generates yes or no values i.e., whether the thyroid disease is present in the patient or not. If the patient’s output value results true then further Naïve Bayes algorithm is applied to calculate which stage is the patient currently in. This adds as a major advantage to the healthcare staff in the easy analysis of the thyroid disease and also avoid certain lab tests if not necessary. The patient’s thyroid stage here is divided into 3 stages i.e., minor, major, and critical. The Naïve Bayes algorithm is applied if the Decision Tree returns thyroid true or positive value. The Naïve Bayes algorithm in machine learning is a supervised learning algorithm which is based on Bayes’ Theorem.

It is used for solving classification-based problems. The model can be built fast and it is also cost-effective one. In this way our proposed system can make a major contribution in the healthcare field and also generate positive outcomes with good accuracy with a cost and time saving method for the thyroid patients.

### Advantages of Proposed System:
1. High accuracy
2. High efficiency

## System Architecture:

### Hardware & Software Requirements:

#### Hardware Requirements:
- System: i3 or above.
- Ram: 4 GB.
- Hard Disk: 40 GB

#### Software Requirements:
- Operating system: Windows 8 or Above.
- Coding Language: Python

## Modules:

1. **Upload Thyroid Dataset:** This module allows users to upload dataset details and then the application will read and display dataset values and then find and plot a graph of normal and thyroid patients count.
2. **Preprocess Dataset:** This module preprocesses the dataset by removing missing values, converting all non-numeric data into numeric data, shuffling, and then splitting the dataset into train and test. The application uses 80% of the dataset for training and 20% for testing.
3. **Run Naive Bayes Algorithm:** This module inputs 80% of the dataset to Naïve Bayes to train a model and applies 20% test data on the trained model to calculate prediction accuracy.
4. **Run SVM Algorithm:** This module inputs 80% of the dataset to SVM to train a model and applies 20% test data on the trained model to calculate prediction accuracy.
5. **Run Random Forest Algorithm:** This module inputs 80% of the dataset to Random Forest to train a model and applies 20% test data on the trained model to calculate prediction accuracy.
6. **Comparison Graph:** This module plots accuracy comparison graphs between all algorithms.
7. **Predict Disease from Test Data:** This module uploads test data and then random forest will predict whether the test data is normal or contains thyroid disease.

## Conclusion:

Thus, the proposed work will be very much useful to identify thyroid disease in a patient at an early stage using classification-based machine learning techniques. These algorithms give various levels of precision and accuracy. These methods also aid in decreasing the unwanted redundant data from the patient’s database. The algorithms used in the proposed model are cost-effective and also have good output performance and speed. These classification methods make the treatment of the thyroid patient simple by reducing further complex procedures with an affordable price.
