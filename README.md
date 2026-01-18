# Belle Weder
Last Updated: 13/01/2026

## TL;DR
[Slides](https://docs.google.com/presentation/d/16sxZOPwjahf1FJiZXZWMHOnjbC8INxCv/edit?slide=id.p1#slide=id.p1)

After experiencing dangerous weather in rural Cameroon with no mobile service, this project explores offline, hyperlocal weather prediction using computer vision.

Instead of relying on heavy physics-based forecasts, it looks at predicting local weather by simply identifying cloud types from sky pics.

My early experiments use the CCSN cloud image dataset and lightweight machine-learning models to run on modest hardware. So far, performance is low (~50% accuracy score) using transfer learning, mainly due to limited data, low contrast in cloud images vs objects, and limited compute.

Next, the focus is on improving transfer learning results with cloud computing, dataset expansion, and data augmentation to eventually deliver accurate forecasts on any low-cost device.

## Predict the Weather at the Click of a Shutter
When I travelled back to Cameroon this summer to see my family, I visited my family village. It was a 5-hour drive from Yaounde, where we were staying.

After the visit, on our way back from the village, we observed dark-ish clouds on the horizon. However, we weren’t sure if they were rain clouds, and we weren’t sure if they would hit us. (PS: Yes, they were rain clouds. Yes, they did hit us - big time.)

In the moment, we couldn’t check the weather due to the lack of service in the region. Thankfully, we were fine most of the way, apart from two tire punctures, but other travellers weren’t as lucky. We observed mudslides, flooding, mini-buses being pushed in the hills, people standed on the side of the road, ...

Hence the idea for this project. It aims to predict weather at a hyperlocal scale with computer vision, with no need for service.

###  How do we get our weather forecasts in the first place? Physics!

The way meteorological institutions like the UK Met Office or the US National Center for Atmospheric Research get us our weather forecasts is through the use of complex physics formulas (Numerical Weather Prediction - NWP) that are run through supercomputers.

These organisations receive an insane amount of current atmospheric data (temperature, humidity, air pressure, and much more). From this data, they can forecast weather from a few hours in advance to hundreds of years. These calculations are run multiple times a day, the forecasts are continuously monitored and updated to validate accuracy, as new data comes is recorded.

### Is Machine Learning used at all?

The standard NWP methods we use are very resource intensive (time and computation), and are still prone to errors. ML has been used in the field but has usually come out with blurry forecasts (too vague) which haven’t been able to rival classic NWP forecasts. That is, until Google's GenCast. GenCast is an AI Based weather prediction model trained on 40 years of historical weather data. It can predict weather up to 15 days in advance with the same or higher accuracy than physics based forecasts.

### Going Hyperlocal by Identifying Clouds
Google GenCast reaches a prediction resolution of ~ 28 x 28 km. Global NWP models reach a resolution of ~ 1 - 4 km^2.

Is there a way to bridge that gap? Well, why not get hyperlocal weather predictions with Machine Learning models through computer vision. Imagine this:
- You take a picture of the sky
- The cloud formation is identified
- That cloud formation data is combined with current atmospheric conditions  (wind speed, temperature, pressure, humidity) - either hyperlocal or taken from open sources
- Predict the weather

Note: With this image data, prediction resolution could improve to a hopeful <4 km^2.

### Applications of the Project
- Better weather predictions, especially in regions where localised weather effects are prevalent - i.e. mountains, regions where there is sparse data due to lack of ground instrumentation (i.e. Arctic, oceans, deserts, developing countries, …)
- Weather prediction in and by the sea - extremely hard to do forecasting there
- Commodities trading (e.g. energy and agricultural products) as the result of the above

*What's the ideal end state of this project?*
- Create a consumer application that users can use to identify clouds and get weather predictions. For this, the model would ideally need:
    - High accuracy >90%
    - Low hardware requirements (i.e. low resolution camera, low power processing unit, etc) - so that the model can run on the worst performing devices in use in Africa
- Create a standalone device that can be attached to urban and in nature furniture and infrastructure (street poles, watchtowers, buoys in the sea, boat masts, etc…) to continuously take measures and make predictions.

## So, what have I wrought so far?
### Computer Vision
*Chose a starting dataset*
For the computer vision side of this project, we’re using the CCSN dataset of cloud images.

The CCSN dataset contains 2543 cloud images divided into 12 categories: Ac, Sc, Ns, Cu, Ci, Cc, Cb, As, Ct, Cs, St. Ci = cirrus; Cs = cirrostratus; Cc = cirrocumulus; Ac = altocumulus; As = altostratus; Cu = cumulus; Cb = cumulonimbus; Ns = nimbostratus; Sc = stratocumulus; St = stratus; Ct = contrail. All images are fixed resolution 256×256 pixels with the JPEG format.

![CCSN Dataset Label Distributions](resources/figures/ccsn_label_distributions.png)

*Why this dataset?*
Different types of clouds can give us clues about what is happening in the atmosphere – and what we can expect the weather to do.

Clouds such as Cumulus, Altocumulus, and Cirrocumulus forecast fair weather.

On the other hand, clouds like Altostratus and Nimbostratus are signs of continuous rain or snow.

And Cumulonimbus signal thunderstorms, hail, and tornados.

*I tested out a few different models*
Trained 10 of the key `scikit-learn` models (e.g. Random Forest, Gaussian Naive Bayes, Logistic Regression, Support Vector Classification, K Nearest Neighbours) on the dataset, and chose to explore the most promising ones based on score and time_elapsed.

```python
evaluate_classifiers(X, y, scoring: str = "f1_macro", cv: int = 3, n_jobs: int = -1, random_state: int = 42, verbose: int = 1 )
```

![CCSN Classifiers Evaluations](resources/figures/ccsn_evaluate_classifiers_result.png)

In this case, I chose to explore random forest (and mlp further).

Random Forest Classification Report:
![Random Forest Test Classification Report](resources/figures/rf_clf_report_test_set.png)

The reason for the lack of accuracy was most likely due to:
- Small dataset - barely a few hundred samples per class (MNIST datasets which are the benchmark for many Computer vision models have ~ 6K images per class)
- Low resolution images - though there’s a big tradeoff with training time when raising image resolution raises the training time
- Hard to discern edges and lack of contrast in the images

Following this, I moved on to exploring transfer learning using `vit_b_16` and `convnext`. This is because transfer learning reduces required dataset size, and improves generisability by reusing pretrained model layers, effectively also reducing computation costs.

See `notebooks/01_machine_learning_clean.ipynb` where I employed [`convnext`](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_small.html), and `notebooks/02_transfer_learning_5.ipynb` where I employed [`vit_b_16`](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html).

### Results
Managed to reach an accuracy of ~0.50 (using `vit_b_16` on pytorch) after 15 epochs, only modifying and training the output layer of the model.

### Discussion
The accuracy of 0.50 (using `vit_b_16` on pytorch) is a huge improvement on the 0.30 accuracy from before.

However, before applying transfer learning to the cloud images, I applied it to the *[102 Category Flower Dataset](https://docs.pytorch.org/vision/0.17/generated/torchvision.datasets.Flowers102.html)*. The model reached an accuracy of ~0.90 after only 5 epochs?!
What could this difference in accuracy be due to?
  - It's harder to identify the underlying patterns in clouds vs flowers. ImageNet, which is the dataset the pre-trained model I employed was trained on contains 1000 object classes (ranging from fish and birds to clothes and everyday objects). These objects can be split into edges and shapes (what the lower levels of CNNs learns), which can be easily combined into more complex features and object (middle layers), which can then be combined into fully detailed objects (high layers). It was likely easier to train on the flowers because they're similar to the ImageNet objects, which is not at all the case for clouds.
  ![CNN learned features byt Layer](resources/figures/cnn_learning_facial_features.png)
- How do we move forward? There are 3 ways we can move forward from this:
  1. So far w've been freezing the weights of all layers and only unfreezing the head and penultimate layer. One suggestion would be to gradually unfreeze lower levels of the CNN to train on. I have tried to this locally but I've encountered long processing times and execution errors. Solution here would be to employ a cloud provider such as **AWS**.
  2. More training data. As mentioned in the previous update, I've found a database of cloud images. Expanding the dataset with those images could have an impact on the accuracy here.
  3. Improving the contrast and sharpening the edges of the images before training.

### Next Steps
My main focus at the moment, is to build a functioning (>90% accuracy) model. With the small size of the dataset, transfer learning is likely the best solution to get a well functioning model ASAP. Therefore these are the next steps I’ll be taking:

1. Employing cloud computing (AWS)
2. Expanding the CCSN dataset:
    1. Adding the NASA data
    2. Data Augmentation
    3. Improving image contrast
3. (Training a CNN from scratch)

## Updates

UPDATE (20/12/2025):
- Discovered that NASA JPL has an earth science project called GLOBE Observer. It has volunteers collect data and make observations using their smartphones. The data is then available through an open dataset. The open dataset has 1 million + weather entries with the type of clouds observed. At a glance, I would estimate that 10% of them have images linked to them. I’ll be working on extracting all of the images with their classes and adding them to their respective folders. While looking at the dataset, I also noticed that some classes tend to be combined/appear together in the atmosphere. It might be useful to also combine these into single classes or employ multi-label classification.

UPDATE (05/01/2025):
- Employed transfer learning on some pre-trained computer vision models (which use CNNs). Managed to reach an accuracy of ~0.50 (using `vit_b_16` on pytorch) which is a huge improvement on the 0.30 accuracy from before. Before applying transfer learning to the cloud images, I applied it to the *102 Category Flower Dataset*. The model reached an accuracy of ~0.90 after only 5 epochs?!
What could this difference in accuracy be due to?
  - It's harder to identify the underlying patterns in clouds vs flowers. ImageNet, which is the dataset the pre-trained model I employed was trained on contains 1000 object classes (ranging from fish and birds to clothes and everyday objects). These objects can be split into edges and shapes (what the lower levels of CNNs learns), which can be easily combined into more complex features and object (middle layers), which can then be combined into fully detailed objects (high layers). It was likely easier to train on the flowers because they're similar to the ImageNet objects, which is not at all the case for clouds.
  ![CNN learned features byt Layer](resources/figures/cnn_learning_facial_features.png)
- How do we move forward? There are 3 ways we can move forward from this:
  1. So far I've been freezing the weights of all layers and only unfreezing the head (the final, task-specific layers - the fully connected (FC) layers and the output layer). One suggestion would be to gradually unfreeze lower levels of the CNN to train on. I have tried to this locally but I've encountered long processing times and execution errors. Solution here would be to employ a cloud provider such as AWS.
  2. Improving the contrast and sharpening the edges of the images before training.
  3. More training data. As mentioned in the previous update, I've found a database of cloud images. Expanding the dataset with those images could have an impact on the accuracy here.

# Appendix
### Random Forest Discussion
The confusion matrices for Random Forest:
![Confusion Matrix 1](resources/figures/rf_clf_confusion_matrix.png)
<small><small>Raw and normalised</small></small>
![Confusion Matrix 1](resources/figures/zeroed_ccsn_evaluate_classifiers_result.png)
<small><small>Raw and normalised with zeroed diagonal</small></small>

Oof, that's absolutely dismal performance.

Let's look at some specific errors such as such as Ac vs Cc:
![Ac vs Cc](resources/figures/ac_vs_cc.png)
