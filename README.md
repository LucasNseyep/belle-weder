# Belle Weder Vision
## Why?
When I travelled back to Cameroon this summer to see my family, I visited a village, a 5-hour drive from Yaounde, where we were staying. On our way back from the village, we observed dark-ish clouds on the horizon. However, we weren’t sure if they were rain clouds, and we weren’t sure if they would hit us. (PS: Yes, they were. Yes, they did hit us - big time.) 

At the time, we couldn’t check the weather due to the lack of service in the region. We could’ve checked the forecast, but then again, the predictions aren’t always helpful at a hyperlocal scale. Thankfully, we were fine most of the way, apart from a tire puncture, but we did see mudslides and mini-buses being pushed in the hills.

Hence, this project. It aims to predict weather at a hyperlocal scale with computer vision, with no need for service.

## Progress so far
- Through SVM, we’re able to distinguish between clouds and clear skies at a ~90% accuracy!
- We aren’t completely able to distinguish the different [types of atmospheric clouds](https://www.noaa.gov/jetstream/clouds/ten-basic-clouds) ([here](https://weather.metoffice.gov.uk/learn-about/weather/types-of-weather/clouds/cloud-names-classifications) too). We’re at ~20% accuracy so far.

## Next steps
- Optimising the current models through hyperparameters
- Exploring other machine learning models
- Grouping the different types of clouds into categories, which we can use to make a first classification, before finally classifying by type (logic tree) - suggestion: by atmospheric levels: high, medium, low.

## Resources:
- [CCSN databse](https://www.kaggle.com/datasets/mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database)
