
# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation

The project done on AWS sagemaker using startr files provided from Udacity.the data downloaded from the link provided on Udacity throuh the train and deploy file.

## Dataset
The data set choosen to complete this project was the dog breed classification dataset provided from Udaicty. The dataset contains images from 133 dog breeds divided into training, testing and validation datasets.

### Access
The data uploaded to S3 bucket and sagemaker have access to it.

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

The model that I choosed was ResNet18 , because first , I'm little bit familiar with it from the lessons, how to train with it and it's hyperparameters. with trials and errors and keep working on the training script until it works. the resources was not such enough to try somethings new from the scratch , but for sure in the future will try more and will get much deeper in other models.The hyperparameters I choosed is: 
1-Learning rate range (0.001, 0.1).
Learning rate controls how quickly or slowly a neural network model learns a problem and controls the amount of apportioned error that the weights of the model are updated with each time they are updated. So when it comes to find the best lr will directly affects the preformance of the model. 
2-Batch size range (32, 64, 128, 256).
The number of training examples in one forward/backward pass. trying to set range between 32 and 256 to make sure the batch size is not too low or too large. trying to make balance between the training process speed and he model performance.
I wanted to include more especially epoch , and to increase the training jobs. but I was afraid from the cost specially this kind of training instance is not cheap.



- Include a screenshot of completed training jobs

![4training jop](https://user-images.githubusercontent.com/37002455/195200609-48ad4332-e363-42a8-bed4-9e927accdf20.jpg)
![4training jop2](https://user-images.githubusercontent.com/37002455/195200617-868a3bd7-e4a9-48ef-8c6b-7b902c0fc1f9.png)


- Logs metrics during the training process
![log1](https://user-images.githubusercontent.com/37002455/195200273-a393098b-98d0-4481-9d94-d69adceceaec.jpg)
![log2](https://user-images.githubusercontent.com/37002455/195200476-4fab2b63-591a-497d-bd85-e60777e55e47.jpg)
![log3](https://user-images.githubusercontent.com/37002455/195200502-7a633519-5284-44b6-804d-eb6471ada248.jpg)


## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
