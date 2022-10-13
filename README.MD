
# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation

The project done on AWS sagemaker using startr files provided from Udacity.the data downloaded from the link provided by Udacity( train and deploy ), hpo.py , train_model.py files.

## Dataset
The data set choosen to complete this project was the dog breed classification dataset provided by Udaicty. The dataset contains images from 133 dog breeds divided into training, testing and validation datasets.

### Access
The data uploaded to S3 bucket and sagemaker have access to it.

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

The model that I choosed was ResNet18 , because first , I'm little bit familiar with it from the lessons, how to train with it and it's hyperparameters. with trials , errors and keep working on the training script until it works. the resources was not such enough to try somethings new from the scratch , but for sure in the future will try more and will get much deeper in other models.The hyperparameters I choosed is: 
1-Learning rate range (0.001, 0.1).
Learning rate controls how quickly or slowly a neural network model learns a problem and controls the amount of apportioned error that the weights of the model are updated with each time they are updated. So when it comes to find the best lr will directly affects the preformance of the model. 
2-Batch size range (32, 64, 128, 256).
The number of training examples in one forward/backward pass. trying to set the range between 32 and 256 to make sure the batch size is not too low or too large to attempt to make balance between the training process speed and he model performance.
I wanted to include more hyperparameters especially epoch , and to increase the training jobs. but I was afraid from the cost specially this kind of training instance is not cheap.



- Include a screenshot of completed training jobs

![4training jop](https://user-images.githubusercontent.com/37002455/195200609-48ad4332-e363-42a8-bed4-9e927accdf20.jpg)
![4training jop2](https://user-images.githubusercontent.com/37002455/195200617-868a3bd7-e4a9-48ef-8c6b-7b902c0fc1f9.png)


- Logs metrics during the training process
![log1](https://user-images.githubusercontent.com/37002455/195200273-a393098b-98d0-4481-9d94-d69adceceaec.jpg)
![log2](https://user-images.githubusercontent.com/37002455/195200476-4fab2b63-591a-497d-bd85-e60777e55e47.jpg)
![log3](https://user-images.githubusercontent.com/37002455/195200502-7a633519-5284-44b6-804d-eb6471ada248.jpg)


## Debugging and Profiling
As in lesson 4, After importing debuger library, setup hooks in train, test function then configured the debuger rules Vanishing Gradients to track the gradients of the loss function if it's going too low. Overfitting to avoid make the model performing too well on the training set but poorly on new data. Overtraining to check if the results in poor generalization and Poor weight initialization.


### Results
There were some issues detected in the profiler, but mostly around increasing batch size and check if there is a bottleneck. Even myself during the training process I realized that increasing batch size the model preform better, I think as a best practice to try to make a range for epoch also as hyperparameter, but I didn't do that because increasing training time and the resources is limited.


![summery](https://user-images.githubusercontent.com/37002455/195224223-b2d78563-d87f-4f20-a11d-b69a3e1d9665.jpg)


## Model Deployment
I made a script has 3 functions that required to deploy a PyTorch model called infrances2.py. model_fn, input_fn, predict_fn this functions will take an image input and preform the same transforms as hpo and train_model scripts. After that comes the deployment of the endpoint.

Then I coded a call function code to triggers the endpoint to make a prediction. 

![endpoint](https://user-images.githubusercontent.com/37002455/195225981-1fd91b45-d35a-44bc-9575-5126faa2964c.jpg)


