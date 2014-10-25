ExtractData <- function(filename)
{
  x <- read.csv(filename)
  x <- x[which(x$kurtosis_roll_belt == "" | is.na(x$kurtosis_roll_belt)), ]
  
  #It's definitely debatable if user name should be here
  goodRows <- c(#"user_name",
                "roll_belt",
                "pitch_belt",
                "yaw_belt",
                "total_accel_belt",
                "gyros_belt_x",
                "gyros_belt_y",
                "gyros_belt_z",
                "accel_belt_x",
                "accel_belt_y",
                "accel_belt_z",
                "magnet_belt_x",
                "magnet_belt_y",
                "magnet_belt_z",
                "roll_arm",
                "pitch_arm",
                "yaw_arm",
                "total_accel_arm",
                "gyros_arm_x",
                "gyros_arm_y",
                "gyros_arm_z",
                "accel_arm_x",
                "accel_arm_y",
                "accel_arm_z",
                "magnet_arm_x",
                "magnet_arm_y",
                "magnet_arm_z",
                "roll_dumbbell",
                "pitch_dumbbell",
                "yaw_dumbbell",
                "classe")
  
  return(x[, names(x) %in% goodRows]);
}

library(caret)
#Taken From http://topepo.github.io/caret/similarity.html
FindMaxDisimmilar <- function(numModels)
{
  tag <- read.csv("tag_data.csv", row.names = 1)
  tag <- as.matrix(tag)
  
  ## Select only models for classification
  classModels <- tag[tag[,"Classification"] == 1,]
  
  all <- 1:nrow(classModels)
  ## Seed the analysis with the SVM model
  start <- grep("(rf)", rownames(classModels), fixed = TRUE)
  pool <- all[all != start]
  
  ## Select 4 model models by maximizing the Jaccard
  ## dissimilarity between sets of models
  nextMods <- maxDissim(classModels[start,,drop = FALSE],
                        classModels[pool, ],
                        method = "Jaccard",
                        n = numModels)
  
  rownames(classModels)[c(start, nextMods)]
};

#result:
#[1] "Random Forest (rf)"                                                       
#[2] "Penalized Multinomial Regression (multinom)"                              
#[3] "Support Vector Machines with Radial Basis Function Kernel (svmRadialCost)"
#[4] "Stepwise Diagonal Linear Discriminant Analysis (sddaLDA)"                 
#[5] "SIMCA (CSimca)"                                                           
#[6] "Partial Least Squares (kernelpls)"     

library(caret)

#Look at Correlations & Linear Combinations
correlations <- cor(data[, 2:ncol(data)]);
sum(abs(correlations[upper.tri(correlations)]) > 0.999); # 0 is good!
findLinearCombos(cor(data[, 2:ncol(data)])); #none!

#LoadData
data <- ExtractData('pml-training.csv')
set.seed(1234)
inTrain = createDataPartition(data$classe, p = 3/4)[[1]]
training = data[ inTrain,]
testing = data[-inTrain,]

#We're going to smooth because RF is overfitting

#Find 5 models most dissimilar to Rnadom Forests
FindMaxDisimmilar(5)


set.seed(1234)

#sddaLDA.Fit <- train(classe ~ ., data=training, method="sddaLDA");

#
rf.pca.cv.Fit <- train(classe ~ ., data=training, method="rf", preProcess=c("pca"), trControl=trainControl(method="cv"));
multinom.pca.cv.Fit <- train(classe ~ ., data=training, method="multinom", preProcess=c("pca"), trControl=trainControl(method="cv"));
svmRadialCost.pca.cv.Fit <- train(classe ~ ., data=training, method="svmRadialCost", preProcess=c("pca"), trControl=trainControl(method="cv"));
SIMCA.pca.cv.Fit <- train(classe ~ ., data=training, method="CSimca", preProcess=c("pca"), trControl=trainControl(method="cv"));
kernelpls.pca.cv.Fit <- train(classe ~ ., data=training, method="kernelpls", preProcess=c("pca"), trControl=trainControl(method="cv"));

rf.cv.Fit <- train(classe ~ ., data=training, method="rf", trControl=trainControl(method="cv"));
multinom.cv.Fit <- train(classe ~ ., data=training, method="multinom", trControl=trainControl(method="cv"));
svmRadialCost.cv.Fit <- train(classe ~ ., data=training, method="svmRadialCost", trControl=trainControl(method="cv"));
SIMCA.cv.Fit <- train(classe ~ ., data=training, method="CSimca", preProcess=c("pca"), trControl=trainControl(method="cv"));
kernelpls.cv.Fit <- train(classe ~ ., data=training, method="kernelpls", preProcess=c("pca"), trControl=trainControl(method="cv"));

rf.pca.Fit <- train(classe ~ ., data=training, method="rf", preProcess=c("pca"));
multinom.pca.Fit <- train(classe ~ ., data=training, method="multinom", preProcess=c("pca"));
svmRadialCost.pca.Fit <- train(classe ~ ., data=training, method="svmRadialCost", preProcess=c("pca"));
SIMCA.pca.Fit <- train(classe ~ ., data=training, method="CSimca", preProcess=c("pca"));
kernelpls.pca.Fit <- train(classe ~ ., data=training, method="kernelpls", preProcess=c("pca"));

rf.Fit <- train(classe ~ ., data=training, method="rf");
multinom.Fit <- train(classe ~ ., data=training, method="multinom");
svmRadialCost.Fit <- train(classe ~ ., data=training, method="svmRadialCost");
SIMCA.Fit <- train(classe ~ ., data=training, method="CSimca");
kernelpls.Fit <- train(classe ~ ., data=training, method="kernelpls");


accuracy <- function(model, data)
{
  numCorrect <- predict(model, newdata=data) == data$classe;
  return(sum(numCorrect) / nrow(data));
}

massPredict <- function(modelNames, newData)
{
  sapply(modelNames, function(modelName) { fit <- eval(parse(text=modelName)); accuracy(fit, newData) });
}

models.fit <- c(rf.pca.cv.Fit,multinom.pca.cv.Fit,svmRadialCost.pca.cv.Fit,SIMCA.pca.cv.Fit,kernelpls.pca.cv.Fit,rf.cv.Fit,multinom.cv.Fit,svmRadialCost.cv.Fit,SIMCA.cv.Fit,kernelpls.cv.Fit,rf.pca.Fit,multinom.pca.Fit,svmRadialCost.pca.Fit,SIMCA.pca.Fit,kernelpls.pca.Fit,rf.Fit,multinom.Fit,svmRadialCost.Fit,SIMCA.Fit,kernelpls.Fit)
models.names <- c("rf.pca.cv.Fit","multinom.pca.cv.Fit","svmRadialCost.pca.cv.Fit","SIMCA.pca.cv.Fit","kernelpls.pca.cv.Fit","rf.cv.Fit","multinom.cv.Fit","svmRadialCost.cv.Fit","SIMCA.cv.Fit","kernelpls.cv.Fit","rf.pca.Fit","multinom.pca.Fit","svmRadialCost.pca.Fit","SIMCA.pca.Fit","kernelpls.pca.Fit","rf.Fit","multinom.Fit","svmRadialCost.Fit","SIMCA.Fit","kernelpls.Fit")

model.accuracy <- data.frame(names = models.names,
                           insample.accuracy = massPredict(models.names, training),
                           outsample.accuracy = massPredict(models.names, testing));





submission.data <- ExtractData('pml-testing.csv')
submission.output <- predict(rf.Fit, newdata=submission)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(submission.output)







model.accuracy[sort(model.accuracy$names), ]



models.names



stacked.training = data.frame(rf = predict(rf.Fit, newdata=training),
                              multinom = predict(multinom.Fit, newdata=training),
                              svmRadialCost = predict(svmRadialCost.Fit, newdata=training),
                              SIMCA = predict(SIMCA.Fit, newdata=training),
                              kernelpls = predict(kernelpls.Fit, newdata=training),
                              classe = training$classe);

stacked.testing = data.frame(rf = predict(rf.Fit, newdata=testing),
                              multinom = predict(multinom.Fit, newdata=testing),
                              svmRadialCost = predict(svmRadialCost.Fit, newdata=testing),
                              SIMCA = predict(SIMCA.Fit, newdata=testing),
                              kernelpls = predict(kernelpls.Fit, newdata=testing),
                              classe = testing$classe);







stacked.Fit <- train(classe ~ ., data=stacked.training, method="rf")

sum((predict(rf.Fit, newdata=training)) == training$classe)
sum((predict(rf.cv.Fit, newdata=training)) == training$classe)
sum((predict(rf.pca.Fit, newdata=training)) == training$classe)
sum((predict(rf.pca.cv.Fit, newdata=training)) == training$classe)

sum((predict(rf.Fit, newdata=testing)) == testing$classe)
sum((predict(rf.cv.Fit, newdata=testing)) == testing$classe)
sum((predict(rf.pca.Fit, newdata=testing)) == testing$classe)
sum((predict(rf.pca.cv.Fit, newdata=testing)) == testing$classe)

sum((predict(multinom.Fit, newdata=testing)) == testing$classe)
sum((predict(multinom.cv.Fit, newdata=testing)) == testing$classe)
sum((predict(multinom.pca.Fit, newdata=testing)) == testing$classe)
sum((predict(multinom.pca.cv.Fit, newdata=testing)) == testing$classe)

sum((predict(svmRadialCost.Fit, newdata=testing)) == testing$classe)
sum((predict(svmRadialCost.cv.Fit, newdata=testing)) == testing$classe)
sum((predict(svmRadialCost.pca.Fit, newdata=testing)) == testing$classe)
sum((predict(svmRadialCost.pca.cv.Fit, newdata=testing)) == testing$classe)

sum((predict(SIMCA.Fit, newdata=testing)) == testing$classe)
sum((predict(SIMCA.cv.Fit, newdata=testing)) == testing$classe)
sum((predict(SIMCA.pca.Fit, newdata=testing)) == testing$classe)
sum((predict(SIMCA.pca.cv.Fit, newdata=testing)) == testing$classe)


sum((predict(kernelpls.Fit, newdata=testing)) == testing$classe)
sum((predict(kernelpls.cv.Fit, newdata=testing)) == testing$classe)
sum((predict(kernelpls.pca.Fit, newdata=testing)) == testing$classe)
sum((predict(kernelpls.pca.cv.Fit, newdata=testing)) == testing$classe)

results <- 



sum((predict(rf.Fit, newdata=testing))==testing$classe)
sum((predict(stacked.Fit, newdata=stacked.testing)) == stacked.testing$classe)

nrow(testing)
