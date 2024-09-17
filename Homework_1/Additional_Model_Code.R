############################################
# Additional Model code for Homework 1 
# this will serve as the code for the SCAD and MC+ penalty models, and we will use 
# some type of common score to compare the models fairly 
############################################ 

# loading the necessary libraries 
library("grpreg")
library("lars")
library("ncvreg")

# loading the data 
X <- read.csv("/home/adbucks/Documents/sta_629/Homework_1/X_SRBCT.csv", header = FALSE)
y <- read.csv("/home/adbucks/Documents/sta_629/Homework_1/Y_SRBCT.csv", header = FALSE) 
head(X)
dim(X)
dim(y)

data("Prostate")
class(Prostate$X)
class(Prostate$y)

# comparing 
class(X) # need to convert to matrix array 

X <- as.matrix(X)
class(y) # need to convert to numeric
y <- y$V1


# now we can do the actual modeling with the data loaded and get to plotting
# and eventually scoring 
# using another package to avoid the grouped covariates problem 
# need to coerce y into a numeric 
# trying another way 

mc_model2 <- ncvreg(X, y) # MCP default 
# maybe we're having the transpose problem again

plot(mc_model2, type = "scale", vertical.line = TRUE)

summary(mc_model2)

# it appears we have many 0 coefficients and a sparse model, which, when 
# selecting specific alpha, allows us to see which variables are most 
# strongly correlated with the outcome variable

# we can now try this with the SCAD penalty to observe any differences
mc_model_scad <- ncvreg(X, y, penalty = "SCAD")

# can now observe any differences 
plot(mc_model_scad, type = "scale")

summary(mc_model_scad)

# we can try cross-validating to get around the lambda issue 
cvfit <- cv.ncvreg(X, y)
summary(cvfit)

cvfit_scad <- cv.ncvreg(X, y, penalty = "SCAD")
summary(cvfit_scad)


